from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from mem0 import MemoryClient 

@dataclass
class MemorySearchResult:
    text: str
    score: float
    metadata: Dict[str, Any]


class MemoryProvider:
    """
    Abstract memory provider interface.
    """

    def add_memory(
        self,
        *,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemorySearchResult]:
        raise NotImplementedError

    def delete_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError


class InMemoryProvider(MemoryProvider):
    """
    Minimal in-process memory for smoke tests / offline runs.
    """

    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []

    def add_memory(
        self,
        *,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self._items.append(
            {
                "text": text,
                "metadata": metadata or {},
                "user_id": user_id,
                "agent_id": agent_id,
                "app_id": app_id,
                "run_id": run_id,
            }
        )

    def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemorySearchResult]:
        q = query.lower().strip()
        scored: List[MemorySearchResult] = []
        for item in self._items:
            if user_id is not None and item.get("user_id") != user_id:
                continue
            if agent_id is not None and item.get("agent_id") != agent_id:
                continue
            if app_id is not None and item.get("app_id") != app_id:
                continue
            if run_id is not None and item.get("run_id") != run_id:
                continue
            text = item["text"]
            score = 1.0 if q and q in text.lower() else 0.1
            scored.append(MemorySearchResult(text=text, score=score, metadata=item["metadata"]))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: top_k]

    def delete_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not any([user_id, agent_id, app_id, run_id, filters]):
            raise ValueError("delete_all requires at least one filter")

        kept: List[Dict[str, Any]] = []
        for item in self._items:
            if user_id is not None and item.get("user_id") != user_id:
                kept.append(item)
                continue
            if agent_id is not None and item.get("agent_id") != agent_id:
                kept.append(item)
                continue
            if app_id is not None and item.get("app_id") != app_id:
                kept.append(item)
                continue
            if run_id is not None and item.get("run_id") != run_id:
                kept.append(item)
                continue
            if filters:
                matched = True
                for k, v in filters.items():
                    if item.get(k) != v:
                        matched = False
                        break
                if not matched:
                    kept.append(item)
            # else: drop item
        self._items = kept


class Mem0MemoryProvider(MemoryProvider):
    """
    Mem0-backed memory provider.
    Requires `mem0ai` to be installed and MEM0_API_KEY set.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
            
        self._client = MemoryClient(api_key=api_key)

    def add_memory(
        self,
        *,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        messages = [{"role": "user", "content": text}]
        want_ids = os.getenv("MEM0_PRINT_IDS", "0") == "1"
        force_sync = os.getenv("MEM0_SYNC_WRITE", "1") == "1"
        res = self._client.add(
            messages=messages,
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            run_id=run_id,
            metadata=metadata or {},
            async_mode=False if (force_sync or want_ids) else True,
        )
        if want_ids:
            try:
                mid = None
                if isinstance(res, dict):
                    mid = res.get("memory_id") or res.get("id")
                    if mid is None and isinstance(res.get("results"), list) and res["results"]:
                        first = res["results"][0]
                        if isinstance(first, dict):
                            mid = first.get("memory_id") or first.get("id")
                print(
                    {
                        "memory_id": mid,
                        "user_id": user_id,
                        "run_id": run_id,
                        "app_id": app_id,
                        "agent_id": agent_id,
                        "metadata": metadata or {},
                    }
                )
            except Exception:
                pass

    def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemorySearchResult]:
        # Prefer explicit filters if provided; else scope with OR across entity ids.
        # Mem0 stores memories per-entity; AND across user/agent often returns empty.
        if filters is None:
            ors: List[Dict[str, Any]] = []
            if run_id is not None:
                ors.append({"run_id": run_id})
            if user_id is not None:
                ors.append({"user_id": user_id})
            if agent_id is not None:
                ors.append({"agent_id": agent_id})
            if app_id is not None:
                ors.append({"app_id": app_id})
            filters = {"OR": ors} if ors else None

        threshold = float(os.getenv("MEM0_SEARCH_THRESHOLD", "0.0"))
        keyword_search = os.getenv("MEM0_KEYWORD_SEARCH", "1") == "1"
        res = self._client.search(
            query=query,
            version="v2",
            filters=filters,
            top_k=top_k,
            threshold=threshold,
            keyword_search=keyword_search,
        )
        # SDK may return a list or a dict with "results"
        if isinstance(res, dict):
            items = res.get("results", [])
        else:
            items = res

        out: List[MemorySearchResult] = []
        for item in items:
            if isinstance(item, str):
                out.append(MemorySearchResult(text=item, score=0.0, metadata={}))
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("memory", item.get("text", ""))
            score = float(item.get("score", 0.0))
            meta = item.get("metadata", {}) or {}
            out.append(MemorySearchResult(text=text, score=score, metadata=meta))
        return out

    def delete_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if filters is None:
            filters = {}
            if user_id is not None:
                filters["user_id"] = user_id
            if agent_id is not None:
                filters["agent_id"] = agent_id
            if app_id is not None:
                filters["app_id"] = app_id
            if run_id is not None:
                filters["run_id"] = run_id

        if not filters:
            raise ValueError("delete_all requires at least one filter")

        # Prefer hard delete by memory_id to avoid soft-delete renaming.
        try:
            # Mem0 stores memories per-entity scope; AND-ing multiple entity ids
            # can return empty. Query by a single scope and (optionally) filter
            # by non-entity fields client-side.
            entity_keys = {"run_id", "user_id", "agent_id", "app_id"}
            scope_key = None
            for key in ("run_id", "user_id", "agent_id", "app_id"):
                if key in filters:
                    scope_key = key
                    break
            if scope_key is None:
                scope_key = next(iter(filters.keys()))

            and_filters = {"AND": [{scope_key: filters[scope_key]}]}
            delete_ids: List[Dict[str, str]] = []
            page = 1
            while True:
                res = self._client.get_all(filters=and_filters, page=page, page_size=200, version="v2")
                if isinstance(res, dict):
                    items = res.get("results", [])
                else:
                    items = res
                if not items:
                    break
                for item in items:
                    if isinstance(item, dict):
                        # Apply non-entity filters client-side for correctness.
                        for k, v in filters.items():
                            if k in entity_keys and k != scope_key:
                                continue
                            if item.get(k) != v:
                                break
                        else:
                            mid = item.get("memory_id") or item.get("id")
                            if mid:
                                delete_ids.append({"memory_id": mid})
                page += 1
            if delete_ids:
                self._client.batch_delete(delete_ids)
            else:
                # Fallback: no IDs found, use filtered delete_all
                self._client.delete_all(**filters)
        except Exception:
            # Best-effort only; fallback to filtered delete_all
            try:
                self._client.delete_all(**filters)
            except Exception:
                return
