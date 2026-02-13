# retrieval/retriever.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss

from util.llmUtil import LLMUtil
from util.logUtil import setup_logger

logger = setup_logger()


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def _parse_timestamp(ts: Any) -> Optional[str]:
    """
    你的 timestamp 目前大概率是 "YYYY-MM-DD HH:MM:SS" 或 None。
    这里为了简单，直接按字符串比较（ISO-like 格式可用）。
    如果你 timestamp 可能是别的格式，建议统一成 ISO 再比。
    """
    if ts is None:
        return None
    s = str(ts).strip()
    return s if s else None


@dataclass
class RetrievedDoc:
    score: float
    text: str
    meta: Dict[str, Any]

from config import AgentTaskConfig
cfg = AgentTaskConfig()

class PatientRetriever:
    """
    全量索引检索 + 运行时过滤可视范围
    """

    def __init__(self) -> None:
        self.root = Path(cfg.VECTOR_STORE_DIR)
        self.emb_model = cfg.EMBEDDING_MODEL
        self.normalize = cfg.NORMALIZE_EMBEDDINGS
        self.cache_indexes = cfg.CACHE_INDEXES
        self.llm = LLMUtil()

        # cache: (patient_id, memory_type) -> (index, docs)
        self._cache: Dict[Tuple[str, str], Tuple[faiss.Index, List[Dict[str, Any]]]] = {}

    # -----------------------------
    # loading
    # -----------------------------
    def _load_store(self, patient_id: str, memory_type: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        key = (patient_id, memory_type)
        if self.cache_indexes and key in self._cache:
            return self._cache[key]

        p = self.root / patient_id / memory_type
        if not p.exists():
            raise FileNotFoundError(f"store not found: {p}")

        index_path = p / "index.faiss"
        docs_path = p / "docs.jsonl"

        index = faiss.read_index(str(index_path))
        docs: List[Dict[str, Any]] = []
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(json.loads(line))

        if self.cache_indexes:
            self._cache[key] = (index, docs)
        return index, docs

    # -----------------------------
    # visibility filters
    # -----------------------------
    def _resolve_cutoff_from_event_id(
        self,
        docs: List[Dict[str, Any]],
        cutoff_event_id: str,
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        从 docs 里找到 cutoff_event_id 对应的 visit_idx / timestamp。
        返回 (visit_idx, timestamp_str)；找不到则 (None, None)
        """
        for d in docs:
            meta = d.get("meta") or {}
            if str(meta.get("event_id")) == str(cutoff_event_id):
                vi = _safe_int(meta.get("visit_idx"), default=-1)
                ts = _parse_timestamp(meta.get("timestamp"))
                return (vi if vi >= 0 else None, ts)
        return (None, None)

    def _is_visible(
        self,
        meta: Dict[str, Any],
        *,
        visible_until_visit_idx: Optional[int] = None,
        visible_until_timestamp: Optional[str] = None,
        cutoff_visit_idx: Optional[int] = None,
        cutoff_timestamp: Optional[str] = None,
        include_cutoff: bool = False,
    ) -> bool:
        """
        可见性规则（你可以按需要调整）：

        1) visible_until_visit_idx: 只允许 meta.visit_idx < visible_until_visit_idx
        2) visible_until_timestamp: 只允许 meta.timestamp <= visible_until_timestamp
        3) cutoff_event_id 解析出的 cutoff_visit_idx / cutoff_timestamp:
           - 优先用 timestamp（更细）
           - 如果没有 timestamp，用 visit_idx
           include_cutoff 控制是否允许等于 cutoff 的那条
        """
        vi = _safe_int(meta.get("visit_idx"), default=-1)
        ts = _parse_timestamp(meta.get("timestamp"))

        # (A) 按 visit_idx 限制：只允许过去的 visits
        if visible_until_visit_idx is not None:
            if vi < 0:
                return False
            if vi >= int(visible_until_visit_idx):
                return False

        # (B) 按 timestamp 限制
        if visible_until_timestamp is not None:
            if ts is None:
                return False
            if ts > str(visible_until_timestamp):
                return False

        # (C) cutoff 限制（更像 online 截断）
        if cutoff_timestamp is not None and ts is not None:
            if include_cutoff:
                if ts > cutoff_timestamp:
                    return False
            else:
                if ts >= cutoff_timestamp:
                    return False
        elif cutoff_visit_idx is not None:
            if vi < 0:
                return False
            if include_cutoff:
                if vi > cutoff_visit_idx:
                    return False
            else:
                if vi >= cutoff_visit_idx:
                    return False

        return True

    # -----------------------------
    # search
    # -----------------------------
    def search(
        self,
        *,
        patient_id: str,
        query: str,
        memory_type: str = "event",  # "event" | "note"
        k: int = 8,
        prefetch_k: int = 200,       # 先取多一些再过滤
        # 可视范围（任选其一或组合）
        visible_until_visit_idx: Optional[int] = None,
        visible_until_timestamp: Optional[str] = None,
        cutoff_event_id: Optional[str] = None,
        include_cutoff: bool = False,
        # 额外开关
        require_timestamp: bool = False,  # 若 True 且 meta 没 timestamp 则过滤掉（event 常用）
    ) -> List[RetrievedDoc]:
        index, docs = self._load_store(patient_id, memory_type)

        # embed query
        qv = self.llm.embed_texts([query], model=self.emb_model, batch_size=1)[0]
        q = np.asarray([qv], dtype=np.float32)
        if self.normalize:
            q = _l2_normalize_rows(q)

        # faiss top
        prefetch_k = max(k, int(prefetch_k))
        scores, ids = index.search(q, prefetch_k)

        cutoff_vi = None
        cutoff_ts = None
        if cutoff_event_id:
            cutoff_vi, cutoff_ts = self._resolve_cutoff_from_event_id(docs, cutoff_event_id)
            if cutoff_vi is None and cutoff_ts is None:
                logger.warning(f"[search] cutoff_event_id not found in docs: {cutoff_event_id}")

        out: List[RetrievedDoc] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0:
                continue
            d = docs[idx]
            meta = d.get("meta") or {}

            if require_timestamp and meta.get("timestamp") is None:
                continue

            if not self._is_visible(
                meta,
                visible_until_visit_idx=visible_until_visit_idx,
                visible_until_timestamp=visible_until_timestamp,
                cutoff_visit_idx=cutoff_vi,
                cutoff_timestamp=cutoff_ts,
                include_cutoff=include_cutoff,
            ):
                continue

            out.append(RetrievedDoc(score=float(score), text=d.get("text", ""), meta=meta))
            if len(out) >= k:
                break

        return out


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    r = PatientRetriever()

    # 假设当前在 V12，visit_idx=11（0-based）
    hits = r.search(
        patient_id="P000001",
        query="病人的肝功能状况如何",
        memory_type="event",
        k=8,
        prefetch_k=300,
        visible_until_visit_idx=11,   # 只看 V12 之前
        # cutoff_event_id="P000001-V11-E0032",  # 也可以用事件 id 截断
        include_cutoff=False,
        require_timestamp=False,
    )
    for h in hits:
        print(h.score, h.meta.get("visit_ref"), h.meta.get("event_type"), h.meta.get("event_id"))
        print(h.text)
        print("----")
