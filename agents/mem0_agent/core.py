from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm_providers import LLMProvider
from .memory_providers import MemoryProvider, MemorySearchResult
from .observation import ObservationExtractor
from .prompts import (
    DEFAULT_SYSTEM_PROMPT,
    MEMORY_BLOCK_HEADER,
    OBSERVATION_STORE_TAG,
    DIALOG_STORE_TAG,
    QUERY_REWRITE_SYSTEM_PROMPT,
)


@dataclass
class AgentConfig:
    max_recent_turns: int = 6
    memory_top_k: int = 5
    store_dialog: bool = True
    store_observations: bool = True
    include_memory_in_prompt: bool = True
    # Retrieval policy: "always" | "question_only" | "never"
    retrieval_policy: str = "always"
    # Whether to rewrite user questions before retrieval
    query_rewrite: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

@dataclass
class RetrievalTrace:
    original_query: str
    retrieval_query: str
    memories: List[MemorySearchResult]


class MemoryAugmentedChatAgent:
    """
    Black-box chat agent with a pluggable memory backend.
    Input: dialogue turns (messages list). Output: assistant reply.
    """

    def __init__(
        self,
        *,
        llm: LLMProvider,
        memory: MemoryProvider,
        observation_extractor: Optional[ObservationExtractor] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._extractor = observation_extractor
        self._config = config or AgentConfig()

    def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        json: Optional[bool] = False,
        **kwargs
    ) -> str:
        reply, _ = self.chat_with_trace(
            messages=messages,
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            run_id=run_id,
            json=json,
            **kwargs
        )
        return reply

    def chat_with_trace(
        self,
        *,
        messages: List[Dict[str, str]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
        json: Optional[bool] = False,
        **kwargs
    ) -> tuple[str, RetrievalTrace]:
        if not messages:
            raise ValueError("messages cannot be empty")

        # Build retrieval query from the latest user message (fallback to last turn).
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        query = (last_user or messages[-1]).get("content", "").strip()

        retrieval_query = query
        memories: List[MemorySearchResult] = []
        if (
            self._config.include_memory_in_prompt
            and self._config.memory_top_k > 0
            and query
            and self._should_retrieve(query=query, last_user=last_user)
        ):
            if self._config.query_rewrite and self._is_question(query):
                retrieval_query = self._rewrite_retrieval_query(query=query)
            memories = self._memory.search(
                query=retrieval_query,
                top_k=self._config.memory_top_k,
                user_id=user_id,
                agent_id=agent_id,
                app_id=app_id,
                run_id=run_id,
            )

        prompt_messages = self._compose_prompt(messages, memories)
        prompt_messages[0] = messages[0] 
        if not json:
            reply = self._llm.chat(prompt_messages, **kwargs)
        else:
            reply = self._llm.chat_json_ctx(prompt_messages, **kwargs)

        self._write_memory(
            messages=messages,
            reply=reply,
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            run_id=run_id,
        )
        trace = RetrievalTrace(
            original_query=query,
            retrieval_query=retrieval_query,
            memories=memories,
        )
        return reply, trace

    def _should_retrieve(self, *, query: str, last_user: Optional[Dict[str, str]] = None) -> bool:
        # Authoritative gating: if caller provides a semantic type, trust it.
        # Only explicit question type can trigger retrieval.
        if last_user is not None:
            msg_type = (last_user.get("type") or "").strip().lower()
            if msg_type:
                return msg_type == "question"

        policy = (self._config.retrieval_policy or "always").lower()
        if policy == "never":
            return False
        if policy == "always":
            return True
        if policy == "question_only":
            return self._is_question(query)
        # Fallback to safe default
        return True

    def _is_question(self, text: str) -> bool:
        q = text.strip()
        if not q:
            return False
        if "?" in q or "？" in q:
            return True
        cues = [
            "what", "why", "how", "when", "where", "which", "who",
            "什么", "为啥", "为什么", "怎么", "如何", "是否", "吗",
            "几", "多少", "哪", "哪种", "哪位", "多大", "多久", "何时", "请问",
        ]
        q_lower = q.lower()
        return any(cue in q_lower for cue in cues)

    def _rewrite_retrieval_query(self, *, query: str) -> str:
        try:
            rewritten = self._llm.chat(
                [
                    {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
            ).strip()
            return rewritten or query
        except Exception:
            return query

    def _compose_prompt(
        self, messages: List[Dict[str, str]], memories: List[MemorySearchResult], max_recent_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        prompt: List[Dict[str, str]] = [{"role": "system", "content": self._config.system_prompt}]

        if memories:
            mem_lines = [f"- {m.text}" for m in memories if m.text]
            mem_block = f"{MEMORY_BLOCK_HEADER}:\n" + "\n".join(mem_lines)
            prompt.append({"role": "system", "content": mem_block})

        if self._config.max_recent_turns > 0:
            prompt.extend(self._sanitize_messages_for_llm(messages[-self._config.max_recent_turns :]))
        else:
            prompt.extend(self._sanitize_messages_for_llm(messages))

        return prompt

    @staticmethod
    def _sanitize_messages_for_llm(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Drop non-standard message keys (e.g., custom `type`) before LLM API call.
        """
        out: List[Dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            out.append({"role": role, "content": content})
        return out

    def _write_memory(
        self,
        *,
        messages: List[Dict[str, str]],
        reply: str,
        user_id: Optional[str],
        agent_id: Optional[str],
        app_id: Optional[str],
        run_id: Optional[str],
    ) -> None:
        if self._config.store_dialog:
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user is not None:
                self._memory.add_memory(
                    text=last_user.get("content", ""),
                    metadata={"type": DIALOG_STORE_TAG, "role": "user"},
                    user_id=user_id,
                    agent_id=agent_id,
                    app_id=app_id,
                    run_id=run_id,
                )
            self._memory.add_memory(
                text=reply,
                metadata={"type": DIALOG_STORE_TAG, "role": "assistant"},
                user_id=user_id,
                agent_id=agent_id,
                app_id=app_id,
                run_id=run_id,
            )

        if self._config.store_observations and self._extractor is not None:
            dialogue = self._render_dialogue_for_extraction(messages, reply)
            observations = self._extractor.extract(dialogue)
            for obs in observations:
                self._memory.add_memory(
                    text=obs,
                    metadata={"type": OBSERVATION_STORE_TAG},
                    user_id=user_id,
                    agent_id=agent_id,
                    app_id=app_id,
                    run_id=run_id,
                )

    @staticmethod
    def _render_dialogue_for_extraction(messages: List[Dict[str, str]], reply: str) -> str:
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append(f"assistant: {reply}")
        return "\n".join(parts)
