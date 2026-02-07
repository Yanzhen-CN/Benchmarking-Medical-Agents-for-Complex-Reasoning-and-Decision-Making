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
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


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
    ) -> str:
        if not messages:
            raise ValueError("messages cannot be empty")

        # Build retrieval query from the latest user message (fallback to last turn).
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        query = (last_user or messages[-1]).get("content", "").strip()

        memories: List[MemorySearchResult] = []
        if self._config.include_memory_in_prompt and query and self._should_retrieve(query):
            memories = self._memory.search(
                query=query,
                top_k=self._config.memory_top_k,
                user_id=user_id,
                agent_id=agent_id,
                app_id=app_id,
                run_id=run_id,
            )

        prompt_messages = self._compose_prompt(messages, memories)
        reply = self._llm.chat(prompt_messages)

        self._write_memory(
            messages=messages,
            reply=reply,
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            run_id=run_id,
        )
        return reply

    def _should_retrieve(self, query: str) -> bool:
        policy = (self._config.retrieval_policy or "always").lower()
        if policy == "never":
            return False
        if policy == "always":
            return True
        if policy == "question_only":
            q = query.strip()
            if not q:
                return False
            if "?" in q or "？" in q:
                return True
            # Simple heuristic for question intent (English + Chinese)
            cues = [
                "what", "why", "how", "when", "where", "which", "who",
                "什么", "为啥", "为什么", "怎么", "如何", "是否", "吗",
                "几", "多少", "哪", "哪种", "哪位", "多大", "多久", "何时", "请问",
            ]
            q_lower = q.lower()
            return any(cue in q_lower for cue in cues)
        # Fallback to safe default
        return True

    def _compose_prompt(
        self, messages: List[Dict[str, str]], memories: List[MemorySearchResult]
    ) -> List[Dict[str, str]]:
        prompt: List[Dict[str, str]] = [{"role": "system", "content": self._config.system_prompt}]

        if memories:
            mem_lines = [f"- {m.text}" for m in memories if m.text]
            mem_block = f"{MEMORY_BLOCK_HEADER}:\n" + "\n".join(mem_lines)
            prompt.append({"role": "system", "content": mem_block})

        if self._config.max_recent_turns > 0:
            prompt.extend(messages[-self._config.max_recent_turns :])
        else:
            prompt.extend(messages)

        return prompt

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
