from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import LLMConfig
from util.llmUtil import LLMUtil


class LLMProvider:
    """Abstract LLM provider interface."""

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError

    def chat_json(self, system_prompt: str, user_text: str, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    def chat_json_ctx(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        raise NotImplementedError


class OpenAICompatibleLLMProvider(LLMProvider):
    """
    LLM provider using the project's OpenAI-compatible client (LLMUtil).
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self._config = config or LLMConfig()
        self._llm = LLMUtil()

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        model = kwargs.pop("model", self._config.chat_model)
        temperature = kwargs.pop("temperature", self._config.temperature)
        top_p = kwargs.pop("top_p", self._config.top_p)
        max_tokens = kwargs.pop("max_tokens", self._config.max_tokens)
        return self._llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra=kwargs or None,
        )

    def chat_json(self, system_prompt: str, user_text: str, **kwargs: Any) -> Any:
        model = kwargs.pop("model", self._config.chat_model)
        temperature = kwargs.pop("temperature", 0.0)
        return self._llm.chat_json(
            system_prompt=system_prompt,
            user_text=user_text,
            model=model,
            temperature=temperature,
        )
    
    def chat_json_ctx(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        model = kwargs.pop("model", self._config.chat_model)
        temperature = kwargs.pop("temperature", 0.0)
        return self._llm.chat_json_ctx(
            messages=messages,
            model=model,
            temperature=temperature,
        )
        
    def get_token_usage(self) -> Dict[str, int]:
        return self._llm.get_token_usage()
    
    def get_client(self) -> LLMUtil:
        return self._llm
