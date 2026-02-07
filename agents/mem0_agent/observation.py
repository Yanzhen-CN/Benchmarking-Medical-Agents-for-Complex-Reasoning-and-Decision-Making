from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm_providers import LLMProvider
from .prompts import OBSERVATION_EXTRACT_SYSTEM


class ObservationExtractor:
    """Abstract observation extractor."""

    def extract(self, text: str) -> List[str]:
        raise NotImplementedError


class LLMObservationExtractor(ObservationExtractor):
    """
    Extracts compact factual observations from dialogue using an LLM.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def extract(self, text: str) -> List[str]:
        raw = self._llm.chat_json(system_prompt=OBSERVATION_EXTRACT_SYSTEM, user_text=text)
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        if isinstance(raw, dict) and "observations" in raw:
            obs = raw.get("observations", [])
            if isinstance(obs, list):
                return [str(x).strip() for x in obs if str(x).strip()]
        # Fallback: try parsing if model didn't follow format
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
        return []
