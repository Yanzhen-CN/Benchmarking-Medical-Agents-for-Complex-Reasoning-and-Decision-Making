# util/llmUtil_transformers.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


# =========================
# Token usage (best-effort)
# =========================
@dataclass
class ChatTokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def _to_hulu_conversation(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-style messages to Hulu-Med conversation format (TEXT-ONLY).

    Input message examples:
      {"role": "system", "content": "..."}
      {"role": "user", "content": "..."}
      {"role": "assistant", "content": "..."}  (optional history)

    Output (Hulu-Med):
      [{"role": "...", "content": [{"type":"text","text":"..."}]}, ...]
    """
    conv: List[Dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = m.get("content", "")

        # If content is list/dict (e.g., multimodal blocks), stringify for now.
        text = _safe_str(content)

        conv.append(
            {
                "role": role,
                "content": [{"type": "text", "text": text}],
            }
        )
    return conv


class TransformersLLMUtil:
    """
    Hulu-Med (HF) backend using Transformers + AutoProcessor.

    Keeps your original interface:
      chat(messages=[...], model=..., temperature=..., top_p=..., max_tokens=...)
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        trust_remote_code: bool = True,
        add_system_prompt: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.add_system_prompt = add_system_prompt

        # dtype
        dtype_l = (dtype or "").lower()
        if dtype_l in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype_l in ("fp16", "float16"):
            torch_dtype = torch.float16
        elif dtype_l in ("fp32", "float32"):
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Hulu-Med recommends AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        # tokenizer is inside processor
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        # model
        model_kwargs: Dict[str, Any] = dict(
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,  # warning is fine; keeps compat with your env
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        ).eval()

        # Ensure pad token id
        if self.tokenizer is not None and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.token_usage = ChatTokenUsage()

    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,  # keep for API compatibility; ignored
        temperature: float = 0.6,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,   # extra generate kwargs
        enable_thinking: bool = False,            # maps to use_think in decode
    ) -> str:
        """
        Returns assistant response string.

        Notes:
        - Hulu-Med uses AutoProcessor(conversation=..., add_generation_prompt=True)
        - For text-only, we pass modals=["text"] to generate.
        - enable_thinking=False -> use_think=False (only final answer)
          enable_thinking=True  -> use_think=True  (include thinking)
        """
        conversation = _to_hulu_conversation(messages)

        # Build inputs via processor
        inputs = self.processor(
            conversation=conversation,
            return_tensors="pt",
            add_generation_prompt=True,
            add_system_prompt=self.add_system_prompt,
        )

        # Move tensors to model device
        model_device = getattr(self.model, "device", None)
        inputs = {
            k: (v.to(model_device) if isinstance(v, torch.Tensor) and model_device is not None else v)
            for k, v in inputs.items()
        }

        # prompt token usage (best-effort)
        prompt_tokens = 0
        # Many processors output input_ids for text; if present, count it
        if isinstance(inputs.get("input_ids"), torch.Tensor):
            prompt_tokens = int(inputs["input_ids"].numel())

        # Generate kwargs
        gen_kwargs: Dict[str, Any] = dict(
            do_sample=True,
            temperature=float(temperature),
            use_cache=True,
            modals=["text"],
        )
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)

        # Max new tokens
        gen_kwargs["max_new_tokens"] = int(max_tokens) if max_tokens is not None else 512

        # pad_token_id
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        # allow external overrides
        if extra:
            gen_kwargs.update(extra)

        output_ids = self.model.generate(**inputs, **gen_kwargs)

        # completion token usage (best-effort)
        completion_tokens = 0
        if isinstance(inputs.get("input_ids"), torch.Tensor) and isinstance(output_ids, torch.Tensor):
            completion_tokens = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])
            if completion_tokens < 0:
                completion_tokens = 0

        # Hulu-Med custom decode
        out = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            use_think=bool(enable_thinking),
        )[0].strip()

        # accumulate usage
        self.token_usage.prompt_tokens += prompt_tokens
        self.token_usage.completion_tokens += completion_tokens
        self.token_usage.total_tokens += (prompt_tokens + completion_tokens)

        return out

    def get_token_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.token_usage.prompt_tokens,
            "completion_tokens": self.token_usage.completion_tokens,
            "total_tokens": self.token_usage.total_tokens,
        }

    def reset_token_usage(self) -> None:
        self.token_usage = ChatTokenUsage()


# -----------------------------
# quick test
# -----------------------------
if __name__ == "__main__":
    model_path = os.getenv("HF_MODEL", "/data/xzh/Hulu-Med-7B")
    llm = TransformersLLMUtil(model_path)

    ans = llm.chat(
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": "what's your model?"},
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=256,
        enable_thinking=False,  # True -> include thinking (if model supports)
    )
    print(ans)
    print("usage:", llm.get_token_usage())