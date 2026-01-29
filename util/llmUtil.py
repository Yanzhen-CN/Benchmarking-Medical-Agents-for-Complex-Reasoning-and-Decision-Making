# util/llmUtil.py
# -*- coding: utf-8 -*-
"""
Notes
- "chat_json": does NOT require tool/function calling; it enforces "ONLY JSON" and retries if parse fails.
- "embed_texts": uses client.embeddings.create(model=..., input=[...]).
"""

from __future__ import annotations

import os
import re
import json
import time
import random
from typing import Any, Dict, List, Optional, Sequence, Union, Callable, Tuple

from openai import OpenAI
from util.logUtil import setup_logger
from config import LLMConfig
config = LLMConfig()
logger = setup_logger()

import threading
from collections import deque

class _RateLimiter:
    """
    Thread-safe limiter:
    - inflight semaphore limits concurrent requests
    - optional QPS token bucket via sliding window
    """
    def __init__(self, max_inflight: int = 8, qps: Optional[float] = None):
        self.sem = threading.Semaphore(max_inflight)
        self.qps = qps
        self.lock = threading.Lock()
        self.ts = deque()  # timestamps of recent requests

    def acquire(self):
        self.sem.acquire()
        if self.qps is None or self.qps <= 0:
            return

        # Sliding window: ensure <= qps requests per 1s
        while True:
            now = time.time()
            with self.lock:
                # purge old
                while self.ts and now - self.ts[0] >= 1.0:
                    self.ts.popleft()

                if len(self.ts) < self.qps:
                    self.ts.append(now)
                    return

                # need wait
                wait = 1.0 - (now - self.ts[0])
            if wait > 0:
                time.sleep(min(wait, 0.2))  # small sleep slices

    def release(self):
        self.sem.release()

def _retry_sleep(attempt: int) -> None:
    time_to_wait = random.random() * 5 * (2 ** attempt)
    logger.info(f"Sleep {time_to_wait}s before retrying after attempt {attempt}...")
    time.sleep(time_to_wait)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _extract_json_candidate(text: str) -> Optional[str]:
    """
    Extract a likely JSON string from model output.
    Tries:
    - fenced ```json ... ```
    - first top-level {...} or [...]
    """
    t = text.strip()
    if not t:
        return None

    m = _JSON_FENCE_RE.search(t)
    if m:
        return m.group(1).strip()

    # Heuristic: find first '{' or '[' then last matching '}' or ']'
    # This is intentionally simple; robust parsing is handled by retries.
    m2 = _JSON_OBJ_RE.search(t)
    if m2:
        return m2.group(1).strip()

    return None


def _normalize_placeholders(obj: Any) -> Any:
    """
    Recursively replace placeholder strings like "___" or all underscores with None.
    """
    if isinstance(obj, str):
        s = obj.strip()
        if s == "" or s == "___" or re.fullmatch(r"_+", s):
            return None
        return obj
    if isinstance(obj, list):
        return [_normalize_placeholders(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize_placeholders(v) for k, v in obj.items()}
    return obj

class LLMUtil:
    """
    Wrapper for LLM interactions (chat completions, embeddings) using OpenAI-compatible API.
    """

    def chat(self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Simple chat completion. Returns the assistant message content (string).
        """
        logger.debug(f"Chat request to model {model} with {len(messages)} messages.")
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature, 
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)

        last_err: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            self._limiter.acquire()
            try:
                resp = self.client.chat.completions.create(**payload)
                answer = (resp.choices[0].message.content or "").strip()
                logger.debug(f"Chat response received (length {len(answer)}).")
                return answer
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(f"chat error (attempt {attempt}/{self.max_retries}): {last_err}")
                _retry_sleep(attempt)
            finally:
                self._limiter.release()

        raise RuntimeError(f"chat failed after {self.max_retries} retries. Last error: {last_err}")

    def chat_json(self,
        *,
        system_prompt: str,
        user_text: str,
        model: str = "qwen-turbo",
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        strict_only_json: bool = True,
        normalize_placeholders: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ask the model to return JSON only. Retries if JSON parsing fails.

        Returns parsed dict/list (usually dict).
        """

        # A strong "only JSON" constraint
        guard = "Return ONLY valid JSON. Do not include markdown, code fences, or explanations."
        if strict_only_json:
            sys = f"{system_prompt.strip()}\n\n{guard}"
        else:
            sys = system_prompt.strip()

        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": user_text},
            ]
            try:
                text = self.chat(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra=extra,
                )
                cand = _extract_json_candidate(text) or text.strip()

                parsed = json.loads(cand)
                if normalize_placeholders:
                    parsed = _normalize_placeholders(parsed)
                return parsed  # type: ignore[return-value]

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(f"chat_json attempt {attempt} failed to parse JSON: {last_err}")

                # Repair prompt: feed back the error and ask for corrected JSON only
                repair_user = (
                    "Your previous output could not be parsed as JSON.\n"
                    f"Error: {last_err}\n"
                    "Please output ONLY corrected valid JSON that matches the required schema.\n"
                    "Do not add any extra text.\n"
                    "Original task input:\n"
                    f"{user_text}"
                )
                user_text = repair_user

                # small backoff
                _retry_sleep(attempt)

        raise RuntimeError(f"chat_json failed after {self.max_retries} retries. Last error: {last_err}")


    # -----------------------------
    # Embedding utilities
    # -----------------------------

    def embed_texts(self,
        texts: Sequence[str],
        *,
        model: str = "text-embedding-v4",
        batch_size: int = 4096,
        normalize_newlines: bool = True,
    ) -> List[List[float]]:
        """
        Create embeddings for a list of texts via OpenAI-compatible embeddings endpoint.

        Returns: List[embedding_vector]
        """
        out: List[List[float]] = []
        if not texts:
            return out

        def _prep(t: str) -> str:
            if normalize_newlines:
                return t.replace("\n", " ").strip()
            return t.strip()

        def _emb_create(buf: List[str]) -> List[List[float]]:
            last_err: Optional[str] = None
            for attempt in range(1, self.max_retries + 1):
                self._limiter.acquire()
                try:
                    resp = self.client.embeddings.create(model=model, input=buf)
                    return [d.embedding for d in resp.data]
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    logger.warning(f"embeddings retryable error (attempt {attempt}/5): {last_err}")
                    _retry_sleep(attempt)
                finally:
                    self._limiter.release()
            raise RuntimeError(f"embeddings failed after retries. Last error: {last_err}")
    
        buf: List[str] = []
        for t in texts:
            buf.append(_prep(t))
            if len(buf) >= batch_size:
                resp = _emb_create(buf)
                # preserve order
                out.extend(resp)
                buf = []

        if buf:
            resp = _emb_create(buf)
            out.extend(resp)

        return out
    
    def __init__(self) -> None:
        self.config = LLMConfig()
        self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        self.token_usage = {
            "chat": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "embeddings": {"input_tokens": 0, "total_tokens": 0},
        }
        max_inflight = getattr(self.config, "max_inflight", 16)
        qps = getattr(self.config, "qps", 10)  # e.g., 5 or 10
        self._limiter = _RateLimiter(max_inflight=max_inflight, qps=qps)
        self.max_retries = self.config.max_retries




# # -----------------------------
# # Minimal self-test (optional)
# # -----------------------------
# if __name__ == "__main__":
#     # Quick smoke test:
#     #   export DASHSCOPE_API_KEY=...
#     #   python util/llmUtil.py
#     llm = LLMUtil()
#     obj = llm.chat_json(
#         system_prompt="You are a JSON generator.",
#         user_text='Return {"ok": true, "x": "___", "y": "____"}',
#     )
#     print("chat_json:", obj)

#     vecs = llm.embed_texts(["hello", "world"])
#     print("embeddings:", len(vecs), len(vecs[0]) if vecs else 0)


from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import os

def _one_job(i: int) -> dict:
    llm = LLMUtil()  # 同进程多线程共享 limiter（因为 limiter 在实例里；想更严格可改成模块级全局 limiter）
    t0 = time.time()
    obj = llm.chat_json(
        system_prompt="You are a JSON generator. Output JSON only.",
        user_text=f'{{"job": {i}, "ok": true, "ts": "{time.time()}"}}',
        model=getattr(config, "model", "qwen-turbo"),
        temperature=0.0,
    )
    dt = time.time() - t0
    return {"i": i, "dt": dt, "obj": obj}

def limiter_smoke_test():
    # 你可以用环境变量临时调参（如果你在 LLMConfig 里接了 env）
    # os.environ["LLM_MAX_INFLIGHT"] = "4"
    # os.environ["LLM_QPS"] = "2"

    total_jobs = 1000          # 总请求数
    thread_workers = 20      # 启很多线程，看看 limiter 是否能“压住”

    t_all = time.time()
    results = []
    fails = 0

    with ThreadPoolExecutor(max_workers=thread_workers) as ex:
        futs = [ex.submit(_one_job, i) for i in range(total_jobs)]
        for fu in as_completed(futs):
            try:
                r = fu.result()
                results.append(r)
                print(f"job {r['i']:02d} done in {r['dt']:.2f}s -> keys={list(r['obj'].keys())}")
            except Exception as e:
                fails += 1
                print("FAILED:", repr(e))

    total_dt = time.time() - t_all
    results.sort(key=lambda x: x["dt"])
    print("\n========== SUMMARY ==========")
    print(f"jobs={total_jobs}, fails={fails}, total_time={total_dt:.2f}s")
    if results:
        print(f"min={results[0]['dt']:.2f}s, median={results[len(results)//2]['dt']:.2f}s, max={results[-1]['dt']:.2f}s")

if __name__ == "__main__":
    limiter_smoke_test()