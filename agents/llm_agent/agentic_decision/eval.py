# agents/llm_agent/agentic_decision/eval.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from util import logUtil
logger = logUtil.setup_logger()
from config import AgentTaskConfig  
cfg = AgentTaskConfig()
# ---- LLM Provider (OpenAI-compatible) ----
from agents.mem0_agent import OpenAICompatibleLLMProvider  # reuse your provider
# ---- Context builder (your existing API) ----
from tasks.agentic_decision.get_messages_for_eval import get_memory_and_context_for_qid
from tasks.agentic_decision.eval_utils import *
from tasks.agentic_decision.prompts import get_agent_qa_prompt, AGENT_ACTION_PROMPT

out_dir = Path("./agents/llm_agent/agentic_decision/results")
# ============================================================
# IO
# ============================================================

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Message normalization (fix: dict content -> str)
# ============================================================

_ALLOWED_ROLES = {"system", "user", "assistant"}  # safest for OpenAI-compatible wrappers


def _to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def normalize_messages(msgs: List[Dict[str, Any]], *, max_chars: int = 12000) -> List[Dict[str, str]]:
    """
    Ensure every message is {role: str, content: str}.
    Unknown roles are mapped to 'assistant' (most compatible).
    """
    out: List[Dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        if role not in _ALLOWED_ROLES:
            role = "assistant"
        content = _to_str(m.get("content"))
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[TRUNCATED]..."
        out.append({"role": role, "content": content})
    return out


# ============================================================
# LLM-only "longitudinal supplement" (pack as much as possible)
# ============================================================

_VISIT_PAT = re.compile(r"\bvisit\s*=\s*(V\d+)\b", re.IGNORECASE)


def infer_visit_ref_from_memory_text(s: str) -> str:
    m = _VISIT_PAT.search(s)
    return m.group(1).upper() if m else "UNK"


def bucket_memories_by_visit(
    memories: List[str],
    visit_order: Optional[Dict[str, int]] = None,
) -> Dict[int, List[str]]:
    """
    Return {visit_idx: [mem_text...]}.
    If visit_order provided (V1->0...), use it; else parse V<number>.
    """
    tmp: Dict[str, List[str]] = defaultdict(list)
    for m in memories:
        s = _to_str(m).strip()
        if not s:
            continue
        vref = infer_visit_ref_from_memory_text(s)
        tmp[vref].append(s)

    buckets: Dict[int, List[str]] = {}
    if visit_order:
        for vref, lst in tmp.items():
            if vref in visit_order:
                buckets[int(visit_order[vref])] = lst
            else:
                buckets.setdefault(-1, []).extend(lst)
        return buckets

    def vref_key(vref: str) -> int:
        if vref.startswith("V") and vref[1:].isdigit():
            return int(vref[1:])
        return -1

    for vref, lst in tmp.items():
        buckets[vref_key(vref)] = lst
    return buckets


def build_memory_supplement_block(
    memories: List[str],
    *,
    visit_order: Optional[Dict[str, int]] = None,
    max_total_chars: int = 24000,
    max_item_chars: int = 900,
    prefer_notes_first: bool = True,
    max_visible_visits: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Greedily pack as many memories (previous visits events/notes) as possible
    under a character budget.
    """
    
    if max_visible_visits is not None and max_visible_visits <= 0:
        return "", {
            "kept_count": 0,
            "used_chars": 0,
            "max_total_chars": max_total_chars,
            "kept_visit_ids": [],
            "num_bucket_visits": 0,
        }
    buckets = bucket_memories_by_visit(memories, visit_order=visit_order)
    visit_ids = sorted(buckets.keys(), reverse=True)  # recent -> old

    def is_note(s: str) -> bool:
        ss = s.lower()
        return ("admission_note" in ss) or ("discharge_note" in ss) or ("[note" in ss)

    used = 0
    kept_lines: List[str] = []
    kept_visits: List[int] = []
    kept_count = 0

    for vid in visit_ids:
        items = buckets[vid]
        if prefer_notes_first:
            items = sorted(items, key=lambda x: (0 if is_note(x) else 1))

        any_added = False
        if max_visible_visits is not None and len(kept_visits) >= max_visible_visits:
            break
        
        for it in items:
            t = it
            if len(t) > max_item_chars:
                t = t[:max_item_chars] + " ...[TRUNCATED]"
            line = f"- {t}"
            add = len(line) + 1
            if used + add > max_total_chars:
                break
            kept_lines.append(line)
            used += add
            kept_count += 1
            any_added = True

        if any_added:
            kept_visits.append(vid)
        if used >= max_total_chars:
            break

    header = (
        "You are given additional longitudinal history from previous visits (events/notes), "
        "packed as much as possible under a context budget.\n"
        "Use it ONLY as supporting context; do not invent facts.\n"
        "=== Longitudinal History (most recent first) ===\n"
    )
    text = header + ("\n".join(kept_lines) if kept_lines else "<EMPTY>")

    meta = {
        "kept_count": kept_count,
        "used_chars": used,
        "max_total_chars": max_total_chars,
        "kept_visit_ids": kept_visits,
        "num_bucket_visits": len(buckets),
    }
    return text, meta


# ============================================================
# LLM call
# ============================================================

def llm_chat_once(
    llm: OpenAICompatibleLLMProvider,
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Wrapper for your OpenAICompatibleLLMProvider.
    Adjust this function if your provider uses different method signature.
    """
    # Most wrappers provide llm.chat(messages=..., **kwargs)
    kwargs = {}
    if model:
        kwargs["model"] = model
    kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking  # example flag for chain-of-thought; adjust as needed
    return llm.chat_json_ctx(messages=messages, **kwargs)  # type: ignore


# ============================================================
# Main
# ============================================================

def run_one_visit(
    questions_jsonl: Path,
    memory_type: str = "event",
    temperature: float = 0.0,
    model: Optional[str] = None,
    visible_visits: int = 10,
    enable_thinking: bool = False,
    ) -> Dict[str, Any]:

    qpath = questions_jsonl
    if not qpath.exists():
        raise SystemExit(f"Missing: {qpath}")


    safe_mkdir(out_dir)

    # run id
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = uuid.uuid4().hex[:8]
    run_id = f"{ts}-{rid}"

    # infer patient id
    first = None
    for x in iter_jsonl(qpath):
        first = x
        break
    patient_id = (
        str(first.get("qid", "")).split("-V")[0]
        if isinstance(first, dict) and first.get("qid")
        else qpath.stem
    )

    llm = OpenAICompatibleLLMProvider()

    scores_by_type: Dict[str, List[float]] = defaultdict(list)
    records: List[Dict[str, Any]] = []

    questions = list(iter_jsonl(qpath))
    for q in tqdm(questions, desc=f"LLM-only Eval {qpath.name} ({memory_type})"):
        qid = q.get("qid")
        qtype = q.get("qtype", "UNKNOWN")
        options = q.get("options") or []
        gt = q.get("answer")

        if not isinstance(qid, str) or not qid:
            continue
        if not isinstance(options, list):
            options = []

        # 1) get context + memories
        pack = get_memory_and_context_for_qid(qid=qid, memory_type=memory_type)
        memories = pack.get("memories") or []
        context_messages = pack.get("context_messages") or []
        if not isinstance(memories, list):
            memories = []
        if not isinstance(context_messages, list):
            context_messages = []

        # 2) normalize context messages for OpenAI compatibility
        ctx = normalize_messages(context_messages, max_chars=cfg.CTX_CHARS)

        # 3) build longitudinal supplement (pack as much as possible)
        supp_text, supp_meta = build_memory_supplement_block(
            memories,
            visit_order=None,  # if you have V->idx mapping, pass it here
            max_total_chars=cfg.MEM_CHARS,
            max_item_chars=cfg.ITEM_CHARS,
            prefer_notes_first=True,
            max_visible_visits=visible_visits
        )

        # 4) build user MCQ
        user_content = format_mcq_user_content(q)

        # 5) final messages: context + system supplement + user question        
        messages = [{"role": "system", "content": AGENT_ACTION_PROMPT}] \
        + ([ {"role": "system", "content": supp_text}] if supp_text != "" else []) \
        + ctx[1:] + [
            get_agent_qa_prompt(),
            {"role": "user", "content": user_content}
        ]
        messages = normalize_messages(messages, max_chars=cfg.CTX_CHARS)  # safety re-normalize
        # 6) call llm
        try:
            reply = llm_chat_once(
                llm,
                messages,
                model=(model.strip() if model else None),
                temperature=temperature,
                max_tokens=cfg.MAX_TOKENS,
                enable_thinking=enable_thinking
            )
            pred = reply["answer"] if isinstance(reply, dict) and "answer" in reply else [str(reply)]
        except Exception as e:
            # record failure but continue
            rec = {
                "qid": qid,
                "qtype": qtype,
                "memory_type": memory_type,
                "error": str(e),
                "supp_meta": supp_meta,
            }
            records.append(rec)
            continue

        score = score_weighted_acc(gt, pred_list=pred)

        scores_by_type[qtype].append(score)
        records.append(
            {
                "qid": qid,
                "qtype": qtype,
                "memory_type": memory_type,
                "pred": pred,
                "reply": reply,
                "score": score,
                "gt": gt,
                "supp_meta": supp_meta,
            }
        )

    # summarize
    summary: Dict[str, Any] = {}
    all_scores: List[float] = []
    for qt, ss in scores_by_type.items():
        arr = np.asarray(ss, dtype=float) if ss else np.zeros((0,), dtype=float)
        summary[qt] = {"n": int(len(ss)), "acc": float(arr.mean()) if len(arr) else 0.0}
        all_scores.extend(ss)

    overall = float(np.mean(all_scores)) if all_scores else 0.0
    summary_out = {
        "patient_id": patient_id,
        "memory_type": memory_type,
        "overall_acc": overall,
        "by_type": summary,
        "run_id": run_id,
        "usage": llm.get_token_usage()
    }

    # write outputs
    pred_path = out_dir / f"{patient_id}.{memory_type}.{run_id}.pred.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sum_path = out_dir / f"{patient_id}.{memory_type}.{run_id}.summary.json"
    sum_path.write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # plot acc by type
    order = ["T3-N", "T3-A", "T3-M", "T3-D"]
    keys = sorted(summary.keys(), key=lambda k: (order.index(k) if k in order else 999, k))
    vals = [summary[k]["acc"] for k in keys]

    fig_path = out_dir / f"{patient_id}.{memory_type}.{run_id}.acc_by_type.png"
    plt.figure(figsize=(7, 4))
    plt.bar(keys, vals)
    plt.ylim(0, 1.0)
    plt.xlabel("Question Type")
    plt.ylabel("Weighted Accuracy")
    plt.title(f"LLM-only Acc by Type ({patient_id}, {memory_type})")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info(f"[DONE] patient={patient_id} memory_type={memory_type} overall_acc={overall:.4f}")
    logger.info(f"predictions: {pred_path}")
    logger.info(f"summary:     {sum_path}")
    logger.info(f"figure:      {fig_path}")
    
    return summary_out
    

def main():
    parser = argparse.ArgumentParser(description="LLM-only evaluation for agentic decision task")
    parser.add_argument("--memory_type", type=str, default="event", help="Type of memory to use (e.g., 'event' or 'note')")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature for response generation")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (if applicable)")
    parser.add_argument("--visible_visits", type=int, default=9999, help="Number of most recent visits to include in context (if applicable)")
    parser.add_argument("--enable_thinking", action="store_true", default=False ,help="Whether to enable 'thinking' (chain-of-thought) in the prompt")
    args = parser.parse_args()

    question_dir =cfg.QUESTIONS_DIR
    if not cfg.CLIP_PAITENT:
        qfiles = sorted(question_dir.glob("*.jsonl"))
    else:
        qfiles = sorted([question_dir / f"P{str(pid).zfill(6)}.jsonl" for pid in cfg.CLIP_PATIENT_IDS])
    
    if not qfiles:
        logger.error(f"No question files found in {question_dir}")
        return
    
    if cfg.DEMO_MODE:
        qfiles = qfiles[:cfg.DEMO_N]
        logger.info(f"DEMO MODE: Only processing {len(qfiles)} files")
    else:
        logger.info(f"Found {len(qfiles)} question files to process")
    
    log_name = f"llm_eval_{args.memory_type}_{args.temperature}_{args.model}_{args.visible_visits}{'_thinking' if args.enable_thinking else ''}_nolimit.json"
    if os.path.exists(out_dir/log_name):
        with open(out_dir/log_name, "r", encoding="utf-8") as f:
            existing_log = json.load(f)
            done_files = set(existing_log.keys())
            qfiles = [qf for qf in qfiles if qf.name not in done_files]
            logger.info(f"Resuming from existing log. {len(qfiles)} files left to process.")
    else:
        existing_log = {}
        logger.info(f"No existing log found. Starting fresh.")
        
    total_usage = {
        "chat": {
            "prompt_tokens": existing_log.get("usage", {}).get("chat", {}).get("prompt_tokens", 0),
            "completion_tokens": existing_log.get("usage", {}).get("chat", {}).get("completion_tokens", 0),
            "total_tokens": existing_log.get("usage", {}).get("chat", {}).get("total_tokens", 0),
        },
        "embedding": {
            "input_tokens": existing_log.get("usage", {}).get("embedding", {}).get("input_tokens", 0),
            "total_tokens": existing_log.get("usage", {}).get("embedding", {}).get("total_tokens", 0),
        },
    }
    total_log = existing_log
    safe_mkdir(out_dir)
    with ProcessPoolExecutor(max_workers=min(cfg.MAXWORKERS, len(qfiles))) as executor:
        futures = {executor.submit(run_one_visit, qf, args.memory_type, args.temperature, args.model, args.visible_visits, args.enable_thinking): qf for qf in qfiles}
        for future in as_completed(futures):
            qf = futures[future]
            try:
                log = future.result()
                logger.info(f"Completed {qf}: {json.dumps(log, ensure_ascii=False, indent=2)}")
                # Aggregate usage
                for k, v in log.get("usage", {}).get("chat", {}).items():
                    total_usage["chat"][k] += v
                for k, v in log.get("usage", {}).get("embedding", {}).items():
                    total_usage["embedding"][k] += v
                total_log[qf.name] = log
                with open(out_dir / log_name, "w", encoding="utf-8") as f:
                    json.dump(total_log, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error processing {qf}: {e}")
    

    
        # for qf in tqdm(qfiles):
        # try:
        #     res = run_one_visit(qf, memory_type=args.memory_type, temperature=args.temperature, model=args.model, visible_visits=args.visible_visits, enable_thinking=args.enable_thinking)
        #     logger.info(f"Result for {qf}: {res}")
        #     for k, v in res.get("usage", {}).get("chat", {}).items():
        #         total_usage["chat"][k] += v
        #     for k, v in res.get("usage", {}).get("embedding", {}).items():
        #         total_usage["embedding"][k] += v
        #     total_log[qf.name] = res
        #     with open(out_dir / log_name, "w", encoding="utf-8") as f:
        #         json.dump(total_log, f, ensure_ascii=False, indent=2)

        # except Exception as e:
        #     logger.error(f"Error processing {qf}: {e}")
    
    logger.info(f"Total LLM token usage across all files: {json.dumps(total_usage, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
