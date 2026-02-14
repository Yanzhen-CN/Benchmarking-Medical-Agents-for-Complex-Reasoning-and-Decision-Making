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

from util import logUtil
logger = logUtil.setup_logger()

# ---- LLM Provider (OpenAI-compatible) ----
from agents.mem0_agent import OpenAICompatibleLLMProvider  # reuse your provider
# ---- Context builder (your existing API) ----
from get_messages_for_eval import get_memory_and_context_for_qid


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
# Build question prompt (MCQ)
# ============================================================

def format_mcq_user_content(q: Dict[str, Any]) -> str:
    question = q.get("question", "")
    if not isinstance(question, str):
        question = _to_str(question)

    options = q.get("options") or []
    if not isinstance(options, list):
        options = []

    lines = [question.strip(), "", "Options:"]
    for i, opt in enumerate(options):
        if not isinstance(opt, str):
            opt = _to_str(opt)
        lines.append(f"{i+1}. {opt}")
    lines += [
        "",
        "Instruction: Answer with EXACTLY one option string from the list above (copy-paste).",
    ]
    return "\n".join(lines)


def extract_pred_option(reply: Any, options: List[str]) -> str:
    r = reply if isinstance(reply, str) else _to_str(reply)
    r_strip = r.strip()
    r_low = r_strip.lower()

    # exact match
    for opt in options:
        if isinstance(opt, str) and opt.strip().lower() == r_low:
            return opt

    # substring match (first hit)
    for opt in options:
        if isinstance(opt, str) and opt.strip() and opt.strip().lower() in r_low:
            return opt

    return r_strip


def score_weighted_acc(gt_answer: Any, pred: str) -> float:
    """
    Supports:
      - gt_answer: str
      - gt_answer: dict {option_str: weight, ...}
    """
    if isinstance(gt_answer, dict):
        return float(gt_answer.get(pred, 0.0) or 0.0)
    if isinstance(gt_answer, str):
        return 1.0 if pred == gt_answer else 0.0
    return 0.0


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
) -> Tuple[str, Dict[str, Any]]:
    """
    Greedily pack as many memories (previous visits events/notes) as possible
    under a character budget.
    """
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
) -> str:
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
    return llm.chat(messages=messages, **kwargs)  # type: ignore


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_jsonl", type=str, required=True, help="A single patient question jsonl (e.g. P000001.jsonl)")
    ap.add_argument("--memory_type", type=str, default="event", choices=["event", "note"], help="Which memory type get_memory_and_context_for_qid uses")
    ap.add_argument("--mem_chars", type=int, default=24000, help="Total char budget for longitudinal supplement block")
    ap.add_argument("--item_chars", type=int, default=900, help="Max chars per memory item")
    ap.add_argument("--ctx_msg_chars", type=int, default=12000, help="Max chars per existing context message content")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", ""), help="Optional model override")
    ap.add_argument("--out_dir", type=str, default="log/llm_eval")
    args = ap.parse_args()

    qpath = Path(args.questions_jsonl)
    if not qpath.exists():
        raise SystemExit(f"Missing: {qpath}")

    out_dir = Path(args.out_dir)
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
    for q in tqdm(questions, desc=f"LLM-only Eval {qpath.name} ({args.memory_type})"):
        qid = q.get("qid")
        qtype = q.get("qtype", "UNKNOWN")
        options = q.get("options") or []
        gt = q.get("answer")

        if not isinstance(qid, str) or not qid:
            continue
        if not isinstance(options, list):
            options = []

        # 1) get context + memories
        pack = get_memory_and_context_for_qid(qid=qid, memory_type=args.memory_type)
        memories = pack.get("memories") or []
        context_messages = pack.get("context_messages") or []
        if not isinstance(memories, list):
            memories = []
        if not isinstance(context_messages, list):
            context_messages = []

        # 2) normalize context messages for OpenAI compatibility
        ctx = normalize_messages(context_messages, max_chars=args.ctx_msg_chars)

        # 3) build longitudinal supplement (pack as much as possible)
        supp_text, supp_meta = build_memory_supplement_block(
            memories,
            visit_order=None,  # if you have V->idx mapping, pass it here
            max_total_chars=args.mem_chars,
            max_item_chars=args.item_chars,
            prefer_notes_first=True,
        )

        # 4) build user MCQ
        user_content = format_mcq_user_content(q)

        # 5) final messages: context + system supplement + user question
        messages = ctx + [{"role": "system", "content": supp_text}, {"role": "user", "content": user_content}]
        messages = normalize_messages(messages, max_chars=args.ctx_msg_chars)  # safety re-normalize

        # 6) call llm
        try:
            reply = llm_chat_once(
                llm,
                messages,
                model=(args.model.strip() or None),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            # record failure but continue
            rec = {
                "qid": qid,
                "qtype": qtype,
                "memory_type": args.memory_type,
                "error": str(e),
                "supp_meta": supp_meta,
            }
            records.append(rec)
            continue

        pred = extract_pred_option(reply, options)
        score = score_weighted_acc(gt, pred)

        scores_by_type[qtype].append(score)
        records.append(
            {
                "qid": qid,
                "qtype": qtype,
                "memory_type": args.memory_type,
                "pred": pred,
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
        "memory_type": args.memory_type,
        "overall_acc": overall,
        "by_type": summary,
        "run_id": run_id,
    }

    # write outputs
    pred_path = out_dir / f"{patient_id}.{args.memory_type}.{run_id}.pred.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sum_path = out_dir / f"{patient_id}.{args.memory_type}.{run_id}.summary.json"
    sum_path.write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # plot acc by type
    order = ["T3-N", "T3-A", "T3-M", "T3-D"]
    keys = sorted(summary.keys(), key=lambda k: (order.index(k) if k in order else 999, k))
    vals = [summary[k]["acc"] for k in keys]

    fig_path = out_dir / f"{patient_id}.{args.memory_type}.{run_id}.acc_by_type.png"
    plt.figure(figsize=(7, 4))
    plt.bar(keys, vals)
    plt.ylim(0, 1.0)
    plt.xlabel("Question Type")
    plt.ylabel("Weighted Accuracy")
    plt.title(f"LLM-only Acc by Type ({patient_id}, {args.memory_type})")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info(f"[DONE] patient={patient_id} memory_type={args.memory_type} overall_acc={overall:.4f}")
    logger.info(f"predictions: {pred_path}")
    logger.info(f"summary:     {sum_path}")
    logger.info(f"figure:      {fig_path}")


if __name__ == "__main__":
    main()
