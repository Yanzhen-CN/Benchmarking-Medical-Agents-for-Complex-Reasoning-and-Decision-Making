# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
import re

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util import logUtil
logger = logUtil.setup_logger()

from util.llmUtil import LLMUtil
from tasks.agentic_decision.get_messages_for_eval import get_memory_and_context_for_qid
from agents.rag_agent.agentic_decision.retriver import PatientRetriever, RetrievedDoc

# ====== NEW: align with provided eval.py ======
from tasks.agentic_decision.prompts import get_agent_qa_prompt, AGENT_ACTION_PROMPT


# -----------------------------
# IO helpers
# -----------------------------
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -----------------------------
# MCQ helpers
# -----------------------------
def _format_mcq_user_content(q: Dict[str, Any]) -> str:
    question = q.get("question", "")
    options = q.get("options") or []
    if not isinstance(options, list):
        options = []
    lines = [str(question).strip(), "", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"{i+1}. {opt}")
    lines.append("")
    lines.append("Please answer with the best option string exactly as listed.")
    return "\n".join(lines)


def _extract_pred_option(reply: Any, options: List[str]) -> str:
    r = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
    r_low = r.lower().strip()

    for opt in options:
        if isinstance(opt, str) and opt.lower().strip() == r_low:
            return opt
    for opt in options:
        if isinstance(opt, str) and opt.lower() in r_low:
            return opt

    return r.strip()


def _score_weighted_acc(gt_answer: Any, pred: str) -> float:
    if isinstance(gt_answer, dict):
        return float(gt_answer.get(pred, 0.0) or 0.0)
    if isinstance(gt_answer, str):
        return 1.0 if pred == gt_answer else 0.0
    return 0.0


# -----------------------------
# RAG prompt helpers
# -----------------------------
def _build_retrieved_system_block(hits: List[RetrievedDoc]) -> str:
    if not hits:
        return "Retrieved evidence (RAG top-k): <EMPTY>"

    lines = ["Retrieved evidence (RAG top-k):"]
    for i, h in enumerate(hits, 1):
        t = h.text if isinstance(h.text, str) else str(h.text)
        if len(t) > 1200:
            t = t[:1200] + "..."
        meta = h.meta or {}
        tag = f"score={h.score:.4f}"
        tag2 = f"visit={meta.get('visit_ref')} type={meta.get('event_type', meta.get('memory_type'))} id={meta.get('event_id', meta.get('note_type'))}"
        lines.append(f"[{i}] ({tag}; {tag2}) {t}")
    return "\n".join(lines)


def _parse_patient_id_from_qid(qid: str) -> str:
    return qid.split("-V")[0] if "-V" in qid else qid.split("-")[0]


def _visible_until_visit_idx_from_visit_id(visit_id: str) -> int | None:
    """
    visit_id 形如: P000001-V12
    返回 visible_until_visit_idx=11 (0-based)，表示只看 V12 之前（V1..V11）
    """
    if not isinstance(visit_id, str) or "-V" not in visit_id:
        return None
    try:
        vnum = int(visit_id.split("-V")[1])
        return max(0, vnum - 1)
    except Exception:
        return None


# ============================================================
# NEW: Message normalization + longitudinal supplement (copy style)
# ============================================================

_ALLOWED_ROLES = {"system", "user", "assistant"}


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


_VISIT_PAT = re.compile(r"\bvisit\s*=\s*(V\d+)\b", re.IGNORECASE)


def infer_visit_ref_from_memory_text(s: str) -> str:
    m = _VISIT_PAT.search(s)
    return m.group(1).upper() if m else "UNK"


def bucket_memories_by_visit(
    memories: List[str],
    visit_order: Optional[Dict[str, int]] = None,
) -> Dict[int, List[str]]:
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


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_jsonl", type=str, required=True)
    ap.add_argument("--memory_type", type=str, default="event", choices=["event", "note"])
    ap.add_argument("--top_k", type=int, default=int(os.getenv("RAG_TOP_K", "16")))
    ap.add_argument("--prefetch_k", type=int, default=int(os.getenv("RAG_PREFETCH_K", "200")))
    ap.add_argument("--include_cutoff", action="store_true", default=True)
    ap.add_argument("--require_timestamp", action="store_true", default=False)
    ap.add_argument("--debug", action="store_true", default=True)
    ap.add_argument("--out_dir", type=str, default="log/rag_eval")
    ap.add_argument("--model", type=str, default=os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--ctx_msg_chars", type=int, default=64000, help="Max chars per existing context message content")
    ap.add_argument("--mem_chars", type=int, default=64000, help="Total char budget for longitudinal supplement block")
    ap.add_argument("--item_chars", type=int, default=64000, help="Max chars per memory item")
 
    args = ap.parse_args()

    qpath = Path(args.questions_jsonl)
    if not qpath.exists():
        raise SystemExit(f"Missing questions file: {qpath}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = uuid.uuid4().hex[:8]
    run_id = f"{ts}-{rid}"

    retriever = PatientRetriever()
    llm = LLMUtil()

    first = next(iter_jsonl(qpath))
    patient_id = _parse_patient_id_from_qid(str(first.get("qid", qpath.stem)))

    scores_by_type: Dict[str, List[float]] = defaultdict(list)
    records: List[Dict[str, Any]] = []

    for q in tqdm(list(iter_jsonl(qpath)), desc=f"RAG Eval {qpath.name} ({args.memory_type})"):
        qid = q.get("qid")
        qtype = q.get("qtype", "UNKNOWN")
        options = q.get("options") or []
        gt = q.get("answer")

        if not isinstance(qid, str) or not qid:
            continue

        # 1) get context pack (context + memories)
        pack = get_memory_and_context_for_qid(qid=qid, memory_type=args.memory_type)

        # provided-file style: pull memories separately
        memories = pack.get("memories") or []
        if not isinstance(memories, list):
            memories = []

        context_messages = pack.get("context_messages") or []
        if not isinstance(context_messages, list):
            context_messages = []

        # normalize context (OpenAI-compatible)
        ctx = normalize_messages(context_messages, max_chars=args.ctx_msg_chars)

        visit_id = pack.get("visit_id")
        cmeta = pack.get("context_meta") or {}
        cutoff_event_id = cmeta.get("cutoff_event_id")

        visible_until_visit_idx = None
        if cutoff_event_id:
            # 事件级截断（最精确）—— retriever 内部处理
            pass
        else:
            visible_until_visit_idx = _visible_until_visit_idx_from_visit_id(visit_id)

        # 2) build query text
        query_text = str(q.get("question", "")).strip() or str(pack.get("question", "")).strip()

        # 3) RAG retrieve
        hits = retriever.search(
            patient_id=patient_id,
            query=query_text,
            memory_type=args.memory_type,
            k=args.top_k,
            prefetch_k=args.prefetch_k,
            visible_until_visit_idx=visible_until_visit_idx,
            cutoff_event_id=cutoff_event_id,
            include_cutoff=args.include_cutoff,
            require_timestamp=args.require_timestamp,
        )

        # 4) build blocks: longitudinal supplement + rag evidence + user MCQ
        supp_text, supp_meta = build_memory_supplement_block(
            memories,
            visit_order=None,
            max_total_chars=args.mem_chars,
            max_item_chars=args.item_chars,
            prefer_notes_first=True,
        )
        sys_rag = _build_retrieved_system_block(hits)
        user_content = _format_mcq_user_content(q)

        # 5) compose messages (ALIGN WITH PROVIDED FILE)
        #    system(action rules) + system(longitudinal) + system(rag evidence) + ctx[1:] + qa prompt + user
        messages = (
            [{"role": "system", "content": AGENT_ACTION_PROMPT},
             {"role": "system", "content": supp_text},
             {"role": "system", "content": sys_rag}]
            + (ctx[1:] if len(ctx) > 1 else [])
            + [get_agent_qa_prompt(),
               {"role": "user", "content": user_content}]
        )
        messages = normalize_messages(messages, max_chars=args.ctx_msg_chars)

        # 6) call llm
        reply = llm.chat_json_ctx(messages=messages, model=args.model)

        pred = _extract_pred_option(reply, options)
        score = _score_weighted_acc(gt, pred)

        scores_by_type[qtype].append(score)

        records.append(
            {
                "qid": qid,
                "qtype": qtype,
                "memory_type": args.memory_type,
                "patient_id": patient_id,
                "visit_id": visit_id,
                "cutoff_event_id": cutoff_event_id,
                "visible_until_visit_idx": visible_until_visit_idx,
                "query_text": query_text,
                "pred": pred,
                "score": score,
                "gt": gt,
                "supp_meta": supp_meta,
                "retrieved": [{"score": h.score, "text": h.text, "meta": h.meta} for h in hits],
            }
        )

        if args.debug:
            logger.info(f"[{qid}] qtype={qtype} score={score} pred={pred} gt={gt}")

    # --- summarize ---
    summary = {}
    all_scores = []
    for qt, ss in scores_by_type.items():
        arr = np.asarray(ss, dtype=float) if ss else np.zeros((0,), dtype=float)
        summary[qt] = {"n": int(len(ss)), "acc": float(arr.mean()) if len(arr) else 0.0}
        all_scores.extend(ss)
    overall = float(np.mean(all_scores)) if all_scores else 0.0

    out_jsonl = out_dir / f"{patient_id}.{args.memory_type}.{run_id}.pred.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_summary = out_dir / f"{patient_id}.{args.memory_type}.{run_id}.summary.json"
    out_summary.write_text(
        json.dumps(
            {
                "patient_id": patient_id,
                "memory_type": args.memory_type,
                "overall_acc": overall,
                "by_type": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # plot acc by type
    order = ["T3-N", "T3-A", "T3-M", "T3-D"]
    keys = sorted(summary.keys(), key=lambda k: (order.index(k) if k in order else 999, k))
    vals = [summary[k]["acc"] for k in keys]

    plt.figure(figsize=(7, 4))
    plt.bar(keys, vals)
    plt.ylim(0, 1.0)
    plt.xlabel("Question Type")
    plt.ylabel("Weighted Accuracy")
    plt.title(f"RAG Acc by Type ({patient_id}, {args.memory_type})")
    plt.tight_layout()
    out_png = out_dir / f"{patient_id}.{args.memory_type}.{run_id}.acc_by_type.png"
    plt.savefig(out_png)
    plt.close()

    logger.info(f"[DONE] patient={patient_id} memory_type={args.memory_type} overall_acc={overall:.4f}")
    logger.info(f"predictions: {out_jsonl}")
    logger.info(f"summary:     {out_summary}")
    logger.info(f"figure:      {out_png}")


if __name__ == "__main__":
    main()
