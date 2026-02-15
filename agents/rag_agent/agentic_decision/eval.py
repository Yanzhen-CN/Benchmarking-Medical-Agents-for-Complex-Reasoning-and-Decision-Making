# agents/rag_agent/agentic_decision/eval.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from util import logUtil
logger = logUtil.setup_logger()

from config import AgentTaskConfig
cfg = AgentTaskConfig()

# ---- LLM Provider (OpenAI-compatible) ----
from agents.mem0_agent import OpenAICompatibleLLMProvider  # same as llm_agent eval.py

# ---- Context builder (your existing API) ----
from tasks.agentic_decision.get_messages_for_eval import get_memory_and_context_for_qid

# ---- Eval utils (same as llm_agent eval.py) ----
from tasks.agentic_decision.eval_utils import *  # format_mcq_user_content, score_weighted_acc, etc.
from tasks.agentic_decision.prompts import get_agent_qa_prompt, AGENT_ACTION_PROMPT

# ---- RAG Retriever ----
from agents.rag_agent.agentic_decision.retriver import PatientRetriever, RetrievedDoc

out_dir = Path("./agents/rag_agent/agentic_decision/results")

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
# (kept identical to llm_agent eval.py for consistency)
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
# RAG helpers
# ============================================================

def _parse_patient_id_from_qid(qid: str) -> str:
    return qid.split("-V")[0] if isinstance(qid, str) and "-V" in qid else str(qid).split("-")[0]


def _visible_until_visit_idx_from_visit_id(visit_id: Any) -> Optional[int]:
    """
    visit_id like: P000001-V12
    return 11 (0-based), meaning only look at V1..V11 (before V12)
    """
    if not isinstance(visit_id, str) or "-V" not in visit_id:
        return None
    try:
        vnum = int(visit_id.split("-V")[1])
        return max(0, vnum - 1)
    except Exception:
        return None


def _build_retrieved_system_block(hits: List[RetrievedDoc], *, max_chars_per_hit: int = 1200) -> str:
    if not hits:
        return "Retrieved evidence (RAG top-k): <EMPTY>"

    lines = ["Retrieved evidence (RAG top-k):"]
    for i, h in enumerate(hits, 1):
        t = h.text if isinstance(h.text, str) else str(h.text)
        if len(t) > max_chars_per_hit:
            t = t[:max_chars_per_hit] + "..."
        meta = h.meta or {}
        tag = f"score={float(getattr(h, 'score', 0.0)):.4f}"
        tag2 = (
            f"visit={meta.get('visit_ref')} "
            f"type={meta.get('event_type', meta.get('memory_type'))} "
            f"id={meta.get('event_id', meta.get('note_type'))}"
        )
        lines.append(f"[{i}] ({tag}; {tag2}) {t}")
    return "\n".join(lines)


# ============================================================
# LLM call (same wrapper as llm_agent eval.py)
# ============================================================

def llm_chat_once(
    llm: OpenAICompatibleLLMProvider,
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    kwargs = {}
    if model:
        kwargs["model"] = model
    kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return llm.chat_json_ctx(messages=messages, **kwargs)  # type: ignore


# ============================================================
# Main
# ============================================================

def run_one_visit_rag(
    questions_jsonl: Path,
    *,
    memory_type: str = "event",
    temperature: float = 0.0,
    model: Optional[str] = None,
    top_k: int = 16,
    prefetch_k: int = 200,
    include_cutoff: bool = True,
    require_timestamp: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    qpath = questions_jsonl
    if not qpath.exists():
        raise SystemExit(f"Missing: {qpath}")



    # run id
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = uuid.uuid4().hex[:8]
    run_id = f"{ts}-{rid}"

    # infer patient id (fallback to filename)
    first = None
    for x in iter_jsonl(qpath):
        first = x
        break
    patient_id = (
        _parse_patient_id_from_qid(str(first.get("qid", "")))
        if isinstance(first, dict) and first.get("qid")
        else qpath.stem
    )

    llm = OpenAICompatibleLLMProvider()
    retriever = PatientRetriever(llm=llm.get_client())

    scores_by_type: Dict[str, List[float]] = defaultdict(list)
    records: List[Dict[str, Any]] = []

    questions = list(iter_jsonl(qpath))
    for q in tqdm(questions, desc=f"RAG Eval {qpath.name} ({memory_type})"):
        qid = q.get("qid")
        qtype = q.get("qtype", "UNKNOWN")
        options = q.get("options") or []
        gt = q.get("answer")

        if not isinstance(qid, str) or not qid:
            continue
        if not isinstance(options, list):
            options = []

        # 1) context + memories (same API)
        pack = get_memory_and_context_for_qid(qid=qid, memory_type=memory_type)
        memories = pack.get("memories") or []
        context_messages = pack.get("context_messages") or []
        if not isinstance(memories, list):
            memories = []
        if not isinstance(context_messages, list):
            context_messages = []

        # 2) normalize context
        ctx = normalize_messages(context_messages, max_chars=cfg.CTX_CHARS)

        # 3) cutoff info (event-level preferred)
        visit_id = pack.get("visit_id")
        cmeta = pack.get("context_meta") or {}
        cutoff_event_id = cmeta.get("cutoff_event_id")

        visible_until_visit_idx: Optional[int] = None
        if cutoff_event_id:
            # retriever will handle event-level cutoff internally
            visible_until_visit_idx = None
        else:
            visible_until_visit_idx = _visible_until_visit_idx_from_visit_id(visit_id)

        # 4) build longitudinal supplement (same as llm_agent)
        supp_text, supp_meta = build_memory_supplement_block(
            memories,
            visit_order=None,
            max_total_chars=cfg.MEM_CHARS,
            max_item_chars=cfg.ITEM_CHARS,
            prefer_notes_first=True,
        )

        # 5) build user MCQ (use eval_utils implementation)
        user_content = format_mcq_user_content(q)

        # 6) RAG retrieve (query = question text)
        query_text = user_content
        pid_for_query = _parse_patient_id_from_qid(qid) or patient_id

        hits = retriever.search(
            patient_id=pid_for_query,
            query=query_text,
            memory_type=memory_type,
            k=top_k,
            prefetch_k=prefetch_k,
            visible_until_visit_idx=visible_until_visit_idx,
            cutoff_event_id=cutoff_event_id,
            include_cutoff=include_cutoff,
            require_timestamp=require_timestamp,
        )

        rag_block = _build_retrieved_system_block(hits)
        # 7) compose messages (MATCH llm_agent style + add RAG block as a system msg)
        messages = (
            [
                {"role": "system", "content": AGENT_ACTION_PROMPT},
                {"role": "system", "content": supp_text},
            ]
            + (ctx[1:] if len(ctx) > 1 else [])
            + [
                get_agent_qa_prompt(),
                {"role": "user", "content": user_content},
                {"role": "system", "content": rag_block},
            ]
        )
        messages = normalize_messages(messages, max_chars=cfg.CTX_CHARS)

        # 8) call llm (OpenAICompatibleLLMProvider)
        try:
            reply = llm_chat_once(
                llm,
                messages,
                model=(model.strip() if model else None),
                temperature=temperature,
                max_tokens=cfg.MAX_TOKENS,
            )
            pred = reply["answer"] if isinstance(reply, dict) and "answer" in reply else [str(reply)]
            if isinstance(pred, str):
                pred = [pred]
            if not isinstance(pred, list):
                pred = [str(pred)]
        except Exception as e:
            records.append(
                {
                    "qid": qid,
                    "qtype": qtype,
                    "memory_type": memory_type,
                    "error": str(e),
                    "patient_id": pid_for_query,
                    "visit_id": visit_id,
                    "cutoff_event_id": cutoff_event_id,
                    "visible_until_visit_idx": visible_until_visit_idx,
                    "supp_meta": supp_meta,
                }
            )
            continue

        # 9) scoring (use eval_utils; expects pred_list: List[str])
        score = score_weighted_acc(gt, pred_list=pred)

        scores_by_type[qtype].append(score)
        rec = {
            "qid": qid,
            "qtype": qtype,
            "memory_type": memory_type,
            "patient_id": pid_for_query,
            "visit_id": visit_id,
            "cutoff_event_id": cutoff_event_id,
            "visible_until_visit_idx": visible_until_visit_idx,
            "query_text": query_text,
            "pred": pred,
            "reply": reply,
            "score": score,
            "gt": gt,
            "supp_meta": supp_meta,
            "retrieved": [{"score": h.score, "text": h.text, "meta": h.meta} for h in hits],
        }
        records.append(rec)

        if debug:
            logger.info(f"[{qid}] qtype={qtype} score={score} pred={pred} gt={gt}")

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
        "rag": {
            "top_k": top_k,
            "prefetch_k": prefetch_k,
            "include_cutoff": include_cutoff,
            "require_timestamp": require_timestamp,
        },
        "usage": llm.get_token_usage(),
    }

    # write outputs
    pred_path = out_dir / f"{patient_id}.{memory_type}.{run_id}.rag.pred.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sum_path = out_dir / f"{patient_id}.{memory_type}.{run_id}.rag.summary.json"
    sum_path.write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # plot acc by type
    order = ["T3-N", "T3-A", "T3-M", "T3-D"]
    keys = sorted(summary.keys(), key=lambda k: (order.index(k) if k in order else 999, k))
    vals = [summary[k]["acc"] for k in keys]

    fig_path = out_dir / f"{patient_id}.{memory_type}.{run_id}.rag.acc_by_type.png"
    plt.figure(figsize=(7, 4))
    plt.bar(keys, vals)
    plt.ylim(0, 1.0)
    plt.xlabel("Question Type")
    plt.ylabel("Weighted Accuracy")
    plt.title(f"RAG Acc by Type ({patient_id}, {memory_type})")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info(f"[DONE] patient={patient_id} memory_type={memory_type} overall_acc={overall:.4f}")
    logger.info(f"predictions: {pred_path}")
    logger.info(f"summary:     {sum_path}")
    logger.info(f"figure:      {fig_path}")

    return summary_out


def main():
    parser = argparse.ArgumentParser(description="RAG evaluation for agentic decision task (aligned with llm_agent eval.py)")
    parser.add_argument("--memory_type", type=str, default="event", help="Type of memory to use (e.g., 'event' or 'note')")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature for response generation")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (if applicable)")
    parser.add_argument("--top_k", type=int, default=int(os.getenv("RAG_TOP_K", "16")))
    parser.add_argument("--prefetch_k", type=int, default=int(os.getenv("RAG_PREFETCH_K", "200")))
    parser.add_argument("--include_cutoff", action="store_true", default=True)
    parser.add_argument("--require_timestamp", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    question_dir = cfg.QUESTIONS_DIR
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
        
    log_name = (f"rag_eval_{args.memory_type}_{args.temperature}_{args.model}"
                f"_{args.top_k}_{args.prefetch_k}{"_include_cutoff" if args.include_cutoff else ""}"
                f"{"_require_timestamp" if args.require_timestamp else ""}{"_debug" if args.debug else ""}.json"
    )
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
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        },
        "embedding": {
            "input_tokens": 0,
            "total_tokens": 0
        },
    }
    total_log = existing_log
    safe_mkdir(out_dir)
    with ProcessPoolExecutor(max_workers=min(cfg.MAXWORKERS, len(qfiles))) as executor:
        futures = {
            executor.submit(
                run_one_visit_rag,
                qf,
                memory_type=args.memory_type,
                temperature=args.temperature,
                model=args.model,
                top_k=args.top_k,
                prefetch_k=args.prefetch_k,
                include_cutoff=args.include_cutoff,
                require_timestamp=args.require_timestamp,
                debug=args.debug,
            ): qf
            for qf in qfiles
        }
        for future in as_completed(futures):
            qf = futures[future]
            try:
                log = future.result()
                logger.info(f"Completed {qf}: {json.dumps(log, ensure_ascii=False, indent=2)}")
                # accumulate usage
                for k, v in log.get("usage", {}).get("chat", {}).items():
                    total_usage["chat"][k] += v
                for k, v in log.get("usage", {}).get("embedding", {}).items():
                    total_usage["embedding"][k] += v
                total_log[qf.name] = log
                with open(out_dir / log_name, "w", encoding="utf-8") as f:
                    json.dump(total_log, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error processing {qf}: {e}")
                
    # for qf in qfiles:
    #     try:
    #         log = run_one_visit_rag(
    #             qf,
    #             memory_type=args.memory_type,
    #             temperature=args.temperature,
    #             model=args.model,
    #             top_k=args.top_k,
    #             prefetch_k=args.prefetch_k,
    #             include_cutoff=args.include_cutoff,
    #             require_timestamp=args.require_timestamp,
    #             debug=args.debug,
    #         )
    #         logger.info(f"Completed {qf}: {json.dumps(log, ensure_ascii=False, indent=2)}")
    #         # accumulate usage
    #         for k, v in log.get("usage", {}).get("chat", {}).items():
    #             total_usage["chat"][k] += v
    #         for k, v in log.get("usage", {}).get("embedding", {}).items():
    #             total_usage["embedding"][k] += v
    #         total_log[qf.name] = log
    #         with open(out_dir / log_name, "w", encoding="utf-8") as f:
    #             json.dump(total_log, f, ensure_ascii=False, indent=2)
    #     except Exception as e:
    #         logger.error(f"Error processing {qf}: {e}")
    logger.info(f"Total LLM Usage: {json.dumps(total_usage, ensure_ascii=False, indent=2)}")



if __name__ == "__main__":
    main()
