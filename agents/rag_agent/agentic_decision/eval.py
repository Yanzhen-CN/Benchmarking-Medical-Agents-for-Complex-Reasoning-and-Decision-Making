# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util import logUtil
logger = logUtil.setup_logger()

from util.llmUtil import LLMUtil
from tasks.agentic_decision.get_messages_for_eval import get_memory_and_context_for_qid
from retrieval.retriever import PatientRetriever, RetrievedDoc


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

    # exact
    for opt in options:
        if isinstance(opt, str) and opt.lower().strip() == r_low:
            return opt

    # substring
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
        # 轻量 meta，方便 debug
        tag2 = f"visit={meta.get('visit_ref')} type={meta.get('event_type', meta.get('memory_type'))} id={meta.get('event_id', meta.get('note_type'))}"
        lines.append(f"[{i}] ({tag}; {tag2}) {t}")
    return "\n".join(lines)


def _parse_patient_id_from_qid(qid: str) -> str:
    # qid: P000001-V12-...
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


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_jsonl", type=str, required=True)
    ap.add_argument("--memory_type", type=str, default="event", choices=["event", "note"])
    ap.add_argument("--top_k", type=int, default=int(os.getenv("RAG_TOP_K", "8")))
    ap.add_argument("--prefetch_k", type=int, default=int(os.getenv("RAG_PREFETCH_K", "200")))
    ap.add_argument("--include_cutoff", action="store_true", default=False)
    ap.add_argument("--require_timestamp", action="store_true", default=False)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out_dir", type=str, default="log/rag_eval")
    ap.add_argument("--model", type=str, default=os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    qpath = Path(args.questions_jsonl)
    if not qpath.exists():
        raise SystemExit(f"Missing questions file: {qpath}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # run id
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = uuid.uuid4().hex[:8]
    run_id = f"{ts}-{rid}"

    # init retriever + llm
    retriever = PatientRetriever()
    llm = LLMUtil()

    # infer patient_id
    first = next(iter_jsonl(qpath))
    patient_id = _parse_patient_id_from_qid(str(first.get("qid", qpath.stem)))

    scores_by_type: Dict[str, List[float]] = defaultdict(list)
    records: List[Dict[str, Any]] = []

    # --- eval loop ---
    for q in tqdm(list(iter_jsonl(qpath)), desc=f"RAG Eval {qpath.name} ({args.memory_type})"):
        qid = q.get("qid")
        qtype = q.get("qtype", "UNKNOWN")
        options = q.get("options") or []
        gt = q.get("answer")

        if not isinstance(qid, str) or not qid:
            continue

        # 1) get context pack (for cutoff + context_messages)
        pack = get_memory_and_context_for_qid(qid=qid, memory_type=args.memory_type)
        context_messages = pack.get("context_messages") or []
        if not isinstance(context_messages, list):
            context_messages = []

        visit_id = pack.get("visit_id")
        cmeta = pack.get("context_meta") or {}
        cutoff_event_id = cmeta.get("cutoff_event_id")

        visible_until_visit_idx = None
        if cutoff_event_id:
            # 最精确：事件级截断
            pass
        else:
            visible_until_visit_idx = _visible_until_visit_idx_from_visit_id(visit_id)

        # 2) build query text (可根据 policy 调整：只用 question / question+options / question+context)
        query_text = str(q.get("question", "")).strip()
        if not query_text:
            query_text = str(pack.get("question", "")).strip()

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

        # 4) compose messages: context + system(evidence) + user(question+options)
        sys_block = _build_retrieved_system_block(hits)
        user_content = _format_mcq_user_content(q)

        messages = list(context_messages) + [
            {"role": "system", "content": sys_block},
            {"role": "user", "content": user_content},
        ]

        # 5) call llm (你们项目里 LLMUtil 的 chat 接口可能不同：这里用最常见的 chat_text / chat_json 模式
        #    如果你的 LLMUtil 只有 openai client，请把这一段替换成你们已有的 query_model/llm.chat
        reply = llm.chat_text(messages=messages, model=args.model)

        pred = _extract_pred_option(reply, options)
        score = _score_weighted_acc(gt, pred)

        scores_by_type[qtype].append(score)

        rec = {
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
            "retrieved": [
                {
                    "score": h.score,
                    "text": h.text,
                    "meta": h.meta,
                }
                for h in hits
            ],
        }
        records.append(rec)

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
