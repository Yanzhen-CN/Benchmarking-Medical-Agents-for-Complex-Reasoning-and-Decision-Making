#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import uuid
import hashlib
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from util import logUtil
logger = logUtil.setup_logger()

from config import AgentTaskConfig
cfg = AgentTaskConfig()

from util.llmUtil import LLMUtil
# mem0 agent
from agents.mem0_agent import (
    MemoryAugmentedChatAgent,
    OpenAICompatibleLLMProvider,
    Mem0MemoryProvider,
    LLMObservationExtractor,
)
from agents.mem0_agent.core import AgentConfig

# context+memory API
from tasks.agentic_decision.get_messages_for_eval import get_memory_and_context_for_qid

# eval utils (与你 rag eval.py 一致用法)
from tasks.agentic_decision.eval_utils import (
    iter_jsonl,
    format_mcq_user_content,
    normalize_messages,
    score_weighted_acc,
)
from tasks.agentic_decision.prompts import get_agent_qa_prompt, AGENT_ACTION_PROMPT

# --------------------------
# output dir (aligned)
# --------------------------
out_dir = Path("./agents/mem0_agent/agentic_decision/results")


# ============================================================
# Small IO helpers
# ============================================================
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# helpers (qid parse / cutoff parse)
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


# ============================================================
# mem0 build agent
# ============================================================
def build_agent(
    *,
    max_recent_turns: int,
    memory_top_k: int,
    retrieval_policy: str,
    query_rewrite: bool,
    llm: Optional[OpenAICompatibleLLMProvider] = None,
) -> Tuple[MemoryAugmentedChatAgent, Mem0MemoryProvider, OpenAICompatibleLLMProvider]:
    llm = OpenAICompatibleLLMProvider() if llm is None else llm
    mem = Mem0MemoryProvider()
    obs = LLMObservationExtractor(llm)

    cfg_agent = AgentConfig(
        max_recent_turns=max_recent_turns,
        memory_top_k=memory_top_k,
        store_dialog=False,
        store_observations=False,
        include_memory_in_prompt=True,  # ✅ mem0 agent 自己检索 topk 并 append prompt
        retrieval_policy=retrieval_policy,
        query_rewrite=query_rewrite,
    )

    agent = MemoryAugmentedChatAgent(
        llm=llm,
        memory=mem,
        observation_extractor=obs,
        config=cfg_agent,
    )
    return agent, mem, llm


# ============================================================
# mem0 store helpers (threadpool, de-dup)
# ============================================================
def store_fact(
    mem: Mem0MemoryProvider,
    text: str,
    *,
    user_id: str,
    agent_id: str,
    app_id: str,
    run_id: str,
) -> None:
    mem.add_memory(
        text=text,
        metadata={"type": "fact"},
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
    )


def flush_memories(
    mem: Mem0MemoryProvider,
    *,
    user_id: str,
    agent_id: str,
    app_id: str,
    run_id: str,
) -> None:
    try:
        mem.delete_all(user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)
    except Exception as exc:
        print(f"[warn] Memory cleanup failed: {exc}", file=sys.stderr)


def _fingerprint_memory(m: Any) -> str:
    if isinstance(m, str):
        s = m.strip()
    else:
        s = json.dumps(m, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _normalize_memory_text(m: Any) -> str:
    if isinstance(m, str):
        return m.strip()
    if isinstance(m, (dict, list)):
        return json.dumps(m, ensure_ascii=False)
    return str(m).strip()


def store_facts_concurrent(
    mem: Mem0MemoryProvider,
    memories: List[Any],
    *,
    user_id: str,
    agent_id: str,
    app_id: str,
    run_id: str,
    max_workers: int = 10,
    chunk_size: int = 16,
    retry: int = 2,
) -> int:
    texts: List[str] = []
    for m in memories:
        t = _normalize_memory_text(m)
        if t:
            texts.append(t)
    if not texts:
        return 0

    def _write_one(text: str) -> bool:
        for attempt in range(retry + 1):
            try:
                store_fact(mem, text, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)
                return True
            except Exception:
                if attempt >= retry:
                    return False
                time.sleep(0.8 * (2**attempt) + random.random() * 0.2)
        return False

    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(lambda buf=ch: sum(_write_one(x) for x in buf)) for ch in chunks]
        for fut in as_completed(futs):
            ok += int(fut.result() or 0)
    return ok


# ============================================================
# reply normalize: pred -> List[str]
# ============================================================
def _reply_to_pred_list(reply: Any) -> List[str]:
    """
    统一成 List[str]，与 rag eval 保持一致。
    兼容：
    - reply 是 dict 且包含 "answer"
    - reply 是 str
    - reply 是 list
    - 其它对象
    """
    if isinstance(reply, dict) and "answer" in reply:
        pred = reply.get("answer")
    else:
        pred = reply

    if isinstance(pred, str):
        pred_list = [pred]
    elif isinstance(pred, list):
        pred_list = [str(x) for x in pred]
    else:
        pred_list = [str(pred)]
    pred_list = [x.strip() for x in pred_list if str(x).strip()]
    return pred_list if pred_list else [""]


# ============================================================
# One patient file
# ============================================================
def run_one_visit_mem0(
    questions_jsonl: Path,
    *,
    memory_type: str = "event",
    top_k: int = 16,
    max_recent_turns: int = 16,
    retrieval_policy: str = "question_only",
    query_rewrite: bool = True,
    index_wait_s: float = 0.0,
    disable_store_memories: bool = False,
    temperature: float = 0.0,
    model: Optional[str] = None,
    enable_thinking: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    qpath = questions_jsonl
    if not qpath.exists():
        raise SystemExit(f"Missing: {qpath}")
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
        _parse_patient_id_from_qid(str(first.get("qid", "")))
        if isinstance(first, dict) and first.get("qid")
        else qpath.stem
    )

    app_id = os.getenv("AGENT_APP_ID", "medagentbench")
    agent_id = os.getenv("AGENT_ID", "bench-agent")
    user_id = patient_id
    llm = OpenAICompatibleLLMProvider()

    agent, mem, llm = build_agent(
        max_recent_turns=max_recent_turns,
        memory_top_k=top_k,
        retrieval_policy=retrieval_policy,
        query_rewrite=query_rewrite,
        llm=llm,
    )

    # patient-level: clean once
    flush_memories(mem, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)

    scores_by_type: Dict[str, List[float]] = defaultdict(list)
    records: List[Dict[str, Any]] = []

    # incremental memory cache
    seen_fp = set()
    last_mem_len = 0
    total_added = 0

    questions = list(iter_jsonl(qpath))
    for q in tqdm(questions, desc=f"Mem0 Eval {qpath.name} ({memory_type})"):
        qid = q.get("qid")
        qtype = q.get("qtype", "UNKNOWN")
        options = q.get("options") or []
        gt = q.get("answer")

        if not isinstance(qid, str) or not qid:
            continue
        if not isinstance(options, list):
            options = []

        # 1) context + memories
        pack = get_memory_and_context_for_qid(qid=qid, memory_type=memory_type)
        memories = pack.get("memories") or []
        context_messages = pack.get("context_messages") or []
        if not isinstance(memories, list):
            memories = []
        if not isinstance(context_messages, list):
            context_messages = []

        # 2) cutoff info (保持字段对齐 rag)
        visit_id = pack.get("visit_id")
        cmeta = pack.get("context_meta") or {}
        cutoff_event_id = cmeta.get("cutoff_event_id")

        visible_until_visit_idx: Optional[int] = None
        if cutoff_event_id:
            visible_until_visit_idx = None
        else:
            visible_until_visit_idx = _visible_until_visit_idx_from_visit_id(visit_id)

        # 3) incremental store to mem0 (optional)
        cand_new = memories[last_mem_len:] if last_mem_len <= len(memories) else memories
        last_mem_len = len(memories)

        new_unique = []
        for m in cand_new:
            fp = _fingerprint_memory(m)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            new_unique.append(m)

        if (not disable_store_memories) and new_unique:
            ok_n = store_facts_concurrent(
                mem,
                new_unique,
                user_id=user_id,
                agent_id=agent_id,
                app_id=app_id,
                run_id=run_id,
                max_workers=int(os.getenv("MEM0_WRITE_WORKERS", "10")),
                chunk_size=int(os.getenv("MEM0_WRITE_CHUNK", "20")),
                retry=int(os.getenv("MEM0_WRITE_RETRY", "2")),
            )
            total_added += ok_n

        if index_wait_s > 0:
            time.sleep(index_wait_s)

        # 4) build messages (对齐 rag：system action prompt + ctx + qa prompt + user)
        user_content = format_mcq_user_content(q)
        ctx = normalize_messages(context_messages, max_chars=cfg.CTX_CHARS)

        messages = (
            [{"role": "system", "content": AGENT_ACTION_PROMPT}]
            + (ctx[1:] if len(ctx) > 1 else [])
            + [
                get_agent_qa_prompt(),
                {"role": "user", "content": user_content},
            ]
        )
        messages = normalize_messages(messages, max_chars=cfg.CTX_CHARS)

        # 5) call mem0 agent
        try:
            if debug:
                reply, trace = agent.chat_with_trace(
                    messages=messages,
                    user_id=user_id,
                    agent_id=agent_id,
                    app_id=app_id,
                    run_id=run_id,
                    json=True,
                    model = model,
                    temperature = temperature,
                    enable_thinking = enable_thinking,
                )
                retrieval_query = getattr(trace, "retrieval_query", None)
                retrieved_texts = [r.text for r in (getattr(trace, "memories", None) or [])]
            else:
                reply = agent.chat(
                    messages=messages,
                    user_id=user_id,
                    agent_id=agent_id,
                    app_id=app_id,
                    run_id=run_id,
                    json=True,
                    model = model,
                    temperature = temperature,
                    enable_thinking = enable_thinking,
                )
                retrieval_query, retrieved_texts = None, []

            pred_list = _reply_to_pred_list(reply)
        except Exception as e:
            records.append(
                {
                    "qid": qid,
                    "qtype": qtype,
                    "memory_type": memory_type,
                    "error": str(e),
                    "patient_id": patient_id,
                    "visit_id": visit_id,
                    "cutoff_event_id": cutoff_event_id,
                    "visible_until_visit_idx": visible_until_visit_idx,
                    "mem_added_this_q": len(new_unique),
                    "mem_total_added": total_added,
                }
            )
            continue

        # 6) scoring (一致：pred_list)
        score = score_weighted_acc(gt, pred_list=pred_list)
        scores_by_type[qtype].append(score)

        rec = {
            "qid": qid,
            "qtype": qtype,
            "memory_type": memory_type,
            "patient_id": patient_id,
            "visit_id": visit_id,
            "cutoff_event_id": cutoff_event_id,
            "visible_until_visit_idx": visible_until_visit_idx,
            "pred": pred_list,
            "reply": reply,
            "score": score,
            "gt": gt,
            # debug/retrieval trace
            "retrieval_query": retrieval_query,
            "retrieved_topk": retrieved_texts[:top_k],
            # mem0 store stats
            "mem_added_this_q": len(new_unique),
            "mem_total_added": total_added,
        }
        records.append(rec)

        if debug:
            logger.info(f"[{qid}] qtype={qtype} score={score} pred={pred_list} gt={gt}")

    # end: cleanup
    flush_memories(mem, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)

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
        "mem0": {
            "top_k": top_k,
            "max_recent_turns": max_recent_turns,
            "retrieval_policy": retrieval_policy,
            "query_rewrite": query_rewrite,
            "index_wait_s": index_wait_s,
            "disable_store_memories": disable_store_memories,
        },
        # 兼容 rag 的 usage 字段：如果 llm provider 有 get_token_usage 就写；否则给空
        "usage": (llm.get_token_usage()),
    }

    # write outputs
    safe_mkdir(out_dir)
    pred_path = out_dir / f"{patient_id}.{memory_type}.{run_id}.mem0.pred.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sum_path = out_dir / f"{patient_id}.{memory_type}.{run_id}.mem0.summary.json"
    sum_path.write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # plot
    order = ["T3-N", "T3-A", "T3-M", "T3-D"]
    keys = sorted(summary.keys(), key=lambda k: (order.index(k) if k in order else 999, k))
    vals = [summary[k]["acc"] for k in keys]

    fig_path = out_dir / f"{patient_id}.{memory_type}.{run_id}.mem0.acc_by_type.png"
    plt.figure(figsize=(7, 4))
    plt.bar(keys, vals)
    plt.ylim(0, 1.0)
    plt.xlabel("Question Type")
    plt.ylabel("Weighted Accuracy")
    plt.title(f"Mem0 Acc by Type ({patient_id}, {memory_type})")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info(f"[DONE] patient={patient_id} memory_type={memory_type} overall_acc={overall:.4f}")
    logger.info(f"predictions: {pred_path}")
    logger.info(f"summary:     {sum_path}")
    logger.info(f"figure:      {fig_path}")

    return summary_out

task_cfg = AgentTaskConfig()

# ============================================================
# Main (multi-patient + resume)  —— 对齐 rag eval.py
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Mem0 evaluation for agentic decision task (aligned with rag eval.py)")
    parser.add_argument("--model", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--memory_type", type=str, default="event", choices=["event", "note"])
    parser.add_argument("--disable_store_memories", action="store_true", default=False)
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
        qfiles = qfiles[: cfg.DEMO_N]
        logger.info(f"DEMO MODE: Only processing {len(qfiles)} files")
    else:
        logger.info(f"Found {len(qfiles)} question files to process")

    safe_mkdir(out_dir)

    log_name = (
        f"mem0_eval_{args.model}_{args.temperature}{ '_thinking' if args.enable_thinking else ''}_{args.memory_type}"
        f"{'_no_store' if args.disable_store_memories else ''}"
        f"{'_debug' if args.debug else ''}.json"
    )

    log_path = out_dir / log_name
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            existing_log = json.load(f)
        done_files = set(existing_log.keys())
        qfiles = [qf for qf in qfiles if qf.name not in done_files]
        logger.info(f"Resuming from existing log. {len(qfiles)} files left to process.")
    else:
        existing_log = {}
        logger.info("No existing log found. Starting fresh.")

    total_usage = {
        "chat": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "embedding": {"input_tokens": 0, "total_tokens": 0},
    }
    total_log: Dict[str, Any] = dict(existing_log)

    if not qfiles:
        logger.info("Nothing to do (all files already completed).")
        return

    # 并行
    # with ProcessPoolExecutor(max_workers=min(cfg.MAXWORKERS, len(qfiles))) as executor:
    #     futures = {
    #         executor.submit(
    #             run_one_visit_mem0,
    #             qf,
    #             memory_type=args.memory_type,
    #             enable_thinking=args.enable_thinking,
    #             model=args.model,
    #             temperature=args.temperature,
    #             top_k=cfg.MAX_KNOWN_FACTS,
    #             max_recent_turns=cfg.KEEP_LAST_N_TURNS,
    #             retrieval_policy=cfg.MEM0_RETRIVAL_POLICY,
    #             query_rewrite=cfg.QUERY_REWRITE,
    #             index_wait_s=cfg.MEM0_INDEX_WAIT_S,
    #             disable_store_memories=args.disable_store_memories,
    #             debug=args.debug,
    #         ): qf
    #         for qf in qfiles
    #     }
    #     for future in as_completed(futures):
    #         qf = futures[future]
    #         try:
    #             log = future.result()
    #             logger.info(f"Completed {qf}: {json.dumps(log, ensure_ascii=False, indent=2)}")

    #             # accumulate usage (与 rag eval 同结构)
    #             usage = log.get("usage", {}) or {}
    #             for k, v in (usage.get("chat", {}) or {}).items():
    #                 if k in total_usage["chat"]:
    #                     total_usage["chat"][k] += int(v or 0)
    #             for k, v in (usage.get("embedding", {}) or {}).items():
    #                 if k in total_usage["embedding"]:
    #                     total_usage["embedding"][k] += int(v or 0)

    #             total_log[qf.name] = log
    #             with log_path.open("w", encoding="utf-8") as f:
    #                 json.dump(total_log, f, ensure_ascii=False, indent=2)

    #         except Exception as e:
    #             logger.error(f"Error processing {qf}: {e}")
    
    for qf in qfiles:
        try:
            log = run_one_visit_mem0(
                qf,
                memory_type=args.memory_type,
                enable_thinking=args.enable_thinking,
                model=args.model,
                temperature=args.temperature,
                top_k=cfg.MAX_KNOWN_FACTS,
                max_recent_turns=cfg.KEEP_LAST_N_TURNS,
                retrieval_policy=cfg.MEM0_RETRIVAL_POLICY,
                query_rewrite=cfg.QUERY_REWRITE,
                index_wait_s=cfg.MEM0_INDEX_WAIT_S,
                disable_store_memories=args.disable_store_memories,
                debug=args.debug,
            )
            logger.info(f"Completed {qf}: {json.dumps(log, ensure_ascii=False, indent=2)}")

            # accumulate usage (与 rag eval 同结构)
            usage = log.get("usage", {}) or {}
            for k, v in (usage.get("chat", {}) or {}).items():
                if k in total_usage["chat"]:
                    total_usage["chat"][k] += int(v or 0)
            for k, v in (usage.get("embedding", {}) or {}).items():
                if k in total_usage["embedding"]:
                    total_usage["embedding"][k] += int(v or 0)

            total_log[qf.name] = log
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(total_log, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error processing {qf}: {e}")

    logger.info(f"Total LLM Usage: {json.dumps(total_usage, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
