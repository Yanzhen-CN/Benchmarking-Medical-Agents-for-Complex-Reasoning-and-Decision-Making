# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util import logUtil
logger = logUtil.setup_logger()

# 你的 mem0 agent 组件
from agents.mem0_agent import (
    MemoryAugmentedChatAgent,
    OpenAICompatibleLLMProvider,
    Mem0MemoryProvider,
    LLMObservationExtractor,
)
from agents.mem0_agent.core import AgentConfig

# 关键：你的“上下文+记忆”API（来自你上传的 get_messages_for_eval.py）
# 确保脚本运行时 sys.path 里包含项目根目录
from tasks.agentic_decision.get_messages_for_eval import get_memory_and_context_for_qid

def build_agent(
    *,
    max_recent_turns: int,
    memory_top_k: int,
    retrieval_policy: str,
    query_rewrite: bool,
) -> Tuple[MemoryAugmentedChatAgent, Mem0MemoryProvider]:
    llm = OpenAICompatibleLLMProvider()
    mem = Mem0MemoryProvider()
    obs = LLMObservationExtractor(llm)

    cfg = AgentConfig(
        max_recent_turns=max_recent_turns,
        memory_top_k=memory_top_k,
        store_dialog=False,
        store_observations=False,
        include_memory_in_prompt=True,    # ✅ 关键：让 agent 自己把 topk memory append 到 prompt
        retrieval_policy=retrieval_policy,
        query_rewrite=query_rewrite,
    )

    agent = MemoryAugmentedChatAgent(
        llm=llm,
        memory=mem,
        observation_extractor=obs,
        config=cfg,
    )
    return agent, mem


_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

def _to_str_content(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # dict / list / 其它对象
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def normalize_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        if role not in _ALLOWED_ROLES:
            # 兜底：未知 role 当成 assistant 或 tool 都行，这里当 tool
            role = "tool"
        out.append(
            {
                "role": role,
                "content": _to_str_content(m.get("content")),
            }
        )
    return out

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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
    # 你示例里 delete_all 是按 user_id/agent_id/app_id/run_id 做隔离的
    try:
        mem.delete_all(user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)
    except Exception as exc:
        print(f"[warn] Memory cleanup failed: {exc}", file=sys.stderr)


def _format_mcq_user_content(q: Dict[str, Any]) -> str:
    """
    把题目+候选项组织成最后一条 user 发言。
    兼容你现有 jsonl：question/options/qtype/...
    """
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
    """
    把模型输出归一到某个 option（尽量鲁棒）。
    - 如果回复中包含某个 option 子串，取第一个匹配
    - 否则回退为原始 reply 字符串（用于 debug）
    """
    r = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
    r_low = r.lower()

    # 精确匹配优先
    for opt in options:
        if isinstance(opt, str) and opt.lower().strip() == r_low.strip():
            return opt

    # 子串匹配
    for opt in options:
        if isinstance(opt, str) and opt.lower() in r_low:
            return opt

    # 最后：返回原文（可能导致 0 分）
    return r.strip()


def _score_weighted_acc(gt_answer: Any, pred: str) -> float:
    """
    加权 acc：
    - 若 gt_answer 是 dict: {option: weight, ...}，score = gt.get(pred, 0)
    - 若 gt_answer 是 str: score = 1 if pred==gt else 0
    """
    if isinstance(gt_answer, dict):
        return float(gt_answer.get(pred, 0.0) or 0.0)
    if isinstance(gt_answer, str):
        return 1.0 if pred == gt_answer else 0.0
    return 0.0


def _build_memory_system_prompt(retrieved: List[str]) -> str:
    """
    按你的要求：在上下文末尾增加一条 system，包含检索返回信息。
    """
    if not retrieved:
        return "Retrieved memory (top-k): <EMPTY>"

    # 适度截断防爆 token
    kept = []
    for i, t in enumerate(retrieved[:50], 1):
        tt = t if isinstance(t, str) else str(t)
        if len(tt) > 1200:
            tt = tt[:1200] + "..."
        kept.append(f"[{i}] {tt}")
    return "Retrieved memory (top-k):\n" + "\n".join(kept)

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import random

def _fingerprint_memory(m: Any) -> str:
    """稳定去重：把 memory 归一成 string 后 hash。"""
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
    """
    并发写入 memories。返回成功写入条数。
    注意：如果 Mem0MemoryProvider 本身不是线程安全的，需要改成“多进程/多 client”。
    绝大多数 http client 是线程安全的，但你们内部实现如果共享 session 也通常 OK。
    """
    # 先归一化 + 过滤空
    texts = []
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
                time.sleep(0.8 * (2 ** attempt) + random.random() * 0.2)
        return False

    # 分 chunk，减少 submit 数量
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for ch in chunks:
            futs.append(ex.submit(lambda buf=ch: sum(_write_one(x) for x in buf)))
        for fut in as_completed(futs):
            ok += int(fut.result() or 0)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_jsonl", type=str, required=True, help="e.g., tasks/.../P000001.jsonl")
    ap.add_argument("--memory_type", type=str, default="event", choices=["event", "note"])
    ap.add_argument("--top_k", type=int, default=int(os.getenv("MEMORY_TOP_K", "5")))
    ap.add_argument("--max_recent_turns", type=int, default=int(os.getenv("AGENT_MAX_RECENT_TURNS", "6")))
    ap.add_argument("--retrieval_policy", type=str, default=os.getenv("AGENT_RETRIEVAL_POLICY", "question_only"))
    ap.add_argument("--query_rewrite", action="store_true", default=(os.getenv("AGENT_QUERY_REWRITE", "1") == "1"))
    ap.add_argument("--index_wait_s", type=float, default=float(os.getenv("MEM0_INDEX_WAIT_S", "0")))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no_delete", action="store_true", help="Do not delete memories after each question (debug only)")
    ap.add_argument("--out_dir", type=str, default="log/mem0_eval")
    args = ap.parse_args()

    qpath = Path(args.questions_jsonl)
    if not qpath.exists():
        raise SystemExit(f"Missing questions file: {qpath}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # run_id：一次评测一套
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = uuid.uuid4().hex[:8]
    run_id = f"{ts}-{rid}"

    app_id = os.getenv("AGENT_APP_ID", "medagentbench")
    agent_id = os.getenv("AGENT_ID", "bench-agent")

    # 逐题清理 memory：用同一个 user_id（比如 patient_id），靠 run_id 隔离；或干脆用 qid 作为 user_id 也行
    # 这里用 patient_id，方便一份 log 对一个 patient 文件
    # qid 格式：P000001-V12-...
    first = next(iter_jsonl(qpath))
    patient_id = str(first.get("qid", "")).split("-V")[0] if first else qpath.stem
    user_id = patient_id

    # 构建 agent：我们设 include_memory_in_prompt=False，自己在 messages 末尾追加 system memory
    agent, mem = build_agent(
        max_recent_turns=args.max_recent_turns,
        memory_top_k=args.top_k,
        retrieval_policy=args.retrieval_policy,
        query_rewrite=args.query_rewrite,
    )

    # 统计容器
    scores_by_type: Dict[str, List[float]] = defaultdict(list)
    records: List[Dict[str, Any]] = []

    # ===== patient-level cache: 只在病人内递增 =====
    # 只在一个病人开始时 flush 一次
    if not args.no_delete:
        flush_memories(mem, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)

    seen_fp = set()          # hash 去重
    last_mem_len = 0         # index diff（依赖 memories 单调增长）
    total_added = 0

    for q in tqdm(list(iter_jsonl(qpath)), desc=f"Eval {qpath.name} ({args.memory_type})"):
        qid = q.get("qid")
        qtype = q.get("qtype", "UNKNOWN")
        options = q.get("options") or []
        gt = q.get("answer")

        if not isinstance(qid, str) or not qid:
            continue

        pack = get_memory_and_context_for_qid(qid=qid, memory_type=args.memory_type)
        memories = pack.get("memories") or []
        context_messages = pack.get("context_messages") or []

        if not isinstance(memories, list):
            memories = []
        if not isinstance(context_messages, list):
            context_messages = []

        # ---- 递增取新增 memories ----
        # 1) 先用 index diff（最快）
        cand_new = memories[last_mem_len:] if last_mem_len <= len(memories) else memories
        last_mem_len = len(memories)

        # 2) 再用 hash 去重兜底（避免 pack 返回不是严格 append-only）
        new_unique = []
        for m in cand_new:
            fp = _fingerprint_memory(m)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            new_unique.append(m)

        # ---- 并发写入新增 ----
        if new_unique:
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

        if args.index_wait_s > 0:
            time.sleep(args.index_wait_s)

        # ---- 组织 messages ----
        user_content = _format_mcq_user_content(q)

        # ✅ 关键：context_messages 先 normalize（dict->str, role->assistant）
        ctx = normalize_messages(context_messages)

        messages = ctx + [{"role": "user", "content": user_content}]

        # ---- 只调用一次：agent 自己检索 topk + append prompt + 回答 ----
        if args.debug:
            reply, trace = agent.chat_with_trace(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                app_id=app_id,
                run_id=run_id,
            )
            retrieval_query = getattr(trace, "retrieval_query", None)
            retrieved_texts = [r.text for r in (trace.memories or [])] if hasattr(trace, "memories") else []
        else:
            reply = agent.chat(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                app_id=app_id,
                run_id=run_id,
            )
            retrieval_query, retrieved_texts = None, []

        pred = _extract_pred_option(reply, options)
        score = _score_weighted_acc(gt, pred)

        scores_by_type[qtype].append(score)

        rec = {
            "qid": qid,
            "qtype": qtype,
            "memory_type": args.memory_type,
            "pred": pred,
            "score": score,
            "gt": gt,
            "retrieval_query": retrieval_query,
            "retrieved_topk": retrieved_texts[: args.top_k],
            "mem_added_this_q": len(new_unique),
            "mem_total_added": total_added,
        }
        records.append(rec)

    # ---- patient 结束时再 flush 一次（可选）----
    if not args.no_delete:
        flush_memories(mem, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)


    # --------------------------
    # 汇总：每类 acc + overall
    # --------------------------
    summary = {}
    all_scores = []
    for qt, ss in scores_by_type.items():
        arr = np.array(ss, dtype=float) if ss else np.zeros((0,), dtype=float)
        summary[qt] = {
            "n": int(len(ss)),
            "acc": float(arr.mean()) if len(arr) else 0.0,
        }
        all_scores.extend(ss)

    overall = float(np.mean(all_scores)) if all_scores else 0.0

    out_jsonl = out_dir / f"{patient_id}.{args.memory_type}.{run_id}.pred.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_summary = out_dir / f"{patient_id}.{args.memory_type}.{run_id}.summary.json"
    out_summary.write_text(
        json.dumps(
            {"patient_id": patient_id, "memory_type": args.memory_type, "overall_acc": overall, "by_type": summary},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------
    # 可视化：每类 acc bar
    # --------------------------
    # 排序：T3-N, T3-A, T3-M, T3-D, others
    order = ["T3-N", "T3-A", "T3-M", "T3-D"]
    keys = sorted(summary.keys(), key=lambda k: (order.index(k) if k in order else 999, k))
    vals = [summary[k]["acc"] for k in keys]

    plt.figure(figsize=(7, 4))
    plt.bar(keys, vals)
    plt.ylim(0, 1.0)
    plt.xlabel("Question Type")
    plt.ylabel("Weighted Accuracy")
    plt.title(f"Mem0Eval Acc by Type ({patient_id}, {args.memory_type})")
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
