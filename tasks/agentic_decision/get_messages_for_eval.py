# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
import json
from tasks.agentic_decision.tools import *


# ============================================================
# Turn indexing + truncation (你给的代码，原封不动整合)
# ============================================================

def _is_system(m: Dict[str, Any]) -> bool:
    return m.get("role") == "system"

def _is_assistant_action(m: Dict[str, Any]) -> bool:
    if m.get("role") != "assistant":
        return False
    c = m.get("content")
    return isinstance(c, dict) and ("action" in c) and ("reason" in c)

def _is_env_obs(m: Dict[str, Any]) -> bool:
    if m.get("role") != "user":
        return False
    c = m.get("content")
    return isinstance(c, dict) and (c.get("name") == "environment" or c.get("name") == "env")

def index_turns(messages: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, end_idx_exclusive) for each turn.
    Turn starts at an assistant action message; include its following env obs if present.
    """
    turns: List[Tuple[int, int]] = []
    i = 0
    n = len(messages)
    while i < n:
        if _is_assistant_action(messages[i]):
            start = i
            end = i + 1
            # include immediate env obs if present
            if end < n and _is_env_obs(messages[end]):
                end += 1
            turns.append((start, end))
            i = end
        else:
            i += 1
    return turns

def find_turn_by_event_id(messages: List[Dict[str, Any]], cutoff_event_id: str) -> Optional[int]:
    """
    Find the turn index whose env observation has event_id == cutoff_event_id.
    Returns turn_idx if found, else None.
    """
    turns = index_turns(messages)
    for t_idx, (s, e) in enumerate(turns):
        for j in range(s, e):
            m = messages[j]
            if _is_env_obs(m):
                c = m.get("content") or {}
                if str(c.get("event_id")) == str(cutoff_event_id):
                    return t_idx
    return None

def truncate_messages_for_eval(
    messages: List[Dict[str, Any]],
    *,
    cutoff_turn_idx: Optional[int] = None,
    cutoff_event_id: Optional[str] = None,
    keep_last_n_turns: int = 8,
) -> List[Dict[str, Any]]:
    """
    Keep:
      - all leading system messages (from beginning, contiguous systems)
      - plus the last N turns BEFORE the cutoff turn (exclusive)

    Cutoff priority:
      1) cutoff_turn_idx (t_index)
      2) cutoff_event_id -> locate turn_idx
      3) if none provided / not found -> use end (i.e., cutoff = num_turns)

    NOTE: Drops the admission summary unless it falls into kept turn window
          (per requirement: "只保留system + 前N轮").
    """
    if keep_last_n_turns < 0:
        keep_last_n_turns = 0

    # 1) keep contiguous leading system prompts
    sys_end = 0
    while sys_end < len(messages) and _is_system(messages[sys_end]):
        sys_end += 1
    system_prefix = messages[:sys_end]

    # 2) build turns
    turns = index_turns(messages)
    num_turns = len(turns)

    # 3) resolve cutoff_turn_idx
    if cutoff_turn_idx is None and cutoff_event_id:
        cutoff_turn_idx = find_turn_by_event_id(messages, cutoff_event_id)

    if cutoff_turn_idx is None:
        cutoff_turn_idx = num_turns
    else:
        cutoff_turn_idx = max(0, min(int(cutoff_turn_idx), num_turns))

    # 4) compute kept window: [start_turn, cutoff_turn_idx)
    start_turn = max(0, cutoff_turn_idx - keep_last_n_turns)
    kept_spans = turns[start_turn:cutoff_turn_idx]

    # 5) slice messages by spans
    sliced: List[Dict[str, Any]] = []
    for (s, e) in kept_spans:
        sliced.extend(messages[s:e])

    return system_prefix + sliced


# ============================================================
# Context loader + Question loader + Unified API
# ============================================================

@dataclass
class MemoryContextConfig:
    questions_jsonl: Path                 # e.g., tasks/.../P000001.jsonl
    patient_json: Path                    # e.g., bench_data/patients/P000001.json
    sequenced_json: Path                  # e.g., bench_data/patients/P000001_sequenced.json
    context_root: Path                    # e.g., tasks/agentic_decision/context
    keep_last_n_turns: int = 8

def _jsonl_iter(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            
def load_question_by_qid(questions_jsonl: Path, qid: str) -> Dict[str, Any]:
    for q in _jsonl_iter(questions_jsonl):
        if q.get("qid") == qid:
            return q
    raise FileNotFoundError(f"qid not found in questions jsonl: {qid}")


def load_context_messages_for_visit(context_root: Path, patient_id: str, visit_id: str) -> List[Dict[str, Any]]:
    """
    context 文件约定：{context_root}/{patient_id}/{visit_id}.context.json
    例如：tasks/agentic_decision/context/P000001/P000001-V12.context.json
    """
    p = context_root / patient_id / f"{visit_id}.context.json"
    if not p.exists():
        raise FileNotFoundError(f"context file not found for visit: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        raise ValueError(f"invalid context schema (missing messages list): {p}")
    return msgs


def _get_memory_and_context_for_question(
    cfg: MemoryContextConfig,
    *,
    qid: str,
    memory_type: str,          # "note" | "event"
    cutoff_event_id: Optional[str] = None,  # 可选：如果你未来想用 event_id 作为截断点
) -> Dict[str, Any]:
    """
    输入：qid + memory_type
    输出：
      - question: 原始题目 dict
      - memories: 前序记忆列表（note 或 event）
      - context_messages: 当前 visit 的截断上下文（system + 截断前 N turn）
    """
    q = load_question_by_qid(cfg.questions_jsonl, qid)

    visit_id = q.get("visit_id")
    if not isinstance(visit_id, str) or not visit_id:
        raise ValueError(f"question missing visit_id: {qid}")

    # patient_id 约定：visit_id "P000001-V12" -> patient_id "P000001"
    patient_id = visit_id.split("-V")[0]
    current_visit_ref = "V" + visit_id.split("-V")[1]  # "V12"

    # 1) 当前 visit context（若找不到直接报错：满足你的假设）
    full_messages = load_context_messages_for_visit(cfg.context_root, patient_id, visit_id)

    # 截断点：优先用显式 cutoff_event_id；否则用题目里的 t_index
    cutoff_turn_idx = q.get("t_index")
    if cutoff_event_id is None:
        # 允许你把它塞在 question.meta 里（如果将来需要）
        meta = q.get("meta") or {}
        if isinstance(meta, dict) and meta.get("cutoff_event_id"):
            cutoff_event_id = str(meta["cutoff_event_id"])

    if cutoff_event_id is None:
        if cutoff_turn_idx is None:
            raise ValueError(f"question missing both t_index and cutoff_event_id: {qid}")
        try:
            cutoff_turn_idx = int(cutoff_turn_idx)
        except Exception:
            raise ValueError(f"invalid t_index in question: {qid}, t_index={cutoff_turn_idx}")

    context_messages = truncate_messages_for_eval(
        full_messages,
        cutoff_turn_idx=cutoff_turn_idx if cutoff_event_id is None else None,
        cutoff_event_id=cutoff_event_id,
        keep_last_n_turns=cfg.keep_last_n_turns,
    )

    # 2) 前序记忆
    if memory_type not in ("note", "event"):
        raise ValueError(f"unknown memory_type: {memory_type} (expected 'note'|'event')")

    patient_obj = json.loads(cfg.patient_json.read_text(encoding="utf-8"))
    visits = patient_obj.get("visits") or []
    visit_order = {v.get("visit_id", "").split("-V")[-1].join(["V"]) if False else None: -1}  # placeholder

    # visit_order: {"V1":0, "V2":1, ...} —— 从 patient_json 的 visits 顺序构建
    visit_order = {}
    for i, v in enumerate(visits):
        vid = v.get("visit_id")
        if not isinstance(vid, str) or "-V" not in vid:
            continue
        vref = "V" + vid.split("-V")[1]
        visit_order[vref] = i

    if memory_type == "note":
        memories = build_note_memory_from_patient_json(patient_obj, current_visit_id=visit_id)
    else:
        seq_events = json.loads(cfg.sequenced_json.read_text(encoding="utf-8"))
        if not isinstance(seq_events, list):
            raise ValueError(f"invalid sequenced json, expected list: {cfg.sequenced_json}")
        memories = build_event_memory_from_sequence_json(
            seq_events,
            visit_order=visit_order,
            current_visit_ref=current_visit_ref,
        )

    return {
        "qid": qid,
        "visit_id": visit_id,
        "memory_type": memory_type,
        "question": q,
        "memories": memories,
        "context_messages": context_messages,
        "context_meta": {
            "cutoff_turn_idx": cutoff_turn_idx if cutoff_event_id is None else None,
            "cutoff_event_id": cutoff_event_id,
            "keep_last_n_turns": cfg.keep_last_n_turns,
        },
    }
    

def get_memory_and_context_for_qid(qid: str, memory_type: str) -> Dict[str, Any]:
    """
    包装一层，方便外部调用。
    # context 不包括问题，只负责根据 qid 定位到对应 visit 的上下文文件，加载并截断。
    """
    result = parse_qid(qid)
    patient_id = result["patient_id"]
    visit_id = result["visit_id"]
    cfg = MemoryContextConfig(
        questions_jsonl=Path(f"tasks/agentic_decision/questions_generated/{patient_id}.jsonl"),
        patient_json=Path(f"bench_data/patients/{patient_id}.json"),
        sequenced_json=Path(f"bench_data/patients_sequence/{patient_id}_sequenced.json"),
        context_root=Path("tasks/agentic_decision/context"),
        keep_last_n_turns=8,
    )
    return _get_memory_and_context_for_question(cfg, qid=qid, memory_type=memory_type)

def get_memory_for_qid(qid: str, memory_type: str) -> List[Dict[str, Any]]:
    return get_memory_and_context_for_qid(qid, memory_type).get("memories", [])

def get_context_for_qid(qid: str, memory_type: str) -> List[Dict[str, Any]]:
    return get_memory_and_context_for_qid(qid, memory_type).get("context_messages", [])

# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    
    out = get_memory_and_context_for_qid(
        qid="P000001-V12-T0002-T3-N-0",
        memory_type="event",  # or "note"
    )
    print("question:", out["question"]["question"])
    print(json.dumps(out, indent=2))

    print("memories:", len(out["memories"]))
    print("context_messages:", len(out["context_messages"]))
