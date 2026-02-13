import argparse
import json
import math
import re
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import Counter, defaultdict
from pathlib import Path as _Path
from util.llmUtil import LLMUtil, ChatTokenUsage
from util.logUtil import setup_logger
from config import LLMConfig
llm_cfg = LLMConfig()
logger = setup_logger()
# --------- constants ----------
ACTIONS = [
    "ask_question",
    "order_labs",
    "order_microbiology",
    "order_imaging",
    "perform_procedure",
    "medication",
    "discharge",
]

EVENT2ACTION = {
    "lab": "order_labs",
    "imaging": "order_imaging",
    "microbiology": "order_microbiology",
    "medication": "medication",
    "procedure": "perform_procedure"
}

# exp decay: 0h -> 1.0, 24h -> 0.01
_DECAY_K = math.log(100.0) / 24.0

def decay_weight(delta_hours: float) -> float:
    return float(math.exp(-_DECAY_K * max(0.0, delta_hours)))

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _norm_unit(u: str) -> str:
    u = norm(u)
    u = u.replace(" / ", "/").replace(" ", "")
    return u

def make_indicator_key(name: str, fluid: str, unit: str) -> str:
    name_n = norm(name)
    fluid_n = norm(fluid).title()
    unit_n = _norm_unit(unit)
    return f"{name_n}||{fluid_n}||{unit_n or 'NA'}"

def parse_time(ts: str) -> datetime:
    # e.g. "2160-03-23 10:00:00"
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except:
        return datetime.strptime(ts, "%Y-%m-%d")

def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
        if p.name.startswith("."):
            continue
        yield p

def iter_jsonl_files(root: Path):
    for p in root.rglob("*.jsonl"):
        if p.name.startswith("."):
            continue
        yield p
        
# --------- LLM ----------

def call_llm_json(llm: LLMUtil, prompt: str, system: str) -> Dict[str, Any]:
    return llm.chat_json(user_text=prompt, system_prompt=system)

# --------- context rendering ----------
def compact_event(ev: Dict[str, Any], max_items: int = 40) -> Dict[str, Any]:
    t_raw = ev.get("type") or ""
    t = t_raw.lower()
    out = {"type": t, "timestamp": ev.get("timestamp")}

    items = ev.get("items", []) or []

    if t == "lab":
        out["items"] = [
            {
                "name": it.get("name"),
                "fluid": it.get("fluid"),
                "unit": it.get("unit") or it.get("units"),
                "flag": it.get("flag"),
            }
            for it in items[:max_items]
        ]
    elif t == "imaging":
        # some datasets put imaging content in name/report instead of items
        out["name"] = ev.get("name")
        out["report"] = (ev.get("report") or "")[:1200]
        out["items"] = items[:min(max_items, len(items))]
    elif t == "microbiology":
        out["items"] = items[:min(max_items, len(items))]
        if "specimen" in ev:
            out["specimen"] = ev.get("specimen")
    elif t == "medication":
        out["items"] = [
            {"drug": it.get("drug"), "route": it.get("route"), "dose": it.get("dose"), "status": it.get("status")}
            for it in items[:max_items]
        ]
    elif t == "procedure":
        out["name"] = ev.get("name")
    else:
        out["items"] = items[:min(max_items, len(items))]

    return out


def build_history(visit: Dict[str, Any], upto_time: datetime) -> List[Dict[str, Any]]:
    hist = []
    for ev in visit.get("event_stream", []) or []:
        ts = parse_time(ev["timestamp"])
        if ts <= upto_time:
            hist.append(compact_event(ev))
    return hist

# --------- GT extraction ----------
def gt_action_from_event(ev: Dict[str, Any]) -> Optional[str]:
    et = (ev.get("type") or "").lower()
    # normalize uppercase types in your sample: MEDICATION/IMAGING etc.
    if et in EVENT2ACTION:
        return EVENT2ACTION[et]
    logger.error(f"Unknown event type: {et} in event at {ev.get('timestamp')}")
    return None

def infer_lab_panel_for_event(
    llm: LLMUtil,
    ev: Dict[str, Any],
    indicator_panel_map: Optional[Dict[str, str]],
) -> Tuple[Optional[List[str]], Dict[str, Any]]:
    """
    Return:
      (panel_name, debug)
    panel_name inferred by voting over indicators in this lab event.
    """
    items = ev.get("items", []) or []
    if not items:
        logger.warning(f"Lab event at {ev.get('timestamp')} has no items. Cannot infer panel.")
        return None, {"reason": "no_items"}

    prompt = f"""
You are given a lab event containing multiple lab indicators.
Infer the most likely clinical test panels/orders name for this event (open-set, no fixed enum).
You need to simulate a doctor's actual diagnostic process, obtaining the provided indicators using as few labs as possible.
You need to provide a list of abbreviations for the inspection items.
Return JSON: {{"panel":["...", ...],"reason":"..."}}.

Lab indicators:
{json.dumps([{"name":it.get("name"),"fluid":it.get("fluid"),"unit":it.get("unit") or it.get("units")} for it in items[:60]], ensure_ascii=False)}
""".strip()
    out = call_llm_json(llm, prompt, system="You are a clinical lab expert.")
    panel = out.get("panel") or []
    return panel, {"llm": out}

def infer_imaging_param_for_event(llm: LLMUtil, ev: Dict[str, Any], history_messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Return structured imaging label:
      {"modality": "...", "target": "..."}  # e.g. {"modality":"ct","target":"head"}
    Use your existing infer_imaging_modality_target_llm if available; else LLM on event items.
    """

    prompt = f"""
Infer imaging modality and body part target from this imaging event.
Return JSON only:
{{"modality":"ct|mri|xray|ultrasound|echo|doppler|pet|other","target":"free text body part"}}.

Imaging event:
{json.dumps(ev, ensure_ascii=False)}
""".strip()
    out = call_llm_json(llm, prompt, system="You are a radiology ordering expert.")
    return {
        "modality": norm(out.get("modality") or "other").lower(),
        "target": norm(out.get("target") or ""),
        "raw": out,
    }

def infer_micro_param_for_event(llm: LLMUtil, ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return {"specimen": "...", "test": "..."} open-set but suggest typical culture naming.
    """
    prompt = f"""
Infer microbiology order parameters from this microbiology event.
Return JSON only: {{"specimen":"...","test":"..."}}.

Event:
{json.dumps(ev, ensure_ascii=False)}
""".strip()
    out = call_llm_json(llm, prompt, system="You are a clinical microbiology expert.")
    return {"specimen": norm(out.get("specimen") or ""), "test": norm(out.get("test") or ""), "raw": out}

def infer_med_param_for_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return {"drug": "...", "dose": "..."}.
    If multiple meds in next event, GT can be list; for MCQ we usually pick 1 (most salient).
    We'll keep a list and let question builder pick.
    """
    meds = []
    for it in ev.get("items", []) or []:
        drug = norm(it.get("drug") or "")
        if not drug:
            continue
        meds.append(
            {
                "drug": drug,
                "dose": norm(it.get("dose") or ""),
                "route": norm(it.get("route") or ""),
                "status": norm(it.get("status") or ""),
            }
        )
    return {"meds": meds}


def infer_proc_param_for_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    For your real schema: each PROCEDURE is one event with 'name'.
    Returns:
      {"procedures": [{"procedure": str}, ...]}
    """
    name = (ev.get("name") or "").strip()
    if not name:
        return {"procedures": []}
    return {"procedures": [{"procedure": name}]}


def group_future_by_timestamp(future: List[Tuple[datetime, Dict[str, Any]]]) -> List[Tuple[datetime, List[Dict[str, Any]]]]:
    """
    future: sorted [(ts, ev), ...]
    return: [(ts, [ev1, ev2, ...]), ...] grouped by exact ts
    """
    grouped: List[Tuple[datetime, List[Dict[str, Any]]]] = []
    cur_ts = None
    cur_bucket: List[Dict[str, Any]] = []
    for ts, ev in future:
        if cur_ts is None or ts != cur_ts:
            if cur_bucket:
                grouped.append((cur_ts, cur_bucket)) # type: ignore
            cur_ts = ts
            cur_bucket = [ev]
        else:
            cur_bucket.append(ev)
    if cur_bucket:
        grouped.append((cur_ts, cur_bucket)) # type: ignore
    return grouped
# --------- distractor generation ----------
def llm_generate_options(
    llm: LLMUtil,
    qtype: str,
    stem: str,
    gt: Any,
    history: List[Dict[str, Any]],
    n_options: int,
    forbid: List[str],
    preference_hint: str,
) -> List[Any]:
    
    if n_options <= 0:
        logger.warning(f"Requested number of options is {n_options}. Returning empty list.")
        return []
    
    """
    Return list of distractor options (length = n_options - 1 typically), excluding GT.
    "options" format depends on qtype:
      - T3-N: strings in ACTIONS
      - T3-A: strings (panels or imaging orders etc.)
      - T3-M: strings (drug names)
      - T3-D: "Yes"/"No"
    """
    prompt = f"""
You will generate multiple-choice distractor options for a clinical question.
Constraints:
- Distractors must be completely irrelevant to the ground-truth (GT) for the current decision.
- Do not output GT or near-synonyms of GT.
- Output JSON only.

Question type: {qtype}
Question stem: {stem}

Ground-truth (GT):
{json.dumps(gt, ensure_ascii=False)}

Preference hints (soft):
{preference_hint}

Forbidden options (hard):
{json.dumps(forbid, ensure_ascii=False)}

Patient history (events up to current time):
{json.dumps(history, ensure_ascii=False)}

Return JSON:
{{
  "options": ["opt1","opt2",...],
  "rationales": ["why opt1 is irrelevant", ...]
}}
Generate exactly {n_options} options.
""".strip()

    out = call_llm_json(llm, prompt, system="You design adversarial-yet-plausible multiple-choice distractors for clinicians.")
    opts = out.get("options", []) or []
    cleaned = []
    seen = set()
    for x in opts:
        s = norm(str(x))
        if not s:
            continue
        if s in seen:
            continue
        if s in forbid:
            continue
        seen.add(s)
        cleaned.append(s)
        if len(cleaned) >= n_options:
            break
    return cleaned

def generate_reason_from_messages_llm(
    llm: LLMUtil,
    messages_so_far: List[Dict[str, Any]],
    current_action: str,
    current_args: Dict[str, Any],
    model: str = llm_cfg.chat_model,
    max_history_msgs: int = 8,   # 控制上下文长度：只取最近 N 条
) -> str:
    """
    Generate assistant 'reason' based ONLY on the prefix messages (no extra summary).
    Returns "" if fails/unsafe; caller should fallback to template reason.
    """

    # 只给最近N条，避免上下文过长；保留所有system也可以，但通常2条system + 最近历史足够
    # 这里做一个简单策略：保留所有 system + 最近 (max_history_msgs) 条非-system
    sys_msgs = [m for m in messages_so_far if m.get("role") == "system"]
    non_sys = [m for m in messages_so_far if m.get("role") != "system"]
    clipped = sys_msgs + non_sys[-max_history_msgs:]

    system_prompt = (
        "You write the 'reason' field for the next assistant action in a doctor-agent dialogue.\n"
        "Hard constraints:\n"
        "- You may ONLY use information present in the provided prefix_messages.\n"
        "- Do NOT invent new patient facts, findings, diagnoses, or outcomes.\n"
        "- Do NOT mention results of the current action (results are not available yet).\n"
        "- Keep it 1-2 sentences, not exceeding 50 words, workflow-rationale focused.\n"
        "Output ONLY JSON: {\"reason\": string, \"constraints_ok\": boolean}.\n"
    )

    user_text = json.dumps({
        "prefix_messages": clipped,
        "current_decision": {
            "action": current_action,
            "args": current_args
        }
    }, ensure_ascii=False)

    obj = llm.chat_json(
        system_prompt=system_prompt,
        user_text=user_text,
        model=model,
        temperature=0.0,
        strict_only_json=True
    )

    if not isinstance(obj, dict) or obj.get("constraints_ok") is not True:
        return ""

    reason = obj.get("reason")
    if not isinstance(reason, str):
        return ""

    reason = reason.strip()
    if not reason:
        return ""
    logger.debug(f"Generated reason from LLM: {reason}")
    # # 轻量防“提前知道结果”的措辞（可以按你们数据集再扩展）
    # banned = ["revealed", "showed", "confirmed", "consistent with", "positive for", "negative for", "rules out"]
    # if any(b in reason.lower() for b in banned):
    #     return ""

    # 限长
    if len(reason) > 1024:
        reason = reason[:1024].rstrip() + "..."

    return reason


def parse_qid(qid: str) -> Dict[str, Any]:
    """
    Parse:
        P000001-V12-T0002-T3-N-0

    Return:
    {
        patient_id: "P000001",
        visit_id: "P000001-V12",
        visit_ref: "V12",
        turn_ref: "T0002",
        turn_index: 2,
        task: "T3",
        subtype: "N",
        q_index: 0,
    }
    """
    if not isinstance(qid, str) or "-" not in qid:
        raise ValueError(f"Invalid qid format: {qid}")

    parts = qid.split("-")

    if len(parts) < 6:
        raise ValueError(f"Invalid qid parts (expect 6+): {qid}")

    patient_id = parts[0]
    visit_ref = parts[1]
    turn_ref = parts[2]
    task = parts[3]
    subtype = parts[4]
    q_index = parts[5]

    # ---------- derive ----------
    visit_id = f"{patient_id}-{visit_ref}"

    # T0002 → 2
    try:
        turn_index = int(turn_ref[1:])
    except Exception:
        raise ValueError(f"Invalid turn_ref in qid: {qid}")

    try:
        q_index = int(q_index)
    except Exception:
        raise ValueError(f"Invalid question index in qid: {qid}")

    return {
        "patient_id": patient_id,
        "visit_id": visit_id,
        "visit_ref": visit_ref,
        "turn_ref": turn_ref,
        "turn_index": turn_index,
        "task": task,
        "subtype": subtype,
        "q_index": q_index,
    }


# ============================================================
# Memory builders
# ============================================================


def _note_dict_to_text(note: Any) -> str:
    """
    admission_note / discharge_note 在你的 patient json 里是 dict。
    这里用一种朴素但稳定的“自然语言组织”：按 key 输出。
    """
    if note is None:
        return ""
    if isinstance(note, str):
        return note.strip()
    if isinstance(note, dict):
        parts = []
        for k, v in note.items():
            if v is None:
                continue
            vv = v.strip() if isinstance(v, str) else str(v)
            if not vv:
                continue
            parts.append(f"{k}: {vv}")
        return "\n".join(parts).strip()
    return str(note).strip()


def build_note_memory_from_patient_json(
    patient_json: Dict[str, Any],
    *,
    current_visit_id: str,
) -> List[Dict[str, Any]]:
    """
    记忆类型1：笔记记忆
    - 收集 current_visit 之前所有 visit 的 admission_note + discharge_note
    - 每条记忆一条 item（visit 粒度 / note 粒度）
    """
    visits = patient_json.get("visits") or []
    vid2idx = {v.get("visit_id"): i for i, v in enumerate(visits)}
    if current_visit_id not in vid2idx:
        raise ValueError(f"[note_memory] current_visit_id not found in patient_json: {current_visit_id}")
    cur_idx = vid2idx[current_visit_id]

    mem: List[Dict[str, Any]] = []
    for i in range(cur_idx):
        v = visits[i]
        vid = v.get("visit_id")

        adm = (((v.get("admission_info") or {}).get("admission_note")))
        dis = (((v.get("discharge_info") or {}).get("discharge_note")))

        adm_text = _note_dict_to_text(adm)
        dis_text = _note_dict_to_text(dis)

        if adm_text:
            mem.append({
                "memory_type": "note",
                "visit_id": vid,
                "note_type": "admission_note",
                "text": adm_text,
            })
        if dis_text:
            mem.append({
                "memory_type": "note",
                "visit_id": vid,
                "note_type": "discharge_note",
                "text": dis_text,
            })
    return mem


def build_event_memory_from_sequence_json(
    sequenced_events: List[Dict[str, Any]],
    *,
    visit_order: Dict[str, int],
    current_visit_ref: str,  # 形如 "V12"
) -> List[Dict[str, Any]]:
    """
    记忆类型2：事件记忆
    - 收集 current_visit 之前所有 visit 的“每一条事件”作为单独记忆
    - 事件来源：扁平化 sequenced json（包含 ADMISSION/DISCHARGE/各类事件）
    """
    if current_visit_ref not in visit_order:
        raise ValueError(f"[event_memory] current_visit_ref not in visit_order: {current_visit_ref}")
    cur_idx = visit_order[current_visit_ref]

    mem: List[Dict[str, Any]] = []
    for ev in sequenced_events:
        vref = ev.get("visit_ref")
        if vref is None:
            continue
        if vref not in visit_order:
            continue
        if visit_order[vref] >= cur_idx:
            continue
        
        mem.append({
            "memory_type": "event",
            "visit_ref": vref,
            "event_id": ev.get("event_id"),
            "event_type": ev.get("event_type"),
            "timestamp": ev.get("timestamp"),
            "content": ev.get("content"),
        })
    return mem

# ============================================================
# IO
# ============================================================

def load_patient_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "visits" not in obj:
        raise ValueError(f"Expected new-schema patient dict with 'visits', got: {type(obj)}")
    return obj


def find_visit(patient: Dict[str, Any], visit_id: str) -> Dict[str, Any]:
    for v in patient.get("visits", []) or []:
        if str(v.get("visit_id")) == str(visit_id):
            return v
    raise KeyError(f"visit_id not found: {visit_id}")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception as e:
                logger.warning(f"Skip bad JSONL line={ln}: {e}")

def patient_event_stats(patient_json_path: Path) -> Dict[str, Any]:
    """
    Return stats for sorting:
      - file_size_bytes
      - num_visits
      - total_events (sum of len(event_stream) across visits)
      - max_visit_events
    """
    st = {
        "file_size_bytes": patient_json_path.stat().st_size if patient_json_path.exists() else 0,
        "num_visits": 0,
        "total_events": 0,
        "max_visit_events": 0,
        "exists": patient_json_path.exists(),
    }
    if not patient_json_path.exists():
        return st

    try:
        patient = load_patient_json(patient_json_path)
        visits = patient.get("visits", []) or []
        st["num_visits"] = len(visits)
        total = 0
        mx = 0
        for v in visits:
            n = len(v.get("event_stream", []) or [])
            total += n
            if n > mx:
                mx = n
        st["total_events"] = total
        st["max_visit_events"] = mx
    except Exception as e:
        logger.warning(f"Failed to read {patient_json_path}: {e}")

    return st


# ============================================================
# Memory item to docs
# ============================================================
def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()

def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    t = _safe_str(text)
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    out = []
    i = 0
    while i < len(t):
        j = min(len(t), i + max_chars)
        out.append(t[i:j])
        if j == len(t):
            break
        i = max(0, j - overlap)
    return out

def memory_item_to_docs(patient_id: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    把 note/event memory item 变成若干条 doc（text+meta）
    NOTE: 这里不做可视范围约束；可视范围过滤放到检索阶段。
    """
    mtype = item.get("memory_type")
    docs: List[Dict[str, Any]] = []

    if mtype == "note":
        text = _safe_str(item.get("text"))
        if not text:
            return []
        meta = {
            "patient_id": patient_id,
            "memory_type": "note",
            "visit_id": item.get("visit_id"),
            "visit_ref": item.get("visit_ref"),   # ✅ 新增：可选
            "visit_idx": item.get("visit_idx"),   # ✅ 新增：可选
            "note_type": item.get("note_type"),
        }
        docs.append({"text": text, "meta": meta})
        return docs

    if mtype != "event":
        return []

    et = _safe_str(item.get("event_type"))
    ts = item.get("timestamp")
    vref = _safe_str(item.get("visit_ref"))
    eid = _safe_str(item.get("event_id"))
    content = item.get("content") or {}

    base_meta = {
        "patient_id": patient_id,
        "memory_type": "event",
        "visit_ref": vref,
        "visit_idx": item.get("visit_idx"),  # ✅ 新增：可选（我们会补上）
        "event_id": eid,
        "event_type": et,
        "timestamp": ts,
    }

    if et == "LAB":
        for idx, x in enumerate(content.get("items") or []):
            text = (
                f"[LAB] t={ts} visit={vref} | "
                f"name={_safe_str(x.get('name'))} | "
                f"value={_safe_str(x.get('value_text') or x.get('value_num'))} {_safe_str(x.get('unit'))} | "
                f"flag={_safe_str(x.get('flag'))} | fluid={_safe_str(x.get('fluid'))} | "
                f"category={_safe_str(x.get('category'))}"
            )
            meta = dict(base_meta)
            meta["item_index"] = idx
            docs.append({"text": text, "meta": meta})
        return docs

    if et == "MEDICATION":
        for idx, x in enumerate(content.get("items") or []):
            text = (
                f"[MED] t={ts} visit={vref} | "
                f"drug={_safe_str(x.get('drug'))} | route={_safe_str(x.get('route'))} | "
                f"dose={_safe_str(x.get('dose'))} | status={_safe_str(x.get('status'))} | "
                f"end={_safe_str(x.get('end_timestamp'))}"
            )
            meta = dict(base_meta)
            meta["item_index"] = idx
            docs.append({"text": text, "meta": meta})
        return docs

    if et == "MICROBIOLOGY":
        specimen = content.get("specimen") or {}
        results = content.get("results") or {}
        comments = results.get("comments") or []
        text = (
            f"[MICRO] t={ts} visit={vref} | "
            f"spec_type={_safe_str(specimen.get('spec_type'))} | test={_safe_str(specimen.get('test_name'))} | "
            f"negative={_safe_str(results.get('negative'))} | "
            f"organisms={_safe_str(results.get('organisms'))} | "
            f"comments={' '.join(map(_safe_str, comments))}"
        )
        docs.append({"text": text, "meta": base_meta})
        return docs

    if et == "IMAGING":
        report = _safe_str(content.get("report"))
        chunks = _chunk_text(report, max_chars=1200, overlap=150)
        for k, ch in enumerate(chunks):
            meta = dict(base_meta)
            meta["chunk_id"] = k
            docs.append({"text": f"[IMAGING] t={ts} visit={vref} | chunk={k+1}/{len(chunks)}: {ch}", "meta": meta})
        return docs

    if et == "PROCEDURE":
        items = content.get("items") or []
        text = (
            f"[PROC] t={ts} visit={vref} | "
            f"items={'; '.join(_safe_str(x.get('name')) for x in items)}"
        )
        docs.append({"text": text, "meta": base_meta})
        return docs

    brief = _safe_str(json.dumps(content, ensure_ascii=False))
    if brief:
        docs.append({"text": f"[{et}] t={ts} visit={vref} | {brief[:2000]}", "meta": base_meta})
    return docs
