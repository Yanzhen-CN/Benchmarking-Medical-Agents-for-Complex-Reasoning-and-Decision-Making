import argparse
import json
import math
import re
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")

def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
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
    max_history_msgs: int = 24,   # 控制上下文长度：只取最近 N 条
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