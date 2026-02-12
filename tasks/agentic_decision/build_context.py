#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import re
from pathlib import Path

from tasks.agentic_decision.tools import *
from util.logUtil import setup_logger
from config import ContextConfig

config = ContextConfig()
logger = setup_logger()

_ALLOWED_ACTIONS = [
    "ask_question",
    "order_labs",
    "order_microbiology",
    "order_imaging",
    "perform_procedure",
    "medication",
    "discharge",
]


# ============================================================
# Helpers
# ============================================================

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _clean_blank(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    t = str(x).strip()
    if not t or re.fullmatch(r"_+", t):
        return None
    return t


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _norm_type(t: Any) -> str:
    """
    New schema has mixed case types: lab / microbiology / IMAGING / MEDICATION / PROCEDURE ...
    Normalize to lower for branching.
    """
    return str(t or "").strip().lower()


def _summarize_admission(adm_note: Dict[str, Any], patient_info: Dict[str, Any]) -> str:
    gender = patient_info.get("gender", "unknown")
    age_first_visit = patient_info.get("age_first_visit", "unknown")
    language = patient_info.get("language", "unknown")
    marital_status = patient_info.get("marital_status", "unknown")
    race = patient_info.get("race", "unknown")

    parts = [
        f"Gender: {gender}.",
        f"Age at first visit: {age_first_visit}.",
        f"Language: {language}.",
        f"Marital status: {marital_status}.",
        f"Race: {race}.",
    ]

    allergies = _clean_blank(adm_note.get("allergies"))
    cc = _clean_blank(adm_note.get("chief_complaint"))
    hpi = _clean_blank(adm_note.get("history_of_present_illness"))
    fam = _clean_blank(adm_note.get("family_history"))
    attending = _clean_blank(adm_note.get("attending"))

    if cc:
        parts.append(f"Chief complaint: {cc}.")
    if hpi:
        parts.append(f"HPI: {hpi}")
    if allergies:
        parts.append(f"Allergies: {allergies}.")
    if fam:
        parts.append(f"Family history: {fam}.")
    if attending:
        parts.append(f"Attending: {attending}.")

    return " ".join(parts).strip() or "New admission."



def _build_lab_observation(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for it in items or []:
        results.append(
            {
                "name": it.get("name"),
                "category": it.get("category"),
                "fluid": it.get("fluid"),
                "value_num": it.get("value_num"),
                "value_text": it.get("value_text"),
                "unit": it.get("unit"),
                "flag": it.get("flag"),
            }
        )
    return {"observation_type": "lab_results", "results": results}


def _build_micro_observation_from_schema(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    New schema microbiology:
      specimen: {specimen_id, spec_type, test_name, test_seq}
      results: {negative: bool, organisms: [...], comments: list[str] or null}

    We convert to a stable observation.
    """
    specimen = ev.get("specimen") or {}
    results = ev.get("results") or {}

    return {
        "observation_type": "microbiology_results",
        "specimen": {
            "specimen_id": specimen.get("specimen_id"),
            "spec_type": specimen.get("spec_type"),
        },
        "test": specimen.get("test_name"),
        "results": {
            "negative": results.get("negative"),
            "organisms": results.get("organisms") or [],
            "comments": results.get("comments"),
        },
    }


def _build_med_observation(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    admin_results = []
    for it in items or []:
        admin_results.append({"drug": it.get("drug"), "status": it.get("status")})
    return {
        "observation_type": "medication",
        "notes": "Medication orders/admin records if available.",
        "administration_results": admin_results,
    }


def _llm_reason(llm: LLMUtil, messages_so_far: List[Dict[str, Any]], current_action: str, current_args: Dict[str, Any]) -> str:
    for _ in range(2):
        try:
            r = generate_reason_from_messages_llm(
                llm,
                messages_so_far=messages_so_far,
                current_action=current_action,
                current_args=current_args,
                model=config.REASON_MODEL,
            )
            if r and isinstance(r, str) and r.strip():
                return r.strip()
        except Exception:
            continue
    return "not_available"


# ============================================================
# Core: one visit prefix -> messages
# ============================================================

def render_visit_prefix_to_messages(
    llm: LLMUtil,
    patient_info: Dict[str, Any],
    visit: Dict[str, Any],
    stop_before_event_id: Optional[str] = None,
    stop_before_timestamp: Optional[str] = None,
    max_events: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert ONE visit from start up to BEFORE a cutoff event into OpenAI-like multi-turn messages.

    Cutoff priority:
      1) stop_before_event_id (exclusive)
      2) stop_before_timestamp (exclusive; compare parsed datetime)
      3) max_events (take first N events after sorting)
    """

    admission_info = visit.get("admission_info", {}) or {}
    adm_note = admission_info.get("admission_note") or {}
    if not isinstance(adm_note, dict):
        adm_note = {}

    messages: List[Dict[str, Any]] = []
    messages.append(
        {
            "role": "system",
            "content": f'''
You are a doctor agent. You can perform these semantic actions when you are interacting with the environment: {", ".join(_ALLOWED_ACTIONS)}. The environment returns only information that exists in this admission record; otherwise it returns not_available. Do not invent new findings.
Meanwhile, you may also asked to answer agentic-desision questions without interacting with the environment; in that case, you should answer based on the admission record and NOT perform any environment actions.
Use the following format for your messages:
- If you are interacting with the environment, your message MUST be a JSON object with keys: {{reason, action, args}}. 
    Reason may explain workflow/clinical rationale but must not add new patient facts beyond the record.
- If you are directly answering a question without environment interaction, your message MUST be a JSON object with keys: {{reason, answer}}.
    The answer should be a list containing only provided option from the question, and reason should be a brief explanation based on the context and admission history.
'''
        }
    )
    messages.append({"role": "user", "content": _summarize_admission(adm_note, patient_info)})

    # sort events by timestamp
    event_stream = visit.get("event_stream", []) or []
    event_stream_sorted = sorted(
        event_stream,
        key=lambda e: _parse_dt(e.get("timestamp")) or datetime.min
    )

    cutoff_dt = _parse_dt(stop_before_timestamp) if stop_before_timestamp else None

    kept: List[Dict[str, Any]] = []
    for ev in event_stream_sorted:
        ev_id = ev.get("event_id")
        ev_dt = _parse_dt(ev.get("timestamp"))

        if stop_before_event_id and ev_id == stop_before_event_id:
            break
        if cutoff_dt and ev_dt and ev_dt >= cutoff_dt:
            break

        kept.append(ev)
        if max_events is not None and len(kept) >= max_events:
            break

    # render
    for ev in kept:
        ev_type = _norm_type(ev.get("type"))
        ev_ts = ev.get("timestamp")
        ev_id = ev.get("event_id")

        # lab
        if ev_type == "lab":
            panels, _ = infer_lab_panel_for_event(llm, ev, None)
            current_args = {"panels": panels, "timestamp": ev_ts}
            reason = _llm_reason(llm, messages, "order_labs", current_args)
            
            items = ev.get("items", []) or []
            messages.append({"role": "assistant", "content": {"reason": reason, "action": "order_labs", "args": current_args}})
            messages.append({"role": "user", "content": {"name": "environment", **_build_lab_observation(items), "timestamp": ev_ts, "event_id": ev_id}})
            continue

        # microbiology (new schema: specimen + results)
        if ev_type == "microbiology":
            specimen = ev.get("specimen") or {}
            test_name = specimen.get("test_name") or specimen.get("name")
            current_args = {"tests": [test_name] if test_name else [], "timestamp": ev_ts}
            reason = _llm_reason(llm, messages, "order_microbiology", current_args)

            messages.append({"role": "assistant", "content": {"reason": reason, "action": "order_microbiology", "args": current_args}})
            messages.append({"role": "user", "content": {"name": "environment", **_build_micro_observation_from_schema(ev), "timestamp": ev_ts, "event_id": ev_id}})
            continue

        # imaging (new schema uses "report")
        if ev_type == "imaging":
            report = _safe_str(ev.get("report"))
            result = infer_imaging_param_for_event(llm,ev)
            modality = result.get("modality") or "imaging"
            target = result.get("target") or ""
            
            current_args = {"modality": modality, "target": target, "timestamp": ev_ts}
            reason = _llm_reason(llm, messages, "order_imaging", current_args)

            messages.append({"role": "assistant", "content": {"reason": reason, "action": "order_imaging", "args": current_args}})
            messages.append(
                {
                    "role": "user",
                    "content": {
                        "name": "environment",
                        "observation_type": "imaging_results",
                        "results": [
                            {
                                "test": f"{target} {modality}".strip(),
                                "content": report,
                                "raw_available": True,
                                "timestamp": ev_ts,
                                "event_id": ev_id,
                            }
                        ],
                    },
                }
            )
            continue

        # medication (new schema: items list)
        if ev_type == "medication":
            items = ev.get("items", []) or []
            rx_items: List[Dict[str, Any]] = []
            for it in items:
                rx_items.append(
                    {
                        "timestamp": it.get("timestamp") or ev_ts,
                        "drug": it.get("drug"),
                        "dose": it.get("dose"),
                        "unit": it.get("unit"),
                        "route": it.get("route"),
                        "status": it.get("status"),
                        "end_timestamp": it.get("end_timestamp"),
                        "drug_type": it.get("drug_type"),
                    }
                )

            current_args = {"timestamp": ev_ts, "prescriptions": rx_items}
            reason = _llm_reason(llm, messages, "medication", current_args)

            messages.append({"role": "assistant", "content": {"reason": reason, "action": "medication", "args": current_args}})
            messages.append({"role": "user", "content": {"name": "environment", **_build_med_observation(items), "timestamp": ev_ts, "event_id": ev_id}})
            continue

        # procedure (new schema: name)
        if ev_type == "procedure":
            proc_name = _clean_blank(ev.get("name")) or "procedure"
            current_args = {"name": proc_name, "timestamp": ev_ts}
            reason = _llm_reason(llm, messages, "perform_procedure", current_args)

            messages.append({"role": "assistant", "content": {"reason": reason, "action": "perform_procedure", "args": current_args}})
            messages.append(
                {
                    "role": "user",
                    "content": {
                        "name": "environment",
                        "observation_type": "procedure_result",
                        "procedure": {"name": proc_name, "brief": "procedure_performed"},
                        "timestamp": ev_ts,
                        "event_id": ev_id,
                    },
                }
            )
            continue

        # other types: skip (keep pipeline robust)
        continue

    return messages


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Input patient JSON (new schema)")
    ap.add_argument("--visit_id", required=True, help="Which visit_id to render (e.g., P000355-V1)")
    ap.add_argument("--stop_before_event_id", default=None, help="Exclusive cutoff event_id (e.g., P000355-V1-E10)")
    ap.add_argument("--stop_before_timestamp", default=None, help="Exclusive cutoff timestamp (YYYY-mm-dd HH:MM:SS)")
    ap.add_argument("--max_events", type=int, default=None, help="Take first N events after time-sort (optional)")
    ap.add_argument("--out", required=True, help="Output json path")
    args = ap.parse_args()

    llm = LLMUtil()
    patient = load_patient_json(Path(args.inp))
    patient_info = patient.get("patient_info", {}) or {}
    visit = find_visit(patient, args.visit_id)

    messages = render_visit_prefix_to_messages(
        llm,
        patient_info=patient_info,
        visit=visit,
        stop_before_event_id=args.stop_before_event_id,
        stop_before_timestamp=args.stop_before_timestamp,
        max_events=args.max_events,
    )

    out_obj = {
        "patient_id": patient_info.get("patient_id"),
        "visit_id": args.visit_id,
        "cutoff": {
            "stop_before_event_id": args.stop_before_event_id,
            "stop_before_timestamp": args.stop_before_timestamp,
            "max_events": args.max_events,
        },
        "messages": messages,
        "token_usage": llm.get_token_usage(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote: {out_path}")
    logger.info(f"Token usage: {llm.get_token_usage()}")


if __name__ == "__main__":
    main()
