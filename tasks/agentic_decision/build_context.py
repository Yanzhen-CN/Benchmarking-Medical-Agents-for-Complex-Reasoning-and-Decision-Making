#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm
from tasks.agentic_decision.tools import *
from util.logUtil import setup_logger
from config import ContextConfig, AgentTaskConfig

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
# Helpers (keep your original helpers)
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
# Core: one visit (FULL) -> messages
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

    For task-1 (visit-level context), call with:
      stop_before_event_id=None, stop_before_timestamp=None, max_events=None
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
    logger.info(f"Rendering visit_id={visit.get('visit_id')} with {len(kept)} events (cutoff_event_id={stop_before_event_id}, cutoff_timestamp={stop_before_timestamp}, max_events={max_events})")

    for ev in tqdm(kept, desc=f"Rendering {len(kept)} events to messages"):
        ev_type = _norm_type(ev.get("type"))
        ev_ts = ev.get("timestamp")
        ev_id = ev.get("event_id")

        if ev_type == "lab":
            panels, _ = infer_lab_panel_for_event(llm, ev, None)
            current_args = {"panels": panels, "timestamp": ev_ts}
            reason = _llm_reason(llm, messages, "order_labs", current_args)

            items = ev.get("items", []) or []
            messages.append({"role": "assistant", "content": {"reason": reason, "action": "order_labs", "args": current_args}})
            messages.append({"role": "user", "content": {"name": "environment", **_build_lab_observation(items), "timestamp": ev_ts, "event_id": ev_id}})
            continue

        if ev_type == "microbiology":
            specimen = ev.get("specimen") or {}
            test_name = specimen.get("test_name") or specimen.get("name")
            current_args = {"tests": [test_name] if test_name else [], "timestamp": ev_ts}
            reason = _llm_reason(llm, messages, "order_microbiology", current_args)

            messages.append({"role": "assistant", "content": {"reason": reason, "action": "order_microbiology", "args": current_args}})
            messages.append({"role": "user", "content": {"name": "environment", **_build_micro_observation_from_schema(ev), "timestamp": ev_ts, "event_id": ev_id}})
            continue

        if ev_type == "imaging":
            report = _safe_str(ev.get("report"))
            result = infer_imaging_param_for_event(llm, ev)
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


def collect_visit_ids_from_questions(questions_jsonl: Path) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    Return:
      - unique visit_id set
      - visit_id -> list[qid] (for optional debug / index)
    Expected question schema contains "visit_id" and "qid" (like your sample).
    """
    visit_ids: Set[str] = set()
    visit2qids: Dict[str, List[str]] = {}

    for q in iter_jsonl(questions_jsonl):
        vid = q.get("visit_id")
        if not vid:
            continue
        vid = str(vid)
        visit_ids.add(vid)

        qid = q.get("qid")
        if qid is not None:
            visit2qids.setdefault(vid, []).append(str(qid))

    return visit_ids, visit2qids


def build_all_visit_contexts_from_questions(
    patient_json: Path,
    questions_jsonl: Path,
    out_dir: Path,
    max_events_per_visit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    TASK-1:
    - scan questions_jsonl -> all visit_ids
    - render FULL visit context (no truncation) for each visit_id
    - write one context json per visit into out_dir/{visit_id}.context.json
    - also write an index file out_dir/index.json
    """
    patient = load_patient_json(patient_json)
    patient_info = patient.get("patient_info", {}) or {}
    patient_id = patient_info.get("patient_id") or "unknown_patient"

    visit_ids, visit2qids = collect_visit_ids_from_questions(questions_jsonl)
    if not visit_ids:
        raise ValueError(f"No visit_id found in questions: {questions_jsonl}")

    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMUtil()  # reuse one llm util for token accounting
    written = []
    missing = []

    for vid in sorted(visit_ids):
        try:
            visit = find_visit(patient, vid)
        except KeyError:
            missing.append(vid)
            logger.warning(f"visit_id {vid} not found in patient {patient_id}, skipping.")
            continue

        messages = render_visit_prefix_to_messages(
            llm=llm,
            patient_info=patient_info,
            visit=visit,
            stop_before_event_id=None,
            stop_before_timestamp=None,
            max_events=max_events_per_visit,
        )

        out_obj = {
            "patient_id": patient_id,
            "visit_id": vid,
            "messages": messages,
            "token_usage": llm.get_token_usage(),
            "source": {
                "patient_json": str(patient_json),
                "questions_jsonl": str(questions_jsonl),
                "qids_in_visit": visit2qids.get(vid, []),
            },
            "build_cfg": {
                "max_events_per_visit": max_events_per_visit,
            },
        }

        out_path = out_dir / f"{vid}.context.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        written.append(str(out_path))
        logger.info(f"Wrote visit context: {out_path}")

    index = {
        "patient_id": patient_id,
        "num_visits_in_questions": len(visit_ids),
        "num_written": len(written),
        "num_missing_in_patient_json": len(missing),
        "missing_visit_ids": missing,
        "outputs": written,
        "token_usage": llm.get_token_usage(),
    }

    index_path = out_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote index: {index_path}")
    logger.info(f"Token usage: {llm.get_token_usage()}")

    return index

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
from datetime import time
import os
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    cfg = AgentTaskConfig()

    files = list(iter_jsonl_files(cfg.QUESTIONS_DIR))
    pids = sorted(set([f.name.split(".")[0] for f in files]))

    # ---- pre-scan stats and sort (desc) ----
    pid_stats = []
    for pid in tqdm(pids, desc="Scanning patient sizes"):
        p_path = Path(cfg.PATIENTS_DIR) / f"{pid}.json"
        st = patient_event_stats(p_path)
        pid_stats.append((pid, st))

    # ALWAYS sort (desc) by: total_events, then file_size
    pid_stats.sort(
        key=lambda x: (x[1].get("total_events", 0), x[1].get("file_size_bytes", 0)),
        reverse=True,
    )

    if cfg.DEMO_MODE:
        top_k = cfg.DEMO_N
        top = pid_stats[:top_k]
        logger.info("Top patients by event_stream length (total_events desc, file_size desc):")
        for rank, (pid, st) in enumerate(top, 1):
            logger.info(
                f"[{rank}] {pid} | total_events={st['total_events']} | "
                f"max_visit_events={st['max_visit_events']} | visits={st['num_visits']} | "
                f"file_size={st['file_size_bytes'] / (1024*1024):.2f} MB"
            )
    else:
        top = pid_stats
        logger.warning("DEMO_MODE is OFF, will process ALL patients sorted by size. This may take a long time!")

    # ---- resume ----
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    final_report, final_report_name = {}, "final_report_agent_context_run.json"

    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    already_run = set()
    report_path = log_dir / final_report_name
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
            if isinstance(previous, dict):
                final_report = previous  # load existing
                already_run = set(previous.keys())
                logger.info(f"Resume enabled: loaded {len(already_run)} finished pids from {report_path}")
        except Exception as e:
            logger.warning(f"Failed to load previous report {report_path}: {e}")

    # filter pending
    pending = [(pid, st) for pid, st in top if pid not in already_run]
    if not pending:
        logger.info("Nothing to do: all selected patients already processed.")
        logger.info(f"Existing report: {report_path}")
        raise SystemExit(0)

    logger.info(f"Will process {len(pending)} patients (skipping {len(top) - len(pending)} already-run).")

    # ---- worker ----
    def _run_one(pid: str, st: dict) -> Tuple[str, dict]:
        out_dir = Path(cfg.CONTEXT_DIR) / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        result = build_all_visit_contexts_from_questions(
            patient_json=Path(cfg.PATIENTS_DIR) / f"{pid}.json",
            questions_jsonl=Path(cfg.QUESTIONS_DIR) / f"{pid}.jsonl",
            out_dir=out_dir,
            max_events_per_visit=cfg.MAX_EVENTS_PER_VISIT,
        )
        return pid, result

    # ---- concurrency ----
    max_workers = getattr(cfg, "MAX_WORKERS", None)
    if not max_workers:
        max_workers = min(8, (os.cpu_count() or 8))
    logger.info(f"Running with max_workers={max_workers}")

    # 主线程收集结果 + 落盘，避免并发写文件
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_one, pid, st): (pid, st) for pid, st in pending}

        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Building contexts (concurrent)"):
            pid, st = futs[fut]
            try:
                pid_done, result = fut.result()
            except Exception as e:
                logger.exception(f"FAILED pid={pid}: {e}")
                # 失败也写入报告，方便你后续定位/重跑
                final_report[pid] = {"error": str(e), "stats": st}
            else:
                final_report[pid_done] = result
                logger.info(f"Done pid={pid_done}: wrote contexts, token_usage={result.get('token_usage', {}).get('chat')}")

                # accumulate usage (best-effort)
                chat_usage = result.get("token_usage", {}).get("chat", {}) if isinstance(result, dict) else {}
                for k in usage:
                    usage[k] += int(chat_usage.get(k, 0) or 0)

            # incremental save after each completion
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2)

    logger.info("All done (concurrent).")
    logger.info(f"Total token usage across processed patients: \n{json.dumps(usage, ensure_ascii=False, indent=2)}")
    logger.info(f"Final report: {report_path}")
