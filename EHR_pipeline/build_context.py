
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import re
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm

from EHR_pipeline.llm_tools import infer_imaging_modality_target_llm, generate_reason_from_messages_llm
from config import ContextConfig

config = ContextConfig()

# ============================================================
# Rendering: one visit -> OpenAI-like multi-turn messages
# ============================================================

_ALLOWED_ACTIONS = [
    "ask_question",
    "order_labs",
    "order_microbiology",
    "order_imaging",
    "perform_procedure",
    "medication",
    "discharge",
]


def render_session_to_messages(session_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert ONE visit/session (session_json["visits"][0]) into OpenAI-like chat messages.

    Pattern:
      - system: rules
      - user: admission summary (grounded in admission_note)
      - [zero or more] tool: vital_signs (standalone environment events)
      - assistant: {"reason":..., "action":..., "args":...}  (ONLY 7 semantic actions)
      - tool (environment.step): observation (grounded in event_stream or notes)
      - ... repeated
      - assistant discharge + (optional) tool discharge_summary (ground truth)

    IMPORTANT:
      - All assistant 'reason' fields MUST be generated via LLM (generate_reason_from_messages_llm).
      - If the LLM returns empty/None or errors, we set reason="not_available" to keep the pipeline robust.
    """

    # --------------------------
    # Helpers
    # --------------------------
    def _parse_dt(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    def _safe_str(x: Any) -> str:
        return "" if x is None else str(x)

    def _clean_blank(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        t = str(x).strip()
        if not t or re.fullmatch(r"_+", t):
            return None
        return t

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

        # Grounded summary ONLY
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

        return " ".join(parts).strip()

    def _infer_imaging_modality_target(content: str) -> Tuple[str, str]:
        if getattr(config, "USE_LLM_FOR_IMAGE_DESC", False):
            modality, target, confidence = infer_imaging_modality_target_llm(
                content, config.IMAGE_DESC_MODEL
            )
            if confidence >= config.IMAGE_DESC_THRESHOLD:
                return modality, target

        # Lightweight heuristics; never invent findings
        u = content.upper()
        if "CHEST" in u and ("PA" in u or "LATERAL" in u or "CXR" in u):
            return "XRay", "Chest"
        if "ULTRASOUND" in u or re.search(r"\bUS\b", u):
            return "Ultrasound", "Abdomen"
        if "CT" in u:
            return "CT", "UnknownTarget"
        if "MRI" in u:
            return "MRI", "UnknownTarget"
        if "ECHO" in u:
            return "Echocardiogram", "Heart"
        return "Imaging", "UnknownTarget"

    def _chunk_panels_from_lab_items(items: List[Dict[str, Any]]) -> List[str]:
        # Standard terms assumption: name is panel
        panels: List[str] = []
        for it in items:
            nm = _clean_blank(it.get("name"))
            if nm and nm not in panels:
                panels.append(nm)
        return panels or ["Labs"]

    def _build_lab_observation(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        for it in items:
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

    def _build_vital_observation(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        vitals = []
        for it in items:
            vitals.append(
                {
                    "name": it.get("name"),
                    "value_num": it.get("value_num"),
                    "value_text": it.get("value_text"),
                    "unit": it.get("unit"),
                    "flag": it.get("flag"),
                }
            )
        return {"observation_type": "vital_signs", "vitals": vitals}

    def _build_micro_observation(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        for it in items:
            results.append(
                {
                    "test": it.get("test"),
                    "specimen": it.get("specimen"),
                    "organism": it.get("organism"),
                    "result": it.get("result"),
                    "abx_susceptibility": it.get("abx_susceptibility"),
                    "flag": it.get("flag"),
                }
            )
        return {"observation_type": "microbiology_results", "results": results}

    def _llm_reason(messages_so_far: List[Dict[str, Any]], current_action: str, current_args: Dict[str, Any]) -> str:
        # All reasons must come from LLM; we retry to reduce transient failures.
        for _ in range(2):
            try:
                r = generate_reason_from_messages_llm(
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

    # --------------------------
    # Locate the visit
    # --------------------------
    patient_info = session_json.get("patient_info", {}) or {}
    visits = session_json.get("visits", []) or []
    if not visits:
        return [], {}

    if len(visits) != 1:
        # this function is "one visit -> one session"; caller should slice.
        visits = [visits[0]]

    visit = visits[0]
    admission_info = visit.get("admission_info", {}) or {}
    discharge_info = visit.get("discharge_info", {}) or {}

    adm_note = admission_info.get("admission_note") or {}
    if not isinstance(adm_note, dict):
        adm_note = {}
    dis_note = discharge_info.get("discharge_note") or {}
    if not isinstance(dis_note, dict):
        dis_note = {}

    # --------------------------
    # Start messages
    # --------------------------
    messages: List[Dict[str, Any]] = []

    messages.append(
        {
            "role": "system",
            "content": (
                "You are a doctor agent. You must ONLY choose from these semantic actions: "
                + ", ".join(_ALLOWED_ACTIONS)
                + ". The environment returns only information that exists in this admission record; otherwise it returns not_available. "
                  "Do not invent new findings."
            ),
        }
    )
    messages.append(
        {
            "role": "system",
            "content": (
                "Each assistant message MUST be a JSON object with keys: {reason, action, args}. "
                "Reason may explain workflow/clinical rationale but must not add new patient facts beyond the record."
            ),
        }
    )

    user_adm_summary = _summarize_admission(adm_note, patient_info) or "New admission."
    messages.append({"role": "user", "content": user_adm_summary})

    # --------------------------
    # Event-driven turns
    # --------------------------
    event_stream = visit.get("event_stream", []) or []
    event_stream_sorted = sorted(
        event_stream, key=lambda e: _parse_dt(e.get("timestamp")) or datetime.min
    )

    for ev in event_stream_sorted:
        ev_type = ev.get("type")
        ev_ts = ev.get("timestamp")

        # 0) vitals: standalone environment events
        if ev_type == "vital":
            items = ev.get("items", []) or []
            messages.append(
                {
                    "role": "tool",
                    "name": "environment.step",
                    "content": {
                        **_build_vital_observation(items),
                        "timestamp": ev_ts,
                        "event_id": ev.get("event_id"),
                    },
                }
            )
            continue

        # 1) labs
        if ev_type == "lab":
            items = ev.get("items", []) or []
            panels = _chunk_panels_from_lab_items(items)
            current_args = {"panels": panels, "timestamp": ev_ts}

            reason = _llm_reason(messages, "order_labs", current_args)

            messages.append(
                {
                    "role": "assistant",
                    "content": {"reason": reason, "action": "order_labs", "args": current_args},
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": "environment.step",
                    "content": {
                        **_build_lab_observation(items),
                        "timestamp": ev_ts,
                        "event_id": ev.get("event_id"),
                    },
                }
            )
            continue

        # 2) microbiology
        if ev_type == "microbiology":
            items = ev.get("items", []) or []
            tests = [it.get("test") for it in items if it.get("test")]
            current_args = {"tests": tests, "timestamp": ev_ts}

            reason = _llm_reason(messages, "order_microbiology", current_args)

            messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "reason": reason,
                        "action": "order_microbiology",
                        "args": current_args,
                    },
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": "environment.step",
                    "content": {
                        **_build_micro_observation(items),
                        "timestamp": ev_ts,
                        "event_id": ev.get("event_id"),
                    },
                }
            )
            continue

        # 3) imaging
        if ev_type == "imaging":
            content = _safe_str(ev.get("content"))
            modality, target = _infer_imaging_modality_target(content)
            current_args = {"modality": modality, "target": target, "timestamp": ev_ts}

            reason = _llm_reason(messages, "order_imaging", current_args)

            messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "reason": reason,
                        "action": "order_imaging",
                        "args": current_args,
                    },
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": "environment.step",
                    "content": {
                        "observation_type": "imaging_results",
                        "results": [
                            {
                                "test": f"{target} {modality}".strip(),
                                "content": content,
                                "raw_available": True,
                                "timestamp": ev_ts,
                                "event_id": ev.get("event_id"),
                            }
                        ],
                    },
                }
            )
            continue

        # 4) medications/prescriptions
        if ev_type == "prescription":
            items = ev.get("items", []) or []

            rx_items: List[Dict[str, Any]] = []
            admin_results: List[Dict[str, Any]] = []
            for it in items:
                rx_items.append(
                    {
                        "timestamp": it.get("timestamp") or ev_ts,
                        "drug_type": it.get("drug_type"),
                        "drug": it.get("drug"),
                        "dose": it.get("dose"),
                        "unit": it.get("unit"),
                        "route": it.get("route"),
                        "end_timestamp": it.get("end_timestamp"),
                    }
                )
                admin_results.append(
                    {
                        "drug": it.get("drug"),
                        "status": it.get("status"),
                    }
                )

            current_args = {"timestamp": ev_ts, "prescriptions": rx_items}
            reason = _llm_reason(messages, "medication", current_args)

            messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "reason": reason,
                        "action": "medication",
                        "args": current_args,
                    },
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": "environment.step",
                    "content": {
                        "observation_type": "medication",
                        "notes": "Prescription recorded; eMAR fields reflect administration outcomes if available.",
                        "timestamp": ev_ts,
                        "event_id": ev.get("event_id"),
                        "administration_results": admin_results,
                    },
                }
            )
            continue

        # 5) procedures
        if ev_type == "procedure":
            proc_name = _clean_blank(ev.get("name")) or "procedure"
            intent = ev.get("intent") or ["diagnostic"]
            current_args = {"name": proc_name, "intent": intent, "timestamp": ev_ts}

            reason = _llm_reason(messages, "perform_procedure", current_args)

            messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "reason": reason,
                        "action": "perform_procedure",
                        "args": current_args,
                    },
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": "environment.step",
                    "content": {
                        "observation_type": "procedure_result",
                        "procedure": {"name": proc_name, "brief": "procedure_performed"},
                        "timestamp": ev_ts,
                        "event_id": ev.get("event_id"),
                    },
                }
            )
            continue

        # Unknown event type -> skip
        continue

    # --------------------------
    # Discharge (ground truth)
    # --------------------------
    final_dx = _clean_blank(dis_note.get("discharge_diagnosis"))
    if not final_dx:
        dx_list = visit.get("diagnosis", []) or []
        if dx_list and isinstance(dx_list[0], dict):
            final_dx = dx_list[0].get("description")
        final_dx = final_dx or "Discharge diagnosis not_available"
        
    dx_aligned: List[str] = []
    for d in (visit.get("diagnosis", []) or []):
        if isinstance(d, dict):
            desc = _clean_blank(d.get("description"))
            if desc:
                dx_aligned.append(desc)
    discharge_ts = discharge_info.get("discharge_time") or "unknown_time"
    
    discharge_args = {
        "final_diagnoses": [final_dx] if isinstance(final_dx, str) else final_dx,
        "summary_style": "discharge_note_like",
        "timestamp": discharge_ts,
        "discharge_note": {
            "discharge_location": discharge_info.get("discharge_location"),
            "discharge_diagnosis": final_dx,
            "diagnoses_icd_aligned": dx_aligned,
            "hospital_course": dis_note.get("hospital_course"),
            "discharge_instructions": dis_note.get("discharge_instructions"),
        }
    }
    discharge_reason = _llm_reason(messages, "discharge", discharge_args)

    messages.append(
        {
            "role": "assistant",
            "content": {
                "reason": discharge_reason,
                "action": "discharge",
                "args": discharge_args,
            },
        }
    )

    ground_truth = {
        "ground_truth_available": True,
        "source": "discharge_note",
        "discharge_time": discharge_ts,
        "discharge_location": discharge_info.get("discharge_location"),
        "discharge_diagnosis": final_dx,
        "diagnoses_icd_aligned": dx_aligned,
        "hospital_course": dis_note.get("hospital_course"),
        "discharge_instructions": dis_note.get("discharge_instructions"),
    }

    return messages, ground_truth


# ============================================================
# New schema adapter: P000001_sequenced.json -> legacy patient schema
# ============================================================

def _parse_dt_str(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _event_timestamp(ev: Dict[str, Any]) -> Optional[str]:
    ts = ev.get("timestamp")
    if ts:
        return ts
    c = ev.get("content") or {}
    if isinstance(c, dict) and c.get("timestamp"):
        return c.get("timestamp")
    return None


def _adapt_sequenced_events_to_patient_json(events: List[Dict[str, Any]], patient_id: str) -> Dict[str, Any]:
    """
    Adapt flat event list schema (P000001_sequenced.json) to:

    {
      "patient_info": {...},
      "visits": [
        {
          "visit_id": "V1",
          "admission_info": {...},
          "discharge_info": {...},
          "event_stream": [ {"type": "lab"/"vital"/"imaging"/"prescription"/"procedure"/"microbiology", ...}, ... ]
        },
        ...
      ]
    }
    """
    patient_info: Dict[str, Any] = {"patient_id": patient_id}

    demo = next((e for e in events if e.get("event_type") == "PATIENT_DEMOGRAPHICS"), None)
    if demo and isinstance(demo.get("content"), dict):
        patient_info.update(demo["content"])

    by_visit: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("event_type") == "PATIENT_DEMOGRAPHICS":
            continue
        vr = e.get("visit_ref") or "UNKNOWN_VISIT"
        by_visit.setdefault(vr, []).append(e)

    visits: List[Dict[str, Any]] = []

    for vr, evs in sorted(by_visit.items(), key=lambda kv: kv[0]):
        adm_ev = next((e for e in evs if e.get("event_type") == "ADMISSION"), None)
        dis_ev = next((e for e in evs if e.get("event_type") == "DISCHARGE"), None)

        admission_info: Dict[str, Any] = {}
        discharge_info: Dict[str, Any] = {}

        if adm_ev:
            c = adm_ev.get("content") or {}
            admission_info = {
                "admission_time": _event_timestamp(adm_ev),
                "location": c.get("location"),
                "admission_type": c.get("admission_type"),
                "insurance": c.get("insurance"),
                "admission_note": {
                    "chief_complaint": c.get("chief_complaint"),
                    "history_of_present_illness": c.get("history_of_present_illness"),
                    "allergies": c.get("allergies"),
                    "family_history": c.get("family_history"),
                    "attending": c.get("attending"),
                },
            }

        if dis_ev:
            c = dis_ev.get("content") or {}
            discharge_info = {
                "discharge_time": c.get("discharge_time") or _event_timestamp(dis_ev),
                "discharge_location": c.get("discharge_location"),
                "discharge_note": c.get("discharge_note") or {},
            }

        event_stream: List[Dict[str, Any]] = []
        for e in evs:
            et = e.get("event_type")
            if et in ("ADMISSION", "DISCHARGE"):
                continue
            ts = _event_timestamp(e)
            c = e.get("content") or {}

            if et == "LAB":
                event_stream.append(
                    {
                        "event_id": e.get("event_id"),
                        "type": "lab",
                        "timestamp": ts,
                        "items": [
                            {
                                "name": c.get("name"),
                                "category": c.get("category"),
                                "fluid": c.get("fluid"),
                                "value_num": c.get("value"),
                                "value_text": c.get("value_text"),
                                "unit": c.get("unit"),
                                "flag": c.get("flag"),
                            }
                        ],
                    }
                )

            elif et == "VITAL":
                # standalone env events
                event_stream.append(
                    {
                        "event_id": e.get("event_id"),
                        "type": "vital",
                        "timestamp": ts,
                        "items": [
                            {
                                "name": c.get("name"),
                                "value_num": c.get("value"),
                                "value_text": str(c.get("value")) if c.get("value") is not None else None,
                                "unit": c.get("unit"),
                                "flag": "abnormal" if c.get("warning") else c.get("flag"),
                            }
                        ],
                    }
                )

            elif et in ("MICROBIOLOGY", "CULTURE", "MICRO"):
                event_stream.append(
                    {
                        "event_id": e.get("event_id"),
                        "type": "microbiology",
                        "timestamp": ts,
                        "items": [
                            {
                                "test": c.get("test") or c.get("test_name") or c.get("name"),
                                "specimen": c.get("specimen") or c.get("spec_type"),
                                "organism": c.get("organism"),
                                "result": c.get("result") or c.get("interpretation"),
                                "abx_susceptibility": c.get("susceptibility") or c.get("abx_susceptibility"),
                                "flag": "abnormal" if c.get("warning") else c.get("flag"),
                            }
                        ],
                    }
                )

            elif et == "MEDICATION":
                items = c.get("items") or []
                rx_items = []
                for it in items:
                    rx_items.append({**it, "timestamp": it.get("timestamp") or c.get("timestamp") or ts})
                event_stream.append(
                    {
                        "event_id": e.get("event_id"),
                        "type": "prescription",
                        "timestamp": c.get("timestamp") or ts,
                        "items": rx_items,
                    }
                )

            elif et == "IMAGING":
                event_stream.append(
                    {
                        "event_id": e.get("event_id"),
                        "type": "imaging",
                        "timestamp": c.get("timestamp") or ts,
                        "content": c.get("report") or c.get("content") or "",
                    }
                )

            elif et == "PROCEDURE":
                event_stream.append(
                    {
                        "event_id": e.get("event_id"),
                        "type": "procedure",
                        "timestamp": c.get("timestamp") or ts,
                        "name": c.get("name"),
                        "code": c.get("code"),
                        "intent": ["diagnostic"],
                    }
                )

            else:
                continue

        event_stream.sort(key=lambda x: _parse_dt_str(x.get("timestamp")) or datetime.min)

        visits.append(
            {
                "visit_id": vr,
                "admission_info": admission_info,
                "discharge_info": discharge_info,
                "event_stream": event_stream,
            }
        )

    return {"patient_info": patient_info, "visits": visits}


# ============================================================
# Batch building: each patient file -> multi-session output
# ============================================================

def build_session_messages(
    patient_info: Dict[str, Any],
    visit: Dict[str, Any]
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    visit_id = visit.get("visit_id", "UNKNOWN_VISIT")
    session_key = str(visit_id)  # key per visit inside a patient file

    session_json = {"patient_info": patient_info, "visits": [visit]}
    messages, ground_truth = render_session_to_messages(session_json)
    return session_key, messages, ground_truth


def process_one_patient_file(pf_str: str, out_dir_str: str, inner_workers: int = 8) -> str:
    pf = Path(pf_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(pf, "r", encoding="utf-8") as f:
        patient_data = json.load(f)

    # Support BOTH schemas:
    # 1) legacy: {"patient_info":..., "visits":[...]}
    # 2) new sequenced: [ {"event_type":..., "visit_ref":..., "content":...}, ... ]
    if isinstance(patient_data, list):
        patient_id = pf.stem
        patient_data = _adapt_sequenced_events_to_patient_json(patient_data, patient_id)

    patient_info = patient_data.get("patient_info", {}) or {}
    visits = patient_data.get("visits", []) or []
    patient_id = patient_info.get("patient_id", pf.stem)

    sessions: Dict[str, Dict[str, Any]] = {}

    # patient 内部：并行 visit -> session
    with ThreadPoolExecutor(max_workers=inner_workers) as ex:
        futures = [ex.submit(build_session_messages, patient_info, v) for v in visits]
        pbar = tqdm(
            total=len(futures),
            desc=f"Visits({patient_id})",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )
        try:
            for fu in as_completed(futures):
                k, msgs, gt = fu.result()
                sessions[k] = {"messages": msgs, "ground_truth": gt}
                pbar.update(1)
        finally:
            pbar.close()

    out_file = out_dir / f"{patient_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "patient_id": patient_id,
                "patient_info": patient_info,
                "sessions": sessions,  # {visit_id: {"messages": ..., "ground_truth": ...}}
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(out_file)


def build_context() -> None:
    inp = Path(config.SEQUENCE_IN_PATH)
    out_path = Path(config.CONTEXT_OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    if inp.is_dir():
        patient_files = sorted([p for p in inp.glob("*.json") if p.is_file()])
    else:
        patient_files = [inp]

    outer_workers = getattr(config, "MAX_PATIENT_WORKERS", None) or 4
    inner_workers = getattr(config, "MAX_SESSION_WORKERS", None) or 8

    with ProcessPoolExecutor(max_workers=outer_workers) as ex:
        futures = [
            ex.submit(process_one_patient_file, str(p), str(out_path), inner_workers)
            for p in patient_files
        ]

        outer_pbar = tqdm(
            total=len(futures),
            desc="Patients",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )

        try:
            for fu in as_completed(futures):
                out_file = fu.result()
                print("Wrote:", out_file)
                outer_pbar.update(1)
        finally:
            outer_pbar.close()


if __name__ == "__main__":
    build_context()