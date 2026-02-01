#!/usr/bin/env python3
"""
Generate structured questions from selected events.

Design:
- LLM outputs only {question(s), keywords}. Ground truth is attached locally.
- LAB / MEDICATION are batched; IMAGING + ADMISSION/DISCHARGE are single.
- Two question variants for LAB / MEDICATION / IMAGING:
  (1) explicit: event_id + timestamp
  (2) relative: visit_num + event_type + visit_type_index
- ADMISSION/DISCHARGE combined into one question per visit (single variant).

Env keys (Qwen/OpenAI compatible, per config.py):
  OPENAI_API_KEY, OPENAI_API_BASE_URL
  QGEN_MODEL, QGEN_BATCH_LAB, QGEN_BATCH_MED, QGEN_SEED
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running this file directly.
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import FactQGenConfig
from util.llmUtil import LLMUtil


def parse_args() -> argparse.Namespace:
    cfg = FactQGenConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events-dir", type=Path, default=cfg.EVENTS_SELECTED_DIR)
    p.add_argument("--out-dir", type=Path, default=cfg.QUESTIONS_OUT_DIR)
    p.add_argument("--model", type=str, default=cfg.MODEL)
    p.add_argument("--batch-lab", type=int, default=cfg.BATCH_SIZE_LAB)
    p.add_argument("--batch-med", type=int, default=cfg.BATCH_SIZE_MED)
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    p.add_argument("--num-patients", type=int, default=0, help="process only first N patients; 0 means all")
    return p.parse_args()


def iter_patient_files(inp: Path) -> Iterable[Path]:
    if inp.is_dir():
        yield from sorted(inp.glob("P*.jsonl"))
    else:
        yield inp


def _visit_num(visit_ref: str) -> str:
    if visit_ref and visit_ref.startswith("V"):
        return visit_ref[1:]
    return visit_ref


def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _load_events(path: Path) -> List[Dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _collect_lab_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [it for it in items if it.get("name")]


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_similar_name(a: str, b: str) -> bool:
    na = _normalize_name(a)
    nb = _normalize_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na.startswith(nb) or nb.startswith(na):
        return True
    if na in nb or nb in na:
        return True
    return False


def _pick_negative_lab_name(
    rng: random.Random,
    current_names: List[str],
    pool_names: List[str],
) -> Optional[str]:
    if not pool_names:
        return None
    rng.shuffle(pool_names)
    for cand in pool_names:
        if any(_is_similar_name(cand, cur) for cur in current_names):
            continue
        return cand
    return None


def _collect_med_drugs_with_dose(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    drugs = []
    seen = set()
    for it in items:
        drug = it.get("drug")
        if not drug:
            continue
        key = (drug, it.get("dose"))
        if key in seen:
            continue
        seen.add(key)
        drugs.append({"drug": drug, "dose": it.get("dose")})
    return drugs


def _format_adm_dis_question(item: Dict[str, Any]) -> str:
    return (
        f"In the {item['visit_num']}th visit, what were the chief complaint and the discharge diagnosis?"
    )


def _make_question_id(event_id: str, suffix: str) -> str:
    return f"{event_id}-{suffix}"


def _format_lab_questions(item: Dict[str, Any]) -> Tuple[str, str]:
    exp = (
        f"During the lab event with event_id {item['event_id']} at timestamp {item['timestamp']}, "
        f"was {item['lab_name']} measured? If yes, what was the value (with unit)? "
        "If not, answer 'not found'."
    )
    rel = (
        f"During the {item['visit_type_index']}th lab event in the {item['visit_num']}th visit, "
        f"was {item['lab_name']} measured? If yes, what was the value (with unit)? "
        "If not, answer 'not found'."
    )
    return exp, rel


def _format_med_questions(item: Dict[str, Any]) -> Tuple[str, str]:
    exp = (
        f"In the medication event with event_id {item['event_id']} at timestamp {item['timestamp']}, "
        "what medication was given, and if the dose was recorded, what was the dose?"
    )
    rel = (
        f"In the {item['visit_type_index']}th medication event in the {item['visit_num']}th visit, "
        "what medication was given, and if the dose was recorded, what was the dose?"
    )
    return exp, rel


def _format_imaging_questions(item: Dict[str, Any]) -> Tuple[str, str]:
    exp = (
        f"Describe the findings from the imaging report with event_id {item['event_id']} "
        f"at timestamp {item['timestamp']}."
    )
    rel = (
        f"Describe the findings from the imaging report from the {item['visit_type_index']}th imaging event "
        f"in the {item['visit_num']}th visit."
    )
    return exp, rel


def _build_imaging_keywords_prompt(item: Dict[str, Any]) -> Tuple[str, str]:
    system = "You extract English medical keywords from a report. Return JSON object only. Do NOT add extra keys."
    prompt_item = {"report": item["report"]}
    user = {
        "task": "Extract 3-8 short keyword phrases from the report.",
        "rules": [
            "Keywords must be copied from the report text.",
            "Return an empty list if the report is empty.",
        ],
        "output_schema": {"keywords": ["string"]},
        "item": prompt_item,
    }
    return system, json.dumps(user, ensure_ascii=False)


def _chat_json(llm: LLMUtil, model: str, system: str, user: str) -> Any:
    return llm.chat_json(system_prompt=system, user_text=user, model=model, temperature=0.0)


def _adm_dis_keywords(item: Dict[str, Any]) -> List[str]:
    keys = []
    cc = item.get("chief_complaint")
    dd = item.get("discharge_diagnosis")
    if cc:
        keys.append(cc)
    if dd:
        keys.append(dd)
    return keys


def _visit_sort_key(visit_ref: Optional[str]) -> Tuple[int, str]:
    if not visit_ref:
        return (0, "")
    if visit_ref.startswith("V") and visit_ref[1:].isdigit():
        return (int(visit_ref[1:]), visit_ref)
    return (0, visit_ref)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    llm = LLMUtil()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_files = list(iter_patient_files(args.events_dir))
    if args.num_patients and args.num_patients > 0:
        patient_files = patient_files[: args.num_patients]

    for pf in patient_files:
        events = _load_events(pf)
        if not events:
            continue

        # Build per-patient LAB name pool for negative sampling
        all_lab_names: List[str] = []
        for e in events:
            if e.get("event_type") != "LAB":
                continue
            items = _collect_lab_items(_safe_list((e.get("content") or {}).get("items")))
            names = [it.get("name") for it in items if it.get("name")]
            if not names:
                continue
            all_lab_names.extend(names)

        patient_id = events[0].get("patient_id") or pf.stem
        out_path = out_dir / f"{patient_id}.jsonl"
        questions: List[Dict[str, Any]] = []

        # Group by visit for admission/discharge pairing
        by_visit: Dict[str, List[Dict[str, Any]]] = {}
        for ev in events:
            by_visit.setdefault(ev.get("visit_ref") or "UNKNOWN_VISIT", []).append(ev)

        # Prepare lab/med/imaging lists
        lab_items: List[Dict[str, Any]] = []
        med_items: List[Dict[str, Any]] = []
        imaging_items: List[Dict[str, Any]] = []
        adm_dis_items: List[Dict[str, Any]] = []

        for visit_ref, evs in by_visit.items():
            admission = next((e for e in evs if e.get("event_type") == "ADMISSION"), None)
            discharge = next((e for e in evs if e.get("event_type") == "DISCHARGE"), None)
            if admission and discharge:
                adm_content = admission.get("content") or {}
                dis_content = discharge.get("content") or {}
                dis_note = dis_content.get("discharge_note") or {}
                adm_dis_items.append(
                    {
                        "patient_id": patient_id,
                        "visit_ref": visit_ref,
                        "visit_num": _visit_num(visit_ref),
                        "chief_complaint": adm_content.get("chief_complaint"),
                        "discharge_diagnosis": dis_note.get("discharge_diagnosis")
                        or dis_content.get("discharge_diagnosis"),
                    }
                )

            for e in evs:
                et = e.get("event_type")
                if et == "LAB":
                    items = _collect_lab_items(_safe_list((e.get("content") or {}).get("items")))
                    if not items:
                        continue
                    current_names = [it.get("name") for it in items if it.get("name")]
                    # 50/50 choose existing vs negative (fallback to existing if no negative)
                    use_existing = rng.random() < 0.5
                    pick = None
                    lab_name = None
                    if use_existing:
                        pick = rng.choice(items)
                        lab_name = pick.get("name")
                    else:
                        pool = [n for n in all_lab_names if n and n not in current_names]
                        neg = _pick_negative_lab_name(rng, current_names, pool)
                        if neg:
                            lab_name = neg
                        else:
                            pick = rng.choice(items)
                            lab_name = pick.get("name")
                    lab_items.append(
                        {
                            "id": _make_question_id(e.get("event_id") or "", "lab"),
                            "patient_id": patient_id,
                            "visit_ref": visit_ref,
                            "visit_num": _visit_num(visit_ref),
                            "visit_type_index": e.get("visit_type_index"),
                            "event_id": e.get("event_id"),
                            "timestamp": e.get("timestamp"),
                            "lab_name": lab_name,
                            "value_num": pick.get("value_num") if pick else None,
                            "value_text": pick.get("value_text") if pick else None,
                            "unit": pick.get("unit") if pick else None,
                            "lab_exists": bool(pick),
                            "event_content": e.get("content"),
                        }
                    )
                elif et == "MEDICATION":
                    items = _safe_list((e.get("content") or {}).get("items"))
                    drugs = _collect_med_drugs_with_dose(items)
                    if not drugs:
                        continue
                    med_items.append(
                        {
                            "id": _make_question_id(e.get("event_id") or "", "med"),
                            "patient_id": patient_id,
                            "visit_ref": visit_ref,
                            "visit_num": _visit_num(visit_ref),
                            "visit_type_index": e.get("visit_type_index"),
                            "event_id": e.get("event_id"),
                            "timestamp": e.get("timestamp"),
                            "drugs": drugs,
                            "event_content": e.get("content"),
                        }
                    )
                elif et == "IMAGING":
                    imaging_items.append(
                        {
                            "id": _make_question_id(e.get("event_id") or "", "img"),
                            "patient_id": patient_id,
                            "visit_ref": visit_ref,
                            "visit_num": _visit_num(visit_ref),
                            "visit_type_index": e.get("visit_type_index"),
                            "event_id": e.get("event_id"),
                            "timestamp": e.get("timestamp"),
                            "report": e.get("content"),
                        }
                    )

        # Admission/Discharge (single, no batch)
        for item in adm_dis_items:
            question = _format_adm_dis_question(item)
            keywords = _adm_dis_keywords(item)
            q = {
                "question_id": f"{patient_id}-{item['visit_ref']}-adm_dis",
                "patient_id": patient_id,
                "visit_ref": item["visit_ref"],
                "visit_num": item["visit_num"],
                "event_type": "ADMISSION_DISCHARGE",
                "question_variant": "single",
                "question": question,
                "keywords": keywords,
                "ground_truth": {
                    "chief_complaint": item.get("chief_complaint"),
                    "discharge_diagnosis": item.get("discharge_diagnosis"),
                },
            }
            questions.append(q)

        # Imaging (single, no batch)
        for item in imaging_items:
            system, user = _build_imaging_keywords_prompt(item)
            obj = _chat_json(llm, args.model, system, user)
            keywords = obj.get("keywords") if isinstance(obj, dict) else []
            base = {
                "patient_id": patient_id,
                "visit_ref": item["visit_ref"],
                "visit_num": item["visit_num"],
                "event_id": item["event_id"],
                "event_type": "IMAGING",
                "visit_type_index": item["visit_type_index"],
                "timestamp": item["timestamp"],
                "keywords": keywords,
                "ground_truth": item.get("report"),
            }
            q_exp, q_rel = _format_imaging_questions(item)
            q1 = {
                **base,
                "question_id": f"{item['id']}",
                "question_variant": "both",
                "question_explicit": q_exp,
                "question_relative": q_rel,
            }
            questions.append(q1)

        # LAB (batched)
        for i in range(0, len(lab_items), args.batch_lab):
            batch = lab_items[i : i + args.batch_lab]
            for item in batch:
                q_exp, q_rel = _format_lab_questions(item)
                answer = "not found"
                if item.get("lab_exists"):
                    v = item.get("value_num") if item.get("value_num") not in (None, "") else item.get("value_text")
                    unit = item.get("unit") or ""
                    answer = f"{v} {unit}".strip()
                keywords = ["not found"]
                if item.get("lab_exists"):
                    keywords = ["found", answer]
                base = {
                    "patient_id": patient_id,
                    "visit_ref": item["visit_ref"],
                    "visit_num": item["visit_num"],
                    "event_id": item["event_id"],
                    "event_type": "LAB",
                    "visit_type_index": item["visit_type_index"],
                    "timestamp": item["timestamp"],
                    "keywords": keywords,
                    "ground_truth": {
                        "event_content": item.get("event_content"),
                    },
                }
                q1 = {
                    **base,
                    "question_id": f"{item['id']}",
                    "question_variant": "both",
                    "question_explicit": q_exp,
                    "question_relative": q_rel,
                }
                questions.append(q1)

        # MEDICATION (batched)
        for i in range(0, len(med_items), args.batch_med):
            batch = med_items[i : i + args.batch_med]
            for item in batch:
                q_exp, q_rel = _format_med_questions(item)
                med_keywords = []
                for d in item.get("drugs", []):
                    drug = d.get("drug")
                    if not drug:
                        continue
                    dose = d.get("dose") or "not recorded"
                    med_keywords.append(f"{drug} | {dose}")
                base = {
                    "patient_id": patient_id,
                    "visit_ref": item["visit_ref"],
                    "visit_num": item["visit_num"],
                    "event_id": item["event_id"],
                    "event_type": "MEDICATION",
                    "visit_type_index": item["visit_type_index"],
                    "timestamp": item["timestamp"],
                    "keywords": med_keywords,
                    "ground_truth": {
                        "event_content": item.get("event_content"),
                        "drug_list": item.get("drugs"),
                    },
                }
                q1 = {
                    **base,
                    "question_id": f"{item['id']}",
                    "question_variant": "both",
                    "question_explicit": q_exp,
                    "question_relative": q_rel,
                }
                questions.append(q1)

        def _q_sort_key(q: Dict[str, Any]) -> Tuple[int, int, str]:
            vkey = _visit_sort_key(q.get("visit_ref"))
            is_adm_dis = 1 if q.get("event_type") == "ADMISSION_DISCHARGE" else 0
            return (vkey[0], is_adm_dis, q.get("question_id") or "")

        questions.sort(key=_q_sort_key)

        with out_path.open("w", encoding="utf-8") as out_f:
            for q in questions:
                out_f.write(json.dumps(q, ensure_ascii=False) + "\n")

        print(f"wrote={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
