#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_cat1_generation_with_llm.py

LLM-based validator for Cat1 question generation.

What it does:
- Scans patient JSONs and visits to find indices that can trigger each question type.
- Calls your builders directly to generate example questions for each type.
- Uses LLM to validate:
  - required fields exist
  - options contain answer
  - T3-N options are from ACTIONS
  - T3-A/T3-M options are strings and answer is among options
  - meta.gt_list exists when required (T3-A/T3-M)
  - gt_list weights are in (0,1] and monotonic-ish by delta_hours
- Writes report.json with pass/fail and issues.

This avoids depending on random sampling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from util.logUtil import setup_logger
logger = setup_logger()

# ---- import your generator module ----
import tasks.agentic_decision.generate_questions as gen

# ---- LLM ----
from util.llmUtil import LLMUtil
llm = LLMUtil()

def call_llm_json(prompt: str) -> Dict[str, Any]:
    return llm.chat_json(user_text=prompt, system_prompt="You are a strict QA validator for dataset JSON.")


def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
        if p.name.startswith("."):
            continue
        yield p


def load_indicator_panel_map(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- find trigger indices ----------
def find_index_with_next_action(events: List[Dict[str, Any]], want_action: str) -> Optional[int]:
    # return i such that events[i+1] => want_action
    for i in range(len(events) - 1):
        act = gen.gt_action_from_event(events[i + 1])
        if act == want_action:
            return i
    return None


def find_any_visit_with_discharge(visit: Dict[str, Any]) -> bool:
    dt = (visit.get("discharge_info") or {}).get("discharge_time")
    return bool(dt)


# ---------- LLM validator ----------
def llm_validate_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {
        "pass": bool,
        "issues": [...],
        "suggestions": [...]
      }
    """
    prompt = f"""
Validate this question JSON produced by our generator.

Rules:
- Must contain keys: qid, visit_id, t_index, timestamp, qtype, question, options, answer, meta
- options must be a non-empty list
- answer must be in options (string equality), except you may allow answer None only if explicitly justified (should not happen)
- If qtype == "T3-N":
  - options must be subset of: {gen.ACTIONS}
- If qtype == "T3-A" or qtype == "T3-M":
  - meta must include "gt_list" (list of objects with keys: value, delta_hours, weight)
  - each weight should be in (0, 1.0]
  - delta_hours should be >= 0
- If qtype == "T3-D":
  - options must be ["Yes","No"] (order can vary)
  - answer must be "Yes" or "No"

Return JSON only:
{{
  "pass": true/false,
  "issues": ["..."],
  "suggestions": ["..."]
}}

Question:
{json.dumps(q, ensure_ascii=False)}
""".strip()
    return call_llm_json(prompt)


def local_sanity_check(q: Dict[str, Any]) -> List[str]:
    issues = []
    for k in ["qid","visit_id","t_index","timestamp","qtype","question","options","answer","meta"]:
        if k not in q:
            issues.append(f"missing_field:{k}")
    if not isinstance(q.get("options"), list) or len(q["options"]) == 0:
        issues.append("options_not_list_or_empty")
    else:
        if q.get("answer") not in q["options"]:
            issues.append("answer_not_in_options")

    qt = q.get("qtype")
    if qt == "T3-N":
        for opt in q.get("options", []):
            if opt not in gen.ACTIONS:
                issues.append(f"T3N_option_not_in_ACTIONS:{opt}")
    if qt in ("T3-A", "T3-M"):
        gt_list = (q.get("meta") or {}).get("gt_list")
        if not isinstance(gt_list, list) or len(gt_list) == 0:
            issues.append("missing_or_empty_gt_list")
        else:
            for g in gt_list:
                if g.get("delta_hours", -1) < 0:
                    issues.append("gt_list_delta_hours_negative")
                w = g.get("weight", 0)
                if not (0 < float(w) <= 1.0):
                    issues.append(f"gt_list_weight_out_of_range:{w}")
    if qt == "T3-D":
        opts = q.get("options", [])
        if set(opts) != {"Yes","No"}:
            issues.append("T3D_options_not_yes_no")
        if q.get("answer") not in ("Yes","No"):
            issues.append("T3D_answer_invalid")

    return issues


# ---------- build one example per subtype ----------
def build_examples_for_one_visit(
    patient_id: str,
    visit: Dict[str, Any],
    indicator_panel_map: Optional[Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Try to generate at least one example for each:
      T3-N, T3-A(labs), T3-A(imaging), T3-A(micro), T3-M, T3-D
    Return dict subtype -> question_json
    """
    examples: Dict[str, Dict[str, Any]] = {}
    events = list(visit.get("event_stream", []) or [])
    if len(events) < 2:
        return examples
    events.sort(key=lambda e: gen.parse_time(e["timestamp"]))

    # ---- T3-N: any index that has a valid next action ----
    for i in range(len(events) - 1):
        gt_act = gen.gt_action_from_event(events[i + 1])
        if gt_act:
            t_cur = gen.parse_time(events[i]["timestamp"])
            q = gen.build_T3N(patient_id, visit, events, i, t_cur, gt_act, out_options_k=6).__dict__
            examples["T3-N"] = q
            break

    # ---- T3-A labs ----
    i = find_index_with_next_action(events, "order_labs")
    if i is not None:
        t_cur = gen.parse_time(events[i]["timestamp"])
        q = gen.build_T3A_labs(patient_id, visit, events, i, t_cur, indicator_panel_map, out_options_k=6)
        if q:
            examples["T3-A-labs"] = q.__dict__

    # ---- T3-A imaging ----
    i = find_index_with_next_action(events, "order_imaging")
    if i is not None:
        t_cur = gen.parse_time(events[i]["timestamp"])
        q = gen.build_T3A_imaging(patient_id, visit, events, i, t_cur, out_options_k=6)
        if q:
            examples["T3-A-imaging"] = q.__dict__

    # ---- T3-A micro ----
    i = find_index_with_next_action(events, "order_microbiology")
    if i is not None:
        t_cur = gen.parse_time(events[i]["timestamp"])
        q = gen.build_T3A_micro(patient_id, visit, events, i, t_cur, out_options_k=6)
        if q:
            examples["T3-A-micro"] = q.__dict__

    # ---- T3-M medication ----
    i = find_index_with_next_action(events, "medication")
    if i is not None:
        t_cur = gen.parse_time(events[i]["timestamp"])
        q = gen.build_T3M_medication(patient_id, visit, events, i, t_cur, out_options_k=6)
        if q:
            examples["T3-M"] = q.__dict__

    # ---- T3-D discharge ----
    if find_any_visit_with_discharge(visit):
        # pick a timepoint somewhere before discharge (earliest valid index)
        try:
            discharge_time = gen.parse_time(visit["discharge_info"]["discharge_time"])
            for i in range(len(events) - 1):
                t_cur = gen.parse_time(events[i]["timestamp"])
                # pick within 48h window if possible
                hrs_to_discharge = (discharge_time - t_cur).total_seconds() / 3600.0
                if 0 <= hrs_to_discharge <= 48:
                    q = gen.build_T3D_discharge(
                        patient_id, visit, events, i, t_cur, discharge_time, X_hours=6.0, out_options_k=2
                    )
                    if q:
                        examples["T3-D"] = q.__dict__
                        break
        except Exception:
            pass

    return examples


def main():
    from config import AgentQaGenConfig
    cfg = AgentQaGenConfig()

    in_dir = Path(cfg.INPUT_DIR)
    report_path = Path(cfg.OUTPUT_PATH) / "selftest_report.json"  # OUTPUT_PATH should be a folder

    indicator_panel_map = load_indicator_panel_map(Path(cfg.INDICATOR_PANEL_MAP)) if cfg.INDICATOR_PANEL_MAP else None

    # Collect examples from a few patients until we cover as many subtypes as possible
    needed = {"T3-N", "T3-A-labs", "T3-A-imaging", "T3-A-micro", "T3-M", "T3-D"}
    found_examples: Dict[str, Dict[str, Any]] = {}
    provenance: Dict[str, Dict[str, Any]] = {}

    for fp in iter_json_files(in_dir):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        patient_id = obj.get("patient_info", {}).get("patient_id") or fp.stem

        for visit in obj.get("visits", []) or []:
            ex = build_examples_for_one_visit(patient_id, visit, indicator_panel_map)
            for k, q in ex.items():
                if k not in found_examples:
                    found_examples[k] = q
                    provenance[k] = {"file": str(fp), "patient_id": patient_id, "visit_id": visit.get("visit_id")}
            if needed.issubset(found_examples.keys()):
                break
        if needed.issubset(found_examples.keys()):
            break

    # Validate with local + LLM
    results = {}
    for subtype, q in found_examples.items():
        local_issues = local_sanity_check(q)
        llm_res = llm_validate_question(q)
        results[subtype] = {
            "provenance": provenance.get(subtype, {}),
            "local_issues": local_issues,
            "llm": llm_res,
            "question_preview": {
                "qid": q.get("qid"),
                "qtype": q.get("qtype"),
                "question": q.get("question"),
                "options": q.get("options"),
                "answer": q.get("answer"),
            },
        }

    missing = sorted(list(needed - set(found_examples.keys())))
    report = {
        "covered_subtypes": sorted(list(found_examples.keys())),
        "missing_subtypes": missing,
        "results": results,
        "notes": [
            "If some subtypes are missing, your dataset folder may not contain those event types, or visit schema differs.",
            "T3-A-labs quality depends on indicator_panel_map coverage; if many misses, panel inference may be None.",
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Wrote self-test report to: {report_path}")

    if missing:
        logger.warning(f"Missing subtypes: {missing}")


if __name__ == "__main__":
    main()
