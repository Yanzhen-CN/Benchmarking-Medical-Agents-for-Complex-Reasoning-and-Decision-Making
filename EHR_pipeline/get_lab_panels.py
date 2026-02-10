#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_panel_mapping_from_indicators_with_llm.py

Goal:
1) Scan patient JSON sequences to collect unique LAB indicators (analytes/measurements).
2) Ask LLM to infer a likely "test panel / order name" for EACH indicator (open set, no enum).
3) Merge/normalize panel names (LLM-driven canonicalization).
4) Output:
   - all_indicators.json
   - llm_batches/*.json
   - indicator_to_panel.json
   - panel_to_indicators.json
   - panel_merge_map.json
   - conflicts.json
"""

from __future__ import annotations

from tqdm import tqdm
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from util.logUtil import setup_logger
logger = setup_logger()

# -----------------------------
# Utils
# -----------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
        if p.name.startswith("."):
            continue
        yield p


# -----------------------------
# Step A: Collect indicators
# -----------------------------
def _norm_unit(u: str) -> str:
    u = norm(u)
    # 统一一些常见写法，避免 "mg / dL" vs "mg/dL"
    u = u.replace(" / ", "/").replace(" ", "")
    return u


def make_indicator_key(name: str, fluid: str, unit: str) -> str:
    """
    Stable composite key to avoid Blood/Urine name collision.
    Use: name||fluid||unit (unit optional but recommended)
    """
    name_n = norm(name)
    fluid_n = norm(fluid).title()  # Blood / Urine / Plasma ...
    unit_n = _norm_unit(unit)

    # unit 为空时也保留占位，避免 key 解析歧义
    return f"{name_n}||{fluid_n}||{unit_n or 'NA'}"


def collect_indicators(input_dir: Path) -> Dict[str, Any]:
    """
    Returns:
      indicators[indicator_key] = {
        "base_name": str,
        "fluid": str,
        "unit": str,
        "count": int,
        "categories": {category: count},
        "specimens": {specimen: count},
        "example_names": {raw_name: count},   # optional audit
      }
    """
    indicators: Dict[str, Any] = {}

    for fp in tqdm(iter_json_files(input_dir), desc="Collecting lab indicators"):
        obj = json.loads(fp.read_text(encoding="utf-8"))

        for v in obj.get("visits", []) or []:
            for ev in v.get("event_stream", []) or []:
                if (ev.get("type") or "").lower() != "lab":
                    continue

                for it in ev.get("items", []) or []:
                    raw_name = it.get("name") or ""
                    name = norm(raw_name)
                    if not name:
                        continue

                    cat = norm(it.get("category") or "")
                    fluid = norm(it.get("fluid") or "")
                    unit = _norm_unit(it.get("unit") or it.get("units") or "")
                    specimen = norm(it.get("specimen") or "")

                    key = make_indicator_key(name=name, fluid=fluid, unit=unit)

                    s = indicators.setdefault(
                        key,
                        {
                            "base_name": name,
                            "fluid": fluid.title() if fluid else "",
                            "unit": unit if unit else "",
                            "count": 0,
                            "categories": Counter(),
                            "specimens": Counter(),
                            "example_names": Counter(),
                        },
                    )
                    s["count"] += 1
                    s["example_names"][raw_name] += 1
                    if cat:
                        s["categories"][cat] += 1
                    if specimen:
                        s["specimens"][specimen] += 1

    out: Dict[str, Any] = {}
    for k, s in indicators.items():
        out[k] = {
            "base_name": s["base_name"],
            "fluid": s["fluid"],
            "unit": s["unit"],
            "count": s["count"],
            "categories": dict(s["categories"].most_common(20)),
            "specimens": dict(s["specimens"].most_common(20)),
            "example_names": dict(s["example_names"].most_common(10)),
        }

    logger.success(f"Collected {len(out)} unique indicators (composite-keyed) from {input_dir}")
    return out



def chunk_list(xs: List[Any], chunk_size: int) -> List[List[Any]]:
    return [xs[i : i + chunk_size] for i in range(0, len(xs), chunk_size)]


# -----------------------------
# LLM calls (use your util/llmUtil.py)
# -----------------------------
from util.llmUtil import LLMUtil  # noqa: E402

llm = LLMUtil()


def call_llm_json(prompt: str) -> Dict[str, Any]:
    return llm.chat_json(user_text=prompt, system_prompt="You are a clinical laboratory domain expert.")


def build_assign_prompt(batch: List[Tuple[str, Dict[str, Any]]]) -> str:
    """
    Ask LLM to assign each indicator -> a test panel/order name (open set).
    """
    items = []
    for indicator_key, st in batch:
        items.append(
            {
                "indicator_key": indicator_key,   # e.g. "WBC||Urine||#/hpf"
                "base_name": st["base_name"],     # e.g. "WBC"
                "fluid": st["fluid"],             # e.g. "Urine"
                "unit": st["unit"],               # e.g. "#/hpf"
                "specimens": st.get("specimens", {}),
            }
        )

    return f"""
You are given a list of lab *indicators* (analytes/measurements) extracted from EHR event streams.
Your job: infer the most likely "test panel / order name" that each indicator belongs to.

Important:
- Panel names are OPEN-SET. Do NOT use any fixed enum. You may create new panel names as needed.
- Prefer clinically standard panel/order names (e.g., "Basic Metabolic Panel", "Coagulation Studies", "Arterial Blood Gas",
  "Cardiac Markers", "Urinalysis", "Lipid Panel", "Inflammatory Markers", "Liver Function Tests", etc.).
- Output MUST be valid JSON only.
- Do NOT drop any indicator; every provided indicator must appear exactly once in assignments.
- If unsure, still assign a best-guess panel name AND list it in "uncertain" with a reason.
- The "assignments" keys MUST be the provided indicator_key exactly (do not rename).

Return JSON in this format:
{{
  "assignments": {{
    "Indicator A": "Panel Name 1",
    "Indicator B": "Panel Name 2"
  }},
  "new_panels": [
    {{
      "panel_name": "Panel Name X",
      "description": "short clinical description"
    }}
  ],
  "uncertain": [
    {{
      "indicator": "Indicator ...",
      "assigned_panel": "Panel Name ...",
      "reason": "why uncertain / alternatives"
    }}
  ]
}}

Here are the indicators:
{json.dumps(items, ensure_ascii=False)}
""".strip()


def build_canonicalize_prompt(panel_summaries: List[Dict[str, Any]]) -> str:
    """
    Ask LLM to merge/normalize panel names into a canonical set.
    """
    return f"""
You are given many "panel names" inferred from lab indicators across batches.
Because of naming variability (synonyms, abbreviations), you must canonicalize them.

Task:
- Create a canonical panel name set.
- Provide a mapping old_panel_name -> canonical_panel_name.
- Keep names clinically standard, concise, and consistent.
- If two names are essentially the same (e.g., "BMP" and "Basic Metabolic Panel"), merge.
- If truly different, keep separate.

Output MUST be valid JSON only.

Return JSON in this format:
{{
  "canonical_panels": [
    {{
      "canonical_name": "...",
      "aliases": ["old name 1", "old name 2"],
      "notes": "short rationale"
    }}
  ],
  "merge_map": {{
    "old name 1": "canonical name",
    "old name 2": "canonical name"
  }}
}}

Here are the panels (each with example indicators):
{json.dumps(panel_summaries, ensure_ascii=False)}
""".strip()


# -----------------------------
# Merge logic
# -----------------------------
def merge_assignments(batch_outputs: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Merge batch outputs into indicator->panel with voting.
    Conflicts occur if the same indicator was assigned to different panel names across batches
    (should be rare unless duplicates or reruns).
    """
    votes: Dict[str, Counter] = defaultdict(Counter)
    sources: Dict[str, List[int]] = defaultdict(list)

    for bi, out in enumerate(batch_outputs):
        assignments = out.get("assignments", {}) or {}
        for ind, panel in assignments.items():
            ind_n = norm(ind)
            panel_n = norm(panel)
            if not ind_n or not panel_n:
                continue
            votes[ind_n][panel_n] += 1
            sources[ind_n].append(bi)

    indicator_to_panel: Dict[str, str] = {}
    conflicts: Dict[str, Any] = {}

    for ind, cnt in votes.items():
        if not cnt:
            continue
        top_panel, top_n = cnt.most_common(1)[0]
        if len(cnt) > 1:
            conflicts[ind] = {
                "vote_counts": dict(cnt),
                "chosen": top_panel,
                "batch_ids": sources[ind],
            }
        indicator_to_panel[ind] = top_panel

    return indicator_to_panel, conflicts


def invert_mapping(indicator_to_panel: Dict[str, str]) -> Dict[str, List[str]]:
    panel_to_inds: Dict[str, List[str]] = defaultdict(list)
    for ind, panel in indicator_to_panel.items():
        panel_to_inds[panel].append(ind)
    # sort for stability
    return {p: sorted(inds) for p, inds in sorted(panel_to_inds.items(), key=lambda x: (-len(x[1]), x[0]))}


def summarize_panels(panel_to_inds: Dict[str, List[str]], max_examples: int = 30) -> List[Dict[str, Any]]:
    summaries = []
    for panel, inds in panel_to_inds.items():
        summaries.append(
            {
                "panel_name": panel,
                "n_indicators": len(inds),
                "example_indicators": inds[:max_examples],
            }
        )
    # sort big panels first
    summaries.sort(key=lambda x: (-x["n_indicators"], x["panel_name"]))
    return summaries


def apply_merge_map(indicator_to_panel: Dict[str, str], merge_map: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for ind, panel in indicator_to_panel.items():
        canon = merge_map.get(panel, panel)
        out[ind] = canon
    return out


# -----------------------------
# Main
# -----------------------------
def get_lab_panels():
    # You can keep BuildConfig like your original, but also allow CLI overrides.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Root folder containing patient JSON files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder")
    parser.add_argument("--batch_size", type=int, default=120, help="How many indicators per LLM call")
    parser.add_argument("--canonicalize_batch", type=int, default=80, help="How many panels per canonicalization call")
    args = parser.parse_args()

    input_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    batch_size = args.batch_size
    canonicalize_batch = args.canonicalize_batch

    if args.input_dir is None or args.output_dir is None:
        # fallback to your BuildConfig if provided
        try:
            from config import BuildConfig  # noqa: WPS433

            config = BuildConfig()
            input_dir = Path(config.labPanelExtract.INPUT_DIR)
            output_dir = Path(config.labPanelExtract.OUTPUT_DIR)
            # allow config override if present
            if hasattr(config.labPanelExtract, "BATCH_SIZE"):
                batch_size = int(config.labPanelExtract.BATCH_SIZE)
        except Exception as e:
            raise RuntimeError(
                "Please pass --input_dir and --output_dir, or ensure config.BuildConfig().labPanelExtract is available."
            ) from e
    else:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

    assert input_dir is not None and output_dir is not None

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "llm_batches").mkdir(parents=True, exist_ok=True)

    # Step A: collect indicators
    indicators = collect_indicators(input_dir)
    (output_dir / "all_indicators.json").write_text(
        json.dumps(indicators, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Sort by frequency
    indicators_sorted = sorted(indicators.items(), key=lambda x: (-x[1]["count"], x[0]))
    batches = chunk_list(indicators_sorted, batch_size)

    # Step B: LLM assign indicator -> open-set panel name
    batch_outputs: List[Dict[str, Any]] = []
    for bi, batch in enumerate(tqdm(batches, desc="LLM assigning panels")):
        prompt = build_assign_prompt(batch)
        out = call_llm_json(prompt)

        (output_dir / "llm_batches" / f"assign_batch_{bi:04d}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        batch_outputs.append(out)

    # Step C: merge assignments (indicator -> panel), invert to panel -> indicators
    indicator_to_panel_raw, conflicts = merge_assignments(batch_outputs)
    panel_to_inds_raw = invert_mapping(indicator_to_panel_raw)

    (output_dir / "indicator_to_panel.raw.json").write_text(
        json.dumps(indicator_to_panel_raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "panel_to_indicators.raw.json").write_text(
        json.dumps(panel_to_inds_raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "conflicts.json").write_text(json.dumps(conflicts, indent=2, ensure_ascii=False), encoding="utf-8")

    # Step D: canonicalize/merge panel names (LLM)
    panel_summaries = summarize_panels(panel_to_inds_raw, max_examples=30)
    canon_batches = chunk_list(panel_summaries, canonicalize_batch)

    merge_map_all: Dict[str, str] = {}
    canonical_panels_all: List[Dict[str, Any]] = []

    for ci, pb in enumerate(tqdm(canon_batches, desc="LLM canonicalizing panel names")):
        prompt = build_canonicalize_prompt(pb)
        out = call_llm_json(prompt)

        (output_dir / "llm_batches" / f"canonicalize_{ci:04d}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        mm = out.get("merge_map", {}) or {}
        # normalize keys/values a bit
        for k, v in mm.items():
            k2, v2 = norm(k), norm(v)
            if k2 and v2:
                merge_map_all[k2] = v2

        cps = out.get("canonical_panels", []) or []
        for cp in cps:
            canonical_panels_all.append(cp)

    (output_dir / "panel_merge_map.json").write_text(
        json.dumps(merge_map_all, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "canonical_panels.json").write_text(
        json.dumps(canonical_panels_all, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Step E: apply merge_map -> final outputs
    indicator_to_panel = apply_merge_map(indicator_to_panel_raw, merge_map_all)
    panel_to_inds = invert_mapping(indicator_to_panel)

    (output_dir / "indicator_to_panel.json").write_text(
        json.dumps(indicator_to_panel, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "panel_to_indicators.json").write_text(
        json.dumps(panel_to_inds, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Report
    logger.info("Final panels (top 30 by size):")
    for p, inds in list(panel_to_inds.items())[:30]:
        logger.info(f"- {p}: {len(inds)} indicators")

    logger.info(
        "\nWrote:\n"
        f"- {output_dir / 'all_indicators.json'}\n"
        f"- {output_dir / 'indicator_to_panel.json'}\n"
        f"- {output_dir / 'panel_to_indicators.json'}\n"
        f"- {output_dir / 'panel_merge_map.json'}\n"
        f"- {output_dir / 'conflicts.json'}\n"
        f"- {output_dir / 'llm_batches/'}"
    )


if __name__ == "__main__":
    get_lab_panels()
