#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_lab_panel_mapping_with_llm.py

1) Scan a folder of patient JSON sequences to collect unique lab test names (with simple stats).
2) Use LLM to classify each test into one of:
   CBC|BMP|CMP|LFT|COAG|ABG|LIPASE|CARDIAC|INFLAMMATORY|CUSTOM
3) Produce a complete mapping dictionary:
   lab_panel_map.json  (test_name -> panel)
plus:
   all_tests.json
   llm_batches/*.json
   conflicts.json
"""
from tqdm import tqdm
import os
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple
from util.logUtil import setup_logger
logger = setup_logger()

PANELS = ["CBC","BMP","CMP","LFT","COAG","ABG","LIPASE","CARDIAC","INFLAMMATORY","CUSTOM"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
        if p.name.startswith("."):
            continue
        yield p


def collect_tests(input_dir: Path) -> Dict[str, Any]:
    """
    Returns:
      tests[test_name] = {
        "count": int,
        "categories": {category: count},
        "fluids": {fluid: count},
      }
    """
    tests: Dict[str, Any] = {}
    for fp in tqdm(iter_json_files(input_dir), desc="Collecting lab tests"):
        obj = json.loads(fp.read_text(encoding="utf-8"))

        for v in obj.get("visits", []) or []:
            for ev in v.get("event_stream", []) or []:
                if (ev.get("type") or "").lower() != "lab":
                    continue
                for it in ev.get("items", []) or []:
                    name = norm(it.get("name") or "")
                    if not name:
                        continue
                    cat = norm(it.get("category") or "")
                    fluid = norm(it.get("fluid") or "")

                    s = tests.setdefault(name, {"count": 0, "categories": Counter(), "fluids": Counter()})
                    s["count"] += 1
                    if cat:
                        s["categories"][cat] += 1
                    if fluid:
                        s["fluids"][fluid] += 1

    # convert counters to dict for json
    out = {}
    for k, s in tests.items():
        out[k] = {
            "count": s["count"],
            "categories": dict(s["categories"].most_common(20)),
            "fluids": dict(s["fluids"].most_common(20)),
        }
    logger.success(f"Collected {len(out)} unique lab tests from {input_dir}")
    return out


# -----------------------------
# LLM part: implement this using your util/llmUtil.py
# -----------------------------
from util.llmUtil import LLMUtil
llm = LLMUtil()
def call_llm_json(prompt: str) -> Dict[str, Any]:
    return llm.chat_json(user_text=prompt, system_prompt="You are a medical lab panel classifier.")


def build_prompt(batch: List[Tuple[str, Dict[str, Any]]]) -> str:
    """
    batch: list of (test_name, stats)
    """
    items_text = []
    for name, st in batch:
        items_text.append({
            "test_name": name,
            "categories": st.get("categories", {}),
            "fluids": st.get("fluids", {}),
        })

    return f"""
You are given a list of lab test items extracted from EHR event streams.
Your job is to classify EACH test into exactly ONE panel from this enum:

{PANELS}

Guidelines:
- Use the test_name primarily; categories/fluids are hints.
- If uncertain or panel does not fit, choose CUSTOM.
- Output MUST be valid JSON only.
- Do NOT drop any item; every provided test_name must appear exactly once in mapping.

Return JSON in this format:
{{
  "mapping": {{
    "test_name_1": "PANEL",
    "test_name_2": "PANEL"
  }},
  "uncertain": [
    {{
      "test_name": "...",
      "suggested_panel": "...",
      "reason": "..."
    }}
  ]
}}

Here are the tests:
{json.dumps(items_text, ensure_ascii=False)}
""".strip()


def chunk_list(xs: List[Any], chunk_size: int) -> List[List[Any]]:
    return [xs[i:i+chunk_size] for i in range(0, len(xs), chunk_size)]


def merge_mappings(batch_outputs: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Returns:
      final_map: test_name -> panel
      conflicts: test_name -> {panels: [..], sources: [...]}
    """
    votes: Dict[str, Counter] = defaultdict(Counter)
    sources: Dict[str, List[int]] = defaultdict(list)

    for i, out in enumerate(batch_outputs):
        mapping = out.get("mapping", {}) or {}
        for test_name, panel in mapping.items():
            votes[test_name][panel] += 1
            sources[test_name].append(i)

    final_map: Dict[str, str] = {}
    conflicts: Dict[str, Any] = {}

    for test_name, cnt in votes.items():
        if not cnt:
            continue
        top_panel, top_n = cnt.most_common(1)[0]
        # conflict if multiple panels have same top count or more than 1 distinct panel
        if len(cnt) > 1:
            conflicts[test_name] = {
                "vote_counts": dict(cnt),
                "chosen": top_panel,
                "batch_ids": sources[test_name],
            }
        final_map[test_name] = top_panel

    return final_map, conflicts


def main():
    from config import BuildConfig
    config = BuildConfig()

    input_dir = config.labPanelExtract.INPUT_DIR
    out_dir = config.labPanelExtract.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "llm_batches").mkdir(parents=True, exist_ok=True)

    # Step A: collect tests
    tests = collect_tests(input_dir)
    (out_dir / "all_tests.json").write_text(json.dumps(tests, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Sort by frequency (most common first)
    tests_sorted = sorted(tests.items(), key=lambda x: (-x[1]["count"], x[0]))
    batches = chunk_list(tests_sorted, config.labPanelExtract.BATCH_SIZE)

    batch_outputs: List[Dict[str, Any]] = []

    # Step B: LLM classify
    for bi, batch in enumerate(batches):
        prompt = build_prompt(batch)
        out = call_llm_json(prompt)

        # Save each batch output for audit
        (out_dir / "llm_batches" / f"batch_{bi:04d}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        batch_outputs.append(out)

    # Step C: merge + conflicts
    final_map, conflicts = merge_mappings(batch_outputs)

    (out_dir / "lab_panel_map.json").write_text(
        json.dumps(final_map, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    (out_dir / "conflicts.json").write_text(
        json.dumps(conflicts, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Step D: report available panels
    panel_counts = Counter(final_map.values())
    logger.info("Panels inferred by LLM:")
    for p in PANELS:
        logger.info(f"- {p}: {panel_counts.get(p, 0)} tests")
    logger.info(f"\nWrote:\n- {out_dir / 'all_tests.json'}\n- {out_dir / 'lab_panel_map.json'}\n- {out_dir / 'conflicts.json'}\n- {out_dir / 'llm_batches/'}")


if __name__ == "__main__":
    main()
