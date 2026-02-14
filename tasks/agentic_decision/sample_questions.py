#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sample_and_merge_questions.py

Read all *.jsonl in in_dir, sample ratio of questions, merge T3-A/M/P -> T3-A,
and write to out_dir (default: question_sorted) preserving file names.

Default sampling mode: global (sample across all questions).
Optionally: per_file (sample ratio within each file).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, DefaultDict
from collections import defaultdict

MERGE_TO_A = {"T3-A", "T3-M", "T3-P"}


def iter_jsonl_files(in_dir: Path) -> List[Path]:
    return sorted(in_dir.glob("*.jsonl"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"JSON parse error in {path} line {ln}: {e}") from e
            if not isinstance(obj, dict):
                obj = {"_value": obj}
            items.append(obj)
    return items


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


def merge_t3_amp_to_a(item: Dict[str, Any]) -> None:
    """
    Your file uses: qtype + qid.
    We'll:
      - if qtype in {T3-A,T3-M,T3-P} -> set to T3-A
      - if qid contains '-T3-M-' or '-T3-P-' -> replace to '-T3-A-'
    Also keep compatibility with other possible type keys.
    """
    # qtype (primary in your data)
    qtype = item.get("qtype")
    if isinstance(qtype, str) and qtype.strip() in MERGE_TO_A:
        item["qtype"] = "T3-A"

    # other possible keys
    for k in ("type", "question_type", "qtype"):
        if k in item and isinstance(item[k], str) and item[k].strip() in MERGE_TO_A:
            item[k] = "T3-A"

    # qid rewrite (only when clearly formatted with '-T3-X-')
    qid = item.get("qid")
    if isinstance(qid, str):
        item["qid"] = qid.replace("-T3-M-", "-T3-A-").replace("-T3-P-", "-T3-A-")


def sample_k(n: int, ratio: float, rng: random.Random, min_keep: int) -> List[int]:
    if n <= 0 or ratio <= 0:
        return []
    k = int(math.floor(n * ratio))
    if min_keep > 0:
        k = max(k, min_keep)
    k = min(k, n)
    if k <= 0:
        return []
    idx = rng.sample(range(n), k)
    idx.sort()
    return idx

from config import AgentQaGenConfig
def main():
    cfg = AgentQaGenConfig()

    in_dir = Path(cfg.OUTPUT_PATH)
    out_dir = Path(cfg.SAMPLE_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_jsonl_files(in_dir)
    if not files:
        raise SystemExit(f"No *.jsonl files found in: {in_dir}")

    rng = random.Random(cfg.SAMPLE_SEED)

    # load & normalize
    per_file_items: List[Tuple[Path, List[Dict[str, Any]]]] = []
    total_in = 0
    for fp in files:
        items = read_jsonl(fp)
        total_in += len(items)
        for it in items:
            merge_t3_amp_to_a(it)
        per_file_items.append((fp, items))

    kept_by_file: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    if cfg.SAMPLE_MODE == "per_file":
        total_out = 0
        for fp, items in per_file_items:
            keep_idx = sample_k(len(items), cfg.SAMPLE_RATIO, rng, cfg.SAMPLE_MIN_KEEP)
            sampled = [items[i] for i in keep_idx]
            kept_by_file[fp.name] = sampled
            total_out += len(sampled)

    else:  # global
        # flatten while remembering origin + original order within file
        flat: List[Tuple[str, int, Dict[str, Any]]] = []
        for fp, items in per_file_items:
            for i, it in enumerate(items):
                flat.append((fp.name, i, it))

        keep_idx = sample_k(len(flat), cfg.SAMPLE_RATIO, rng, cfg.SAMPLE_MIN_KEEP)
        keep_set = set(keep_idx)

        # rebuild per file, preserving original order in each file
        for j, (fname, order_i, it) in enumerate(flat):
            if j in keep_set:
                kept_by_file[fname].append(it)

        total_out = sum(len(v) for v in kept_by_file.values())

        # also create empty files for those with 0 kept? (usually no need)
        # Here we only write files that exist in input, even if empty.
        for fp, _items in per_file_items:
            kept_by_file.setdefault(fp.name, [])

    # write outputs
    for fp, _items in per_file_items:
        out_fp = out_dir / fp.name
        write_jsonl(out_fp, kept_by_file[fp.name])

    print(f"[DONE] in_dir={in_dir} -> out_dir={out_dir}")
    print(f"[STATS] mode={cfg.SAMPLE_MODE} files={len(files)} total_in={total_in} total_out={total_out} ratio={cfg.SAMPLE_RATIO} seed={cfg.SAMPLE_SEED}")


if __name__ == "__main__":
    main()
