import sys
import os
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import copy
import argparse
import json
import glob
from tqdm import tqdm
from datetime import datetime

try:
    from config import TimelineGenConfig
except ImportError:
    print("TimelineGenConfig not found, using default")
    class TimelineGenConfig:
        def __init__(self):
            self.RANDOM_SEED = 42
            self.SORTING_WINDOW_SIZE = 5          # number of visits per question
            self.SORTING_STRIDE = 1
            self.PATIENTS_SEQ_DIR = PROJECT_ROOT / "bench_data" / "patients_sequence"
            self.VISIT_SORTING_DIR = PROJECT_ROOT / "question_data" / "visit_sorting"

def parse_ts(ts_str):
    if not ts_str: return None
    try: return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except: return None

def is_valid_text(text):
    if not text: return False
    s = str(text).strip()
    if not s or s.upper() in ["N/A", "NONE", "NULL", "UNKNOWN"]:
        return False
    return len(s) > 5

def extract_visit_summary(visit_events, visit_ref):
    """
    Combine admission and discharge of a single visit into one summary.
    Returns a dict with keys: ref, original_event_id (admission id), content, timestamp, type='VISIT'.
    If either admission or discharge is missing/invalid, returns None.
    """
    sorted_events = sorted(visit_events, key=lambda x: x.get('timestamp') or "9999-12-31")
    adm_event = next((e for e in sorted_events if e.get('event_type') == 'ADMISSION'), None)
    dis_event = next((e for e in reversed(sorted_events) if e.get('event_type') == 'DISCHARGE'), None)
    if not adm_event or not dis_event:
        return None

    start_time = parse_ts(adm_event.get('timestamp'))
    adm_content = adm_event.get('content', {})
    dis_content = dis_event.get('content', {})

    parts = []
    # Admission part
    if isinstance(adm_content, dict):
        chief = adm_content.get('chief_complaint')
        hpi = adm_content.get('history_of_present_illness')
        if is_valid_text(chief):
            parts.append(f"Admission Chief Complaint: {chief}")
        if is_valid_text(hpi):
            parts.append(f"Admission History of Present Illness: {hpi}")
    # Discharge part
    if isinstance(dis_content, dict):
        note = dis_content.get('discharge_note', {})
        if isinstance(note, dict):
            diag = note.get('discharge_diagnosis')
            instr = note.get('discharge_instructions')
            if is_valid_text(diag):
                parts.append(f"Discharge Diagnosis: {diag}")
            if is_valid_text(instr):
                parts.append(f"Discharge Instructions: {instr}")
    if not parts:
        return None

    content_with_tag = "[VISIT SUMMARY]\n" + "\n".join(parts)
    return {
        "ref": f"{visit_ref}_VISIT",
        "original_event_id": adm_event.get('event_id', f"{visit_ref}_ADM_UNK"),
        "content": content_with_tag,
        "timestamp": start_time,
        "type": "VISIT"
    }

def generate_jsonl_content(patient_events, patient_id, config=None):
    if config is None: config = TimelineGenConfig()
    window_size = config.SORTING_WINDOW_SIZE
    stride = config.SORTING_STRIDE

    # Group events by visit
    visits_map = {}
    for event in patient_events:
        v_ref = event.get('visit_ref')
        if not v_ref or v_ref == 'V0': continue
        visits_map.setdefault(v_ref, []).append(event)

    # Build list of visit summaries with timestamps
    temp_visits = []
    for v_ref, events in visits_map.items():
        summary = extract_visit_summary(events, v_ref)
        if summary and summary['timestamp']:
            temp_visits.append((v_ref, summary['timestamp'], summary))
    temp_visits.sort(key=lambda x: x[1])

    chronological_items = [item for _, _, item in temp_visits]
    total_visits = len(chronological_items)
    if total_visits < window_size:
        return []

    # Sliding window over visits (same as joint_sorting but window_size applies directly)
    start_indices = list(range(0, total_visits - window_size + 1, stride))
    if not start_indices and total_visits >= window_size:
        start_indices = [0]

    jsonl_lines = []
    id_counter = 0

    for start_idx in start_indices:
        end_idx = start_idx + window_size
        chunk = chronological_items[start_idx:end_idx]
        if len(chunk) < window_size:
            continue

        # Assign real order
        for rank, item in enumerate(chunk):
            item['_real_rank'] = rank

        shuffled = copy.deepcopy(chunk)
        random.shuffle(shuffled)

        options = {}
        display_to_real = {}
        provenance = {}
        for display_idx, item in enumerate(shuffled):
            key = str(display_idx)
            options[key] = item['content']
            display_to_real[display_idx] = item['_real_rank']
            provenance[key] = item.get('original_event_id')

        # Fact
        jsonl_lines.append({
            "type": "fact",
            "id": id_counter,
            "data": options
        })
        id_counter += 1

        # Question
        sorted_pairs = sorted(display_to_real.items(), key=lambda x: x[1])
        correct_sequence = [k for k, _ in sorted_pairs]

        question_text = (
            f"I have provided a dictionary of {window_size} clinical visit summaries (labeled 0 to {window_size-1}), "
            "each representing one hospital visit (including admission and discharge). "
            "Based on clinical progression, sort these keys into the correct chronological order.\n"
            "Output strictly a JSON list of integers representing the sequence of keys.\n"
            f"Example format: [0, 2, 1, ...]"
        )

        jsonl_lines.append({
            "type": "question",
            "id": id_counter,
            "data": question_text,
            "ground_truth": correct_sequence,
            "meta": {
                "options_map": provenance,
                "window_range": f"{start_idx}_to_{end_idx}"
            }
        })
        id_counter += 1

    return jsonl_lines

if __name__ == "__main__":
    cfg = TimelineGenConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=str(cfg.PATIENTS_SEQ_DIR))
    parser.add_argument("--output_dir", type=str, default=str(cfg.VISIT_SORTING_DIR))
    args = parser.parse_args()

    random.seed(cfg.RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.input_dir, "P*_sequenced.json"))
    count = 0
    for fpath in tqdm(files, desc="Generating visit sorting"):
        try:
            pid = os.path.basename(fpath).split('_')[0]
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            lines = generate_jsonl_content(data, pid, cfg)
            if lines:
                out_path = os.path.join(args.output_dir, f"{pid}.jsonl")
                with open(out_path, 'w', encoding='utf-8') as f:
                    for line in lines:
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    print(f"Generated {count} files in {args.output_dir}")