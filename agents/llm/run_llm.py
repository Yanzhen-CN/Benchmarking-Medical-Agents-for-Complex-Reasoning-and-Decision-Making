import os
import json
import glob
import time
import random
import threading
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# Add root directory to path to import util.hulu
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]          # script in agents/llm/, up two levels to project root
sys.path.insert(0, str(ROOT_DIR))
from util.hulu import TransformersLLMUtil                 # teammate's class

# ------------------------------------------------------------
# User configuration – modify these as needed
# ------------------------------------------------------------
MODEL_PATH = "/data/xzh/Hulu-Med-7B"          # path to your Hulu-Med model
TASKS = ["trajectory_sorting", "visit_cloze"] # tasks to run
DEMO_N = 10                                   # number of patients per task (None for all)
SPECIFIC_PATIENTS = None                       # list of patient IDs, e.g. ["P001", "P002"], None for all
SPECIFIC_ITEMS = None                          # fine‑grained item control, see example below
ENABLE_THINKING = False                        # whether to enable thinking mode
TEMPERATURE = 0.1                              # generation temperature
MAX_TOKENS = 512                               # max new tokens
MAX_WORKERS = 1                                 # threads (local model should use 1)
RESUME_FAILED = False                           # whether to retry previously failed items

# Example of SPECIFIC_ITEMS:
# [
#     {"patient": "P001", "ids": ["id1", "id2"]},   # only these two items for P001
#     {"patient": "P002"}                            # all items for P002
# ]

# ------------------------------------------------------------
# Global model instance (singleton)
# ------------------------------------------------------------
_model_instance = None
_model_lock = threading.Lock()

def get_model():
    """Thread‑safe singleton model loader."""
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                print(f"🚀 Loading Hulu-Med model from {MODEL_PATH}...")
                _model_instance = TransformersLLMUtil(
                    model_name_or_path=MODEL_PATH,
                    dtype="bfloat16",                # adjust if needed
                    attn_implementation="eager",      # use "flash_attention_2" if installed
                    trust_remote_code=True,
                    add_system_prompt=True,
                )
                print("✅ Model loaded.")
    return _model_instance

# ------------------------------------------------------------
# Configuration (kept for structural similarity with run_llm.py)
# ------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "run_config.yaml"   # not actually used

def load_config(config_path):
    """Return a hard‑coded configuration instead of reading from file."""
    # These values override the ones defined above; they are kept for compatibility
    return {
        "models": ["HULU_THINKING" if ENABLE_THINKING else "HULU"],
        "tasks": TASKS,
        "demo_n": DEMO_N,
        "max_workers": MAX_WORKERS,
        "specific_patients": SPECIFIC_PATIENTS,
        "specific_items": SPECIFIC_ITEMS,
        "resume_failed": RESUME_FAILED,
    }

# ------------------------------------------------------------
# Task execution function (local model version)
# ------------------------------------------------------------
def call_hulu_task(task_item, enable_thinking, max_retries=5):
    """
    Call the local Hulu-Med model.
    Returns a result dict with the same structure as in run_llm.py.
    """
    model = get_model()   # get the singleton instance

    for attempt in range(max_retries):
        try:
            # Record token usage before call (cumulative)
            usage_before = model.get_token_usage()
            prompt_before = usage_before["prompt_tokens"]
            completion_before = usage_before["completion_tokens"]

            # Invoke the model
            response = model.chat(
                messages=task_item['messages'],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                enable_thinking=enable_thinking,
            )

            # Record after call
            usage_after = model.get_token_usage()
            delta_usage = {
                "prompt_tokens": usage_after["prompt_tokens"] - prompt_before,
                "completion_tokens": usage_after["completion_tokens"] - completion_before,
                "total_tokens": (usage_after["prompt_tokens"] + usage_after["completion_tokens"]) -
                                (prompt_before + completion_before)
            }

            result = {
                "status": "success",
                "llm_label": "hulu-med",                     # fixed label for output directory
                "task_type": task_item['task_type'],
                "pid": task_item['pid'],
                "id": task_item['id'],
                "prediction": response,
                "ground_truth": task_item['ground_truth'],
                "usage": delta_usage,
            }
            return result

        except Exception as e:
            # Simple exponential backoff
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"⚠️ Error on attempt {attempt+1}/{max_retries}: {e}. Retrying in {wait_time:.2f}s")
            time.sleep(wait_time)

    # All retries exhausted
    return {
        "status": "error",
        "llm_label": "hulu-med",
        "task_type": task_item['task_type'],
        "pid": task_item['pid'],
        "id": task_item['id'],
        "ground_truth": task_item['ground_truth'],
        "error": "Max retries exceeded."
    }

# ------------------------------------------------------------
# Resume failed items (simplified stub, not implemented)
# ------------------------------------------------------------
def run_resume_failed(config):
    """Stub for resume functionality – not implemented in this local version."""
    print("⚠️ resume_failed is not supported in this local script. Exiting.")
    return

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    config = load_config(CONFIG_PATH)          # now returns the hard‑coded configuration

    # ---------- Decide mode (normal or resume) ----------
    if config.get("resume_failed"):
        run_resume_failed(config)
        return

    # ---------- Extract settings ----------
    raw_models = config["models"]               # e.g. ["HULU"] or ["HULU_THINKING"]
    task_types = config["tasks"]
    demo_n = config["demo_n"]
    max_workers = config["max_workers"]
    specific_patients = config.get("specific_patients")
    specific_items = config.get("specific_items")

    # ---------- Build model configuration list ----------
    # We support only one model; determine whether thinking is enabled.
    model_list = []
    for m in raw_models:
        if m.endswith("_THINKING"):
            base_name = m[:-9]
            model_list.append({
                "name": m,
                "env_name": base_name,
                "enable_thinking": True
            })
        else:
            model_list.append({
                "name": m,
                "env_name": m,
                "enable_thinking": False
            })

    # Convert to the format expected by the rest of the code (similar to run_llm.py)
    model_configs = {}
    valid_models = []
    for m in model_list:
        name = m["name"]
        enable_thinking = m["enable_thinking"]

        label = "hulu-med"      # fixed label for output directory
        model_configs[name] = {
            "name": name,
            "label": label,
            "enable_thinking": enable_thinking,
        }
        valid_models.append(name)

    if not valid_models:
        print("❌ No valid models. Exiting.")
        return

    # ---------- Build task list ----------
    all_tasks = []
    expected_counts = defaultdict(int)          # key: (task, model_label, pid) -> number of items

    # Process specific_items if provided
    if specific_items:
        patient_whitelist = {}
        for item in specific_items:
            pid = item.get("patient")
            if not pid:
                print(f"⚠️ Skipping specific_items entry without patient: {item}")
                continue
            ids = item.get("ids")
            if ids is None:
                patient_whitelist[pid] = None          # all items for this patient
            elif isinstance(ids, list):
                patient_whitelist[pid] = set(ids)
            else:
                print(f"⚠️ Invalid ids format for patient {pid}, ignoring ids restriction.")
                patient_whitelist[pid] = None
        target_patients = set(patient_whitelist.keys())
    else:
        patient_whitelist = None
        target_patients = None

    for task in task_types:
        input_dir = os.path.join(ROOT_DIR, "context_data", task)
        if not os.path.exists(input_dir):
            print(f"⚠️ Directory not found: {input_dir}")
            continue

        files = sorted(glob.glob(os.path.join(input_dir, "P*.jsonl")))

        # Filter patient files
        if specific_items:
            files = [f for f in files if os.path.basename(f).replace(".jsonl", "") in target_patients]
        else:
            if specific_patients:
                target_files = []
                for pid in specific_patients:
                    fpath = os.path.join(input_dir, f"{pid}.jsonl")
                    if fpath in files:
                        target_files.append(fpath)
                    else:
                        print(f"⚠️ Patient file not found: {fpath}")
                files = target_files
            elif demo_n is not None:
                files = files[:demo_n]

        for fpath in files:
            pid = os.path.basename(fpath).replace(".jsonl", "")
            with open(fpath, 'r', encoding='utf-8') as f:
                items = [json.loads(line) for line in f if line.strip()]

            # Further filter by specific_items if needed
            if specific_items:
                allowed_ids = patient_whitelist.get(pid)
                if allowed_ids is not None:
                    items = [it for it in items if it["id"] in allowed_ids]

            num_items = len(items)
            for model in valid_models:
                key = (task, model_configs[model]['label'], pid)
                expected_counts[key] = num_items
                for item in items:
                    all_tasks.append({
                        "task_type": task,
                        "pid": pid,
                        "id": item['id'],
                        "messages": item['messages'],
                        "ground_truth": item['ground_truth'],
                        "llm_name": model
                    })

    print(f"🚀 Running local Hulu-Med on {len(valid_models)} model, {len(task_types)} tasks.")
    print(f"📦 Total requests: {len(all_tasks)}")

    if not all_tasks:
        print("No tasks to run.")
        return

    # ---------- Concurrent execution and result saving ----------
    results = {}
    token_usage = defaultdict(lambda: defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0}))
    completed_counts = defaultdict(int)
    lock = threading.Lock()
    run_dir = os.path.join(ROOT_DIR, "run_llm")
    failed_by_model_task = defaultdict(list)

    # Because the local model is not thread‑safe, we use a lock to serialize calls.
    # (MAX_WORKERS can be >1 but actual execution will be sequential.)
    inference_lock = threading.Lock()
    def wrapped_call(task_item):
        with inference_lock:
            return call_hulu_task(
                task_item,
                enable_thinking=model_configs[task_item['llm_name']]['enable_thinking']
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(wrapped_call, t): t
            for t in all_tasks
        }

        for future in tqdm(as_completed(future_map), total=len(all_tasks), desc="Processing Tasks"):
            res = future.result()
            key = (res['task_type'], res['llm_label'], res['pid'])
            data_to_write = None

            with lock:
                if res['status'] == "success":
                    entry = {
                        "id": res['id'],
                        "prediction": res['prediction'],
                        "ground_truth": res['ground_truth']
                    }
                    results.setdefault(key, []).append(entry)
                    completed_counts[key] += 1

                    if res.get('usage'):
                        model = res['llm_label']
                        task = res['task_type']
                        token_usage[model][task]["prompt"] += res['usage']['prompt_tokens']
                        token_usage[model][task]["completion"] += res['usage']['completion_tokens']
                        token_usage[model][task]["total"] += res['usage']['total_tokens']
                else:
                    entry = {
                        "id": res['id'],
                        "prediction": f"ERROR: {res['error']}",
                        "ground_truth": res['ground_truth']
                    }
                    results.setdefault(key, []).append(entry)
                    completed_counts[key] += 1

                    model = res['llm_label']
                    task = res['task_type']
                    failed_by_model_task[(model, task)].append({
                        "pid": res['pid'],
                        "id": res['id'],
                        "error": res['error']
                    })

                if completed_counts[key] == expected_counts[key]:
                    data_to_write = results.pop(key)
                    del completed_counts[key]

            if data_to_write:
                task, model, pid = key
                out_dir = os.path.join(run_dir, task, model)
                os.makedirs(out_dir, exist_ok=True)
                data_to_write.sort(key=lambda x: x['id'])
                out_path = os.path.join(out_dir, f"{pid}.jsonl")
                with open(out_path, 'w', encoding='utf-8') as f:
                    for entry in data_to_write:
                        f.write(json.dumps({
                            "id": entry['id'],
                            "prediction": entry['prediction'],
                            "ground_truth": entry['ground_truth']
                        }, ensure_ascii=False) + "\n")

    # Write any remaining incomplete results (should not happen normally)
    if results:
        print("\n📦 Writing remaining results (incomplete patients)...")
        for (task, model, pid), data in results.items():
            out_dir = os.path.join(run_dir, task, model)
            os.makedirs(out_dir, exist_ok=True)
            data.sort(key=lambda x: x['id'])
            out_path = os.path.join(out_dir, f"{pid}.jsonl")
            with open(out_path, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps({
                        "id": entry['id'],
                        "prediction": entry['prediction'],
                        "ground_truth": entry['ground_truth']
                    }, ensure_ascii=False) + "\n")

    # ---------- Token usage summary ----------
    print("\n🔢 Token Usage Summary:")
    for model, tasks in token_usage.items():
        for task, tokens in tasks.items():
            print(f"  {model} | {task}: prompt={tokens['prompt']}, completion={tokens['completion']}, total={tokens['total']}")

    # Save token usage statistics
    stats_dir = os.path.join(ROOT_DIR, "agents", "llm", "usage")
    os.makedirs(stats_dir, exist_ok=True)
    for model, tasks in token_usage.items():
        for task, tokens in tasks.items():
            safe_model = model.replace("/", "_")
            safe_task = task.replace("/", "_")
            filename = f"{safe_model}_{safe_task}_usage_summary.json"
            filepath = os.path.join(stats_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": model,
                    "task": task,
                    "prompt_tokens": tokens["prompt"],
                    "completion_tokens": tokens["completion"],
                    "total_tokens": tokens["total"]
                }, f, indent=2)
            print(f"📊 Token usage saved: {filepath}")

    # Save failure logs
    failed_dir = os.path.join(ROOT_DIR, "agents", "llm", "failed")
    os.makedirs(failed_dir, exist_ok=True)
    if failed_by_model_task:
        for (model, task), failures in failed_by_model_task.items():
            safe_model = model.replace("/", "_")
            safe_task = task.replace("/", "_")
            filename = f"{safe_model}_{safe_task}_failed_summary.json"
            filepath = os.path.join(failed_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(failures, f, indent=2, ensure_ascii=False)
            print(f"❌ Failed tasks saved: {filepath} (total {len(failures)} failures)")
    else:
        print("✅ No failed tasks.")

    print(f"\n✅ Done. Results in '{run_dir}', token stats in '{stats_dir}', failed logs in '{failed_dir}'.")

if __name__ == "__main__":
    main()