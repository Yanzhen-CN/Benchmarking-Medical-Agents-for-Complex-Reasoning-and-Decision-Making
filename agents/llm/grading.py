import os
import json
import yaml
import ast
import re
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau
from collections import defaultdict
import threading

ROOT_DIR = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT_DIR / "run_llm"
SCORE_DIR = ROOT_DIR / "score_data"
CONFIG_PATH = Path(__file__).resolve().parent / "grading_config.yaml"

# Global list to collect invalid samples (protected by lock)
invalid_samples = []
invalid_lock = threading.Lock()

def load_config():
    default = {
        "tasks": ["trajectory_sorting", "visit_cloze"],
        "models": None,
        "metric": "kendall_tau"
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'rb') as f:
            cfg = yaml.safe_load(f)
        for k in default:
            cfg.setdefault(k, default[k])
        return cfg
    print(f"⚠️ Config not found, using default: {default}")
    return default

def parse_prediction(pred_str):
    """Robust parsing of model prediction into list of ints."""
    if not isinstance(pred_str, str):
        return None
    pred_str = pred_str.strip()
    # Try JSON
    try:
        return json.loads(pred_str)
    except json.JSONDecodeError:
        pass
    # Try ast.literal_eval
    try:
        result = ast.literal_eval(pred_str)
        if isinstance(result, list):
            return result
    except (SyntaxError, ValueError):
        pass
    # Try to extract first list-like pattern
    match = re.search(r'\[.*?\]', pred_str, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group())
        except:
            pass
    return None

def compute_tau(pred, gt):
    """Compute Kendall's tau, return None if invalid."""
    if not isinstance(pred, list) or not isinstance(gt, list):
        return None
    if len(pred) != len(gt):
        return None
    try:
        tau, _ = kendalltau(pred, gt)
        return float(tau)
    except:
        return None

def process_patient_file(filepath, task, model):
    """Process one patient JSONL file, return list of valid scores, and record invalid ones."""
    scores = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pid = filepath.stem
            sample_id = data.get('id')
            pred_str = data.get('prediction')
            gt = data.get('ground_truth')
            if pred_str is None or gt is None:
                # Missing data – record as invalid
                with invalid_lock:
                    invalid_samples.append(f"{task}: {pid}, {sample_id}")
                continue

            pred = parse_prediction(pred_str)
            if pred is None:
                with invalid_lock:
                    invalid_samples.append(f"{task}: {pid}, {sample_id}")
                continue

            if len(pred) != len(gt):
                with invalid_lock:
                    invalid_samples.append(f"{task}: {pid}, {sample_id}")
                continue

            tau = compute_tau(pred, gt)
            if tau is None:
                # This could happen if elements are not comparable (e.g., non-integers)
                with invalid_lock:
                    invalid_samples.append(f"{task}: {pid}, {sample_id}")
                continue

            scores.append({
                "pid": pid,
                "id": sample_id,
                "tau": tau,
                "prediction": pred,
                "ground_truth": gt
            })
    return scores

def save_patient_scores(scores, task, model, pid):
    out_dir = SCORE_DIR / task / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f:
        for s in scores:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')

def compute_model_summary(all_scores, task, model):
    taus = [s['tau'] for s in all_scores if s['tau'] is not None]
    n_valid = len(taus)
    n_total = len(all_scores)
    mean_tau = float(np.mean(taus)) if n_valid > 0 else None
    std_tau = float(np.std(taus)) if n_valid > 0 else None
    summary = {
        "model": model,
        "task": task,
        "total_samples": n_total,
        "valid_samples": n_valid,
        "mean_tau": mean_tau,
        "std_tau": std_tau
    }
    out_dir = SCORE_DIR / task / model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary

def save_task_summary(task, model_summaries):
    global_summary = {
        "task": task,
        "models": model_summaries
    }
    out_dir = SCORE_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2)

def save_invalid_samples():
    """Write all collected invalid samples to a file."""
    if not invalid_samples:
        return
    invalid_file = SCORE_DIR / "invalid_samples.txt"
    with open(invalid_file, 'w', encoding='utf-8') as f:
        for line in invalid_samples:
            f.write(line + '\n')
    print(f"📄 Invalid samples saved to: {invalid_file} (total {len(invalid_samples)})")

def main():
    config = load_config()
    tasks = config["tasks"]
    allowed_models = config["models"]

    print("Starting grading...")
    for task in tasks:
        task_run_dir = RUN_DIR / task
        if not task_run_dir.exists():
            print(f"Warning: run directory for task {task} not found.")
            continue

        model_summaries = []
        for model_dir in task_run_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if allowed_models and model not in allowed_models:
                print(f"Skipping model {model} (not in config)")
                continue
            print(f"Processing model: {model}, task: {task}")

            all_scores = []
            for patient_file in model_dir.glob("P*.jsonl"):
                pid = patient_file.stem
                scores = process_patient_file(patient_file, task, model)
                if scores:
                    save_patient_scores(scores, task, model, pid)
                    all_scores.extend(scores)
                # else: no valid scores for this patient (already recorded as invalid)

            if all_scores:
                model_summary = compute_model_summary(all_scores, task, model)
                model_summaries.append(model_summary)
                print(f"  {model}: {model_summary['valid_samples']}/{model_summary['total_samples']} valid, mean tau = {model_summary['mean_tau']:.4f}")
            else:
                print(f"  No scores for model {model}")

        if model_summaries:
            save_task_summary(task, model_summaries)
            print(f"Task {task} summary saved.")
        else:
            print(f"No data for task {task}")

    save_invalid_samples()
    print(f"Grading complete. Results in: {SCORE_DIR}")

if __name__ == "__main__":
    main()