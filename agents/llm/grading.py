import os
import json
import yaml
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau

ROOT_DIR = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT_DIR / "run_llm"
SCORE_DIR = ROOT_DIR / "score_data"
CONFIG_PATH = Path(__file__).resolve().parent / "grading_config.yaml"

def load_config():
    default = {
        "tasks": ["trajectory_sorting", "visit_cloze"],
        "models": None,  # None means all models
        "metric": "kendall_tau"
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        for k in default:
            cfg.setdefault(k, default[k])
        return cfg
    print(f"⚠️ Config not found, using default: {default}")
    return default

def parse_prediction(pred_str):
    if not isinstance(pred_str, str):
        return None
    pred_str = pred_str.strip()
    try:
        return json.loads(pred_str)
    except json.JSONDecodeError:
        try:
            return eval(pred_str)
        except:
            return None

def compute_tau(pred, gt):
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
    scores = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pid = filepath.stem
            sample_id = data.get('id')
            pred_str = data.get('prediction')
            gt = data.get('ground_truth')
            if pred_str is None or gt is None:
                continue
            pred = parse_prediction(pred_str)
            tau = compute_tau(pred, gt)
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
                else:
                    print(f"  No valid scores for {pid}")

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

    print(f"Grading complete. Results in: {SCORE_DIR}")

if __name__ == "__main__":
    main()