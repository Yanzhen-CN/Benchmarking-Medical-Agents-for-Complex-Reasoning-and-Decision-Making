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
    if not isinstance(pred_str, str):
        return None
    pred_str = pred_str.strip()
    try:
        return json.loads(pred_str)
    except json.JSONDecodeError:
        pass
    try:
        result = ast.literal_eval(pred_str)
        if isinstance(result, list):
            return result
    except (SyntaxError, ValueError):
        pass
    match = re.search(r'\[.*?\]', pred_str, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group())
        except:
            pass
    return None

def compute_tau(pred, gt):
    if not isinstance(pred, list) or not isinstance(gt, list):
        return None
    if len(pred) != len(gt):
        return None
    try:
        tau, _ = kendalltau(pred, gt)
        return float(tau) if not np.isnan(tau) else None
    except:
        return None

def process_patient_file(filepath, task, model):
    """处理单个患者文件，返回有效分数列表和文件总行数"""
    scores = []
    total_count = 0
    pid = filepath.stem
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            data = json.loads(line)
            sample_id = data.get('id')
            pred_str = data.get('prediction')
            gt = data.get('ground_truth')

            if pred_str is None or gt is None:
                with invalid_lock:
                    invalid_samples.append(f"{task} ({model}): {pid}, {sample_id} [MISSING_DATA]")
                continue

            pred = parse_prediction(pred_str)
            if pred is None:
                with invalid_lock:
                    invalid_samples.append(f"{task} ({model}): {pid}, {sample_id} [PARSE_FAIL]")
                continue

            if len(pred) != len(gt):
                with invalid_lock:
                    invalid_samples.append(f"{task} ({model}): {pid}, {sample_id} [LENGTH_MISMATCH] (gt={len(gt)}, pred={len(pred)})")
                continue

            tau = compute_tau(pred, gt)
            if tau is None:
                with invalid_lock:
                    invalid_samples.append(f"{task} ({model}): {pid}, {sample_id} [TAU_ERROR]")
                continue

            scores.append({
                "pid": pid,
                "id": sample_id,
                "tau": tau,
                "prediction": pred,
                "ground_truth": gt
            })
    return scores, total_count

def save_patient_scores(scores, task, model, pid):
    out_dir = SCORE_DIR / task / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f:
        for s in scores:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')

def compute_model_summary(all_scores, total_samples, task, model):
    valid_samples = len(all_scores)
    taus = [s['tau'] for s in all_scores]
    mean_tau = float(np.mean(taus)) if taus else None
    std_tau = float(np.std(taus)) if taus else None
    summary = {
        "model": model,
        "task": task,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "mean_tau": mean_tau,
        "std_tau": std_tau
    }
    out_dir = SCORE_DIR / task / model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary

def save_task_summary(task, model_summaries):
    task_summary = {
        "task": task,
        "models": model_summaries
    }
    out_dir = SCORE_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(task_summary, f, indent=2)
    return task_summary

def save_global_summary(all_task_summaries):
    """保存所有任务的汇总文件"""
    global_summary = {"tasks": all_task_summaries}
    out_path = SCORE_DIR / "summary.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2)
    print(f"🌍 Global summary saved to: {out_path}")

def save_invalid_samples():
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
    all_task_summaries = []  # 收集每个任务的摘要

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
            total_samples_model = 0
            for patient_file in model_dir.glob("P*.jsonl"):
                pid = patient_file.stem
                scores, file_total = process_patient_file(patient_file, task, model)
                total_samples_model += file_total
                if scores:
                    save_patient_scores(scores, task, model, pid)
                    all_scores.extend(scores)

            if all_scores or total_samples_model > 0:
                model_summary = compute_model_summary(all_scores, total_samples_model, task, model)
                model_summaries.append(model_summary)
                print(f"  {model}: {model_summary['valid_samples']}/{model_summary['total_samples']} valid, mean tau = {model_summary['mean_tau']:.4f}")
            else:
                print(f"  No data for model {model}")

        if model_summaries:
            task_summary = save_task_summary(task, model_summaries)
            all_task_summaries.append(task_summary)
            print(f"Task {task} summary saved.")
        else:
            print(f"No data for task {task}")

    # 保存全局汇总
    if all_task_summaries:
        save_global_summary(all_task_summaries)

    save_invalid_samples()
    print(f"Grading complete. Results in: {SCORE_DIR}")

if __name__ == "__main__":
    main()