import os
import json
import glob
import yaml
import dotenv
import time
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Project root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parents[2]
# Load environment variables
dotenv.load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

def load_config(config_path):
    """Load configuration from YAML file; return default if not found."""
    default = {
        "models": ["QWEN_TURBO"],
        "tasks": ["trajectory_sorting", "visit_cloze"],
        "demo_n": None,
        "max_workers": 10
    }
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        for k in default:
            cfg.setdefault(k, default[k])
        return cfg
    print(f"⚠️ Config file not found: {config_path}. Using default settings.")
    return default

def call_llm_task(task_item, model_config, max_retries=3):
    """Execute a single LLM call with retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=model_config['api_key'],
                base_url=model_config['base_url']
            )
            model_id = os.getenv(f"{task_item['llm_name'].upper()}_MODEL_ID", model_config['label'])

            completion = client.chat.completions.create(
                model=model_id,
                messages=task_item['messages'],
                temperature=0.1
            )
            response = completion.choices[0].message.content
            usage = completion.usage

            result = {
                "status": "success",
                "llm_label": model_config['label'],
                "task_type": task_item['task_type'],
                "pid": task_item['pid'],
                "id": task_item['id'],
                "prediction": response,
                "ground_truth": task_item['ground_truth']
            }
            if usage:
                result["usage"] = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            else:
                result["usage"] = None
            return result
        except Exception as e:
            # Check for rate limit error (429)
            status_code = getattr(e, 'status_code', None)
            if status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Rate limit hit for {model_config['label']}, retrying in {wait_time:.2f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # Other errors: return immediately
                return {
                    "status": "error",
                    "pid": task_item['pid'],
                    "llm": model_config['label'],
                    "error": str(e)
                }
    # Retries exhausted
    return {
        "status": "error",
        "pid": task_item['pid'],
        "llm": model_config['label'],
        "error": "Max retries exceeded due to rate limit."
    }

def main():
    # Load configuration
    config = load_config(CONFIG_PATH)
    model_names = config["models"]
    task_types = config["tasks"]
    demo_n = config["demo_n"]
    max_workers = config["max_workers"]

    # Filter models with missing API keys
    model_configs = {}
    valid_models = []
    for name in model_names:
        api_key = os.getenv(f"{name}_API_KEY")
        base_url = os.getenv(f"{name}_BASE_URL")
        if not api_key:
            print(f"⚠️ Skipping {name}: API key missing.")
            continue
        model_configs[name] = {
            "label": name.lower().replace("_", "-"),
            "api_key": api_key,
            "base_url": base_url
        }
        valid_models.append(name)

    if not valid_models:
        print("❌ No valid models. Exiting.")
        return

    # Build task list
    all_tasks = []
    for task in task_types:
        input_dir = os.path.join(ROOT_DIR, "context_data", task)
        if not os.path.exists(input_dir):
            print(f"⚠️ Directory not found: {input_dir}")
            continue

        files = glob.glob(os.path.join(input_dir, "P*.jsonl"))
        if demo_n is not None:
            files = files[:demo_n]

        for fpath in files:
            pid = os.path.basename(fpath).replace(".jsonl", "")
            with open(fpath, 'r', encoding='utf-8') as f:
                items = [json.loads(line) for line in f if line.strip()]
            for item in items:
                for model in valid_models:
                    all_tasks.append({
                        "task_type": task,
                        "pid": pid,
                        "id": item['id'],
                        "messages": item['messages'],
                        "ground_truth": item['ground_truth'],
                        "llm_name": model
                    })

    print(f"🚀 Benchmarking {len(valid_models)} models on {len(task_types)} tasks.")
    print(f"📦 Total requests: {len(all_tasks)}")

    # Execute tasks concurrently
    results = {}
    token_usage = {}  # token_usage[model][task] = {"prompt":..., "completion":..., "total":...}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(call_llm_task, t, model_configs[t['llm_name']]): t
            for t in all_tasks
        }

        for future in tqdm(as_completed(future_map), total=len(all_tasks), desc="Processing Tasks"):
            res = future.result()
            if res['status'] == "success":
                key = (res['task_type'], res['llm_label'], res['pid'])
                results.setdefault(key, []).append(res)

                if res.get('usage'):
                    model = res['llm_label']
                    task = res['task_type']
                    token_usage.setdefault(model, {}).setdefault(task, {"prompt":0, "completion":0, "total":0})
                    token_usage[model][task]["prompt"] += res['usage']['prompt_tokens']
                    token_usage[model][task]["completion"] += res['usage']['completion_tokens']
                    token_usage[model][task]["total"] += res['usage']['total_tokens']
            else:
                print(f"❌ Failed: PID {res.get('pid')} on {res.get('llm')} - {res.get('error')}")

    # Save prediction results
    run_dir = os.path.join(ROOT_DIR, "run_llm")
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

    # Print token usage summary
    print("\n🔢 Token Usage Summary:")
    for model, tasks in token_usage.items():
        for task, tokens in tasks.items():
            print(f"  {model} | {task}: prompt={tokens['prompt']}, completion={tokens['completion']}, total={tokens['total']}")

    # Save token usage to separate files
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

    print(f"\n✅ Done. Results in '{run_dir}', token stats in '{stats_dir}'.")

if __name__ == "__main__":
    main()