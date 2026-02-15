import os
import json
import glob
import yaml
import dotenv
import time
import random
import threading
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

def load_config(config_path):
    """Load configuration from YAML file; return default if not found."""
    default = {
        "models": ["QWEN_TURBO"],
        "tasks": ["trajectory_sorting", "visit_cloze"],
        "demo_n": None,
        "max_workers": 10,
        "specific_patients": None
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
            # 模型 ID 可通过环境变量覆盖，例如 DEEPSEEK_V3_2_MODEL_ID
            env_model_id_key = f"{model_config['env_prefix']}_MODEL_ID"
            model_id = os.getenv(env_model_id_key, model_config['label'])

            completion = client.chat.completions.create(
                model=model_id,
                messages=task_item['messages'],
                temperature=0.1,
                **model_config.get('params', {})
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
            # 检查是否为限流错误 (429)
            status_code = None
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
            elif hasattr(e, 'http_status'):
                status_code = e.http_status

            if status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Rate limit hit for {model_config['label']}, retrying in {wait_time:.2f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                return {
                    "status": "error",
                    "pid": task_item['pid'],
                    "llm": model_config['label'],
                    "error": str(e)
                }
    return {
        "status": "error",
        "pid": task_item['pid'],
        "llm": model_config['label'],
        "error": "Max retries exceeded due to rate limit."
    }

def main():
    config = load_config(CONFIG_PATH)
    raw_models = config["models"]            # 字符串列表
    task_types = config["tasks"]
    demo_n = config["demo_n"]
    max_workers = config["max_workers"]
    specific_patients = config["specific_patients"]

    # 将原始模型名称标准化为配置字典
    model_list = []
    for m in raw_models:
        if m.endswith("_THINKING"):
            base_name = m[:-9]  # 去掉 "_THINKING"
            model_list.append({
                "name": m,
                "env_name": base_name,
                "params": {"extra_body": {"enable_thinking": True}}
            })
        else:
            model_list.append({
                "name": m,
                "env_name": m,
                "params": {}
            })

    # 过滤有效模型并构建配置
    model_configs = {}
    valid_models = []
    for m in model_list:
        name = m["name"]
        env_name = m["env_name"]
        params = m["params"]
        env_prefix = env_name.replace(".", "_")  # 将点号替换为下划线

        api_key = os.getenv(f"{env_prefix}_API_KEY")
        base_url = os.getenv(f"{env_prefix}_BASE_URL")
        if not api_key:
            print(f"⚠️ Skipping {name} (env prefix: {env_prefix}): API key missing.")
            continue

        label = name.lower().replace("_", "-")
        model_configs[name] = {
            "name": name,
            "label": label,
            "api_key": api_key,
            "base_url": base_url,
            "params": params,
            "env_prefix": env_prefix
        }
        valid_models.append(name)

    if not valid_models:
        print("❌ No valid models. Exiting.")
        return

    # 构建任务列表
    all_tasks = []
    expected_counts = defaultdict(int)  # key: (task, model_label, pid) -> 条目数

    for task in task_types:
        input_dir = os.path.join(ROOT_DIR, "context_data", task)
        if not os.path.exists(input_dir):
            print(f"⚠️ Directory not found: {input_dir}")
            continue

        # 获取所有病人文件，并按文件名排序以保证顺序
        files = sorted(glob.glob(os.path.join(input_dir, "P*.jsonl")))

        # 如果指定了特定病人，则只保留这些文件
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

    print(f"🚀 Benchmarking {len(valid_models)} models on {len(task_types)} tasks.")
    print(f"📦 Total requests: {len(all_tasks)}")

    # 共享数据结构和锁（用于实时保存）
    results = {}
    token_usage = {}
    completed_counts = defaultdict(int)
    lock = threading.Lock()
    run_dir = os.path.join(ROOT_DIR, "run_llm")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(call_llm_task, t, model_configs[t['llm_name']]): t
            for t in all_tasks
        }

        for future in tqdm(as_completed(future_map), total=len(all_tasks), desc="Processing Tasks"):
            res = future.result()
            if res['status'] == "success":
                key = (res['task_type'], res['llm_label'], res['pid'])
                data_to_write = None

                with lock:
                    results.setdefault(key, []).append(res)
                    completed_counts[key] += 1

                    if res.get('usage'):
                        model = res['llm_label']
                        task = res['task_type']
                        token_usage.setdefault(model, {}).setdefault(task, {"prompt":0, "completion":0, "total":0})
                        token_usage[model][task]["prompt"] += res['usage']['prompt_tokens']
                        token_usage[model][task]["completion"] += res['usage']['completion_tokens']
                        token_usage[model][task]["total"] += res['usage']['total_tokens']

                    # 如果该病人已完成所有条目，则准备写入
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
            else:
                print(f"❌ Failed: PID {res.get('pid')} on {res.get('llm')} - {res.get('error')}")

    # 处理未完成的结果（可能由于部分任务失败导致）
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

    # 打印并保存 token 统计
    print("\n🔢 Token Usage Summary:")
    for model, tasks in token_usage.items():
        for task, tokens in tasks.items():
            print(f"  {model} | {task}: prompt={tokens['prompt']}, completion={tokens['completion']}, total={tokens['total']}")

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