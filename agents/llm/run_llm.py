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

CONFIG_PATH = Path(__file__).resolve().parent / "run_config.yaml"

def load_config(config_path):
    default = {
        "models": ["QWEN_TURBO"],
        "tasks": ["trajectory_sorting", "visit_cloze"],
        "demo_n": None,
        "max_workers": 10,
        "specific_patients": None,
        "resume_failed": False   # 新增默认值
    }
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        for k in default:
            cfg.setdefault(k, default[k])
        return cfg
    print(f"⚠️ Config file not found: {config_path}. Using default settings.")
    return default

def call_llm_task(task_item, model_config, max_retries=5):
    # 此函数与原始代码完全相同
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=model_config['api_key'],
                base_url=model_config['base_url']
            )
            env_model_id_key = f"{model_config['env_prefix']}_MODEL_ID"
            model_id = os.getenv(env_model_id_key, model_config['default_model_id'])

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
                    "llm_label": model_config['label'],
                    "task_type": task_item['task_type'],
                    "pid": task_item['pid'],
                    "id": task_item['id'],
                    "ground_truth": task_item['ground_truth'],
                    "error": str(e)
                }
    return {
        "status": "error",
        "llm_label": model_config['label'],
        "task_type": task_item['task_type'],
        "pid": task_item['pid'],
        "id": task_item['id'],
        "ground_truth": task_item['ground_truth'],
        "error": "Max retries exceeded due to rate limit."
    }

def run_resume_failed(config):
    """根据失败日志重新运行失败的任务"""
    from collections import defaultdict
    import threading
    import os
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    # ---------- 1. 解析重跑配置 ----------
    resume_cfg = config["resume_failed"]
    if isinstance(resume_cfg, dict):
        enabled = resume_cfg.get("enabled", True)
        model_filter = resume_cfg.get("model_filter")
        task_filter = resume_cfg.get("task_filter")
    else:
        enabled = bool(resume_cfg)
        model_filter = None
        task_filter = None

    if not enabled:
        print("⚠️ resume_failed is not enabled, but function called.")
        return

    # ---------- 2. 读取失败记录 ----------
    failed_dir = os.path.join(ROOT_DIR, "agents", "llm", "failed")
    if not os.path.isdir(failed_dir):
        print(f"❌ Failed directory not found: {failed_dir}")
        return

    failed_files = glob.glob(os.path.join(failed_dir, "*_failed_summary.json"))
    if not failed_files:
        print("❌ No failed summary files found.")
        return

    # 解析失败记录，按 (model_label, task, pid, id) 存储
    failed_items = []  # 每个元素为 (model_label, task, pid, id)
    for ff in failed_files:
        with open(ff, 'r', encoding='utf-8') as f:
            failures = json.load(f)
        # 从文件名提取 model 和 task（例如 gpt-5-mini_visit_cloze_failed_summary.json）
        basename = os.path.basename(ff).replace("_failed_summary.json", "")
        parts = basename.split("_", 1)  # 第一个下划线前是 model_label，后面是 task
        if len(parts) != 2:
            print(f"⚠️ Skipping file with unexpected name: {ff}")
            continue
        model_label, task = parts
        for fail in failures:
            pid = fail.get("pid")
            item_id = fail.get("id")
            if pid and item_id:
                failed_items.append((model_label, task, pid, item_id))

    print(f"📦 Total failed items to retry: {len(failed_items)}")

    # ---------- 3. 构建有效模型配置（复用原逻辑） ----------
    raw_models = config["models"]
    model_list = []
    for m in raw_models:
        if m.endswith("_THINKING"):
            base_name = m[:-9]
            model_list.append({"name": m, "env_name": base_name, "params": {"extra_body": {"enable_thinking": True}}})
        else:
            model_list.append({"name": m, "env_name": m, "params": {}})

    model_configs = {}
    label_to_name = {}
    for m in model_list:
        name = m["name"]
        env_name = m["env_name"]
        params = m["params"]
        api_key = os.getenv(f"{env_name}_API_KEY")
        base_url = os.getenv(f"{env_name}_BASE_URL")
        if not api_key:
            continue
        label = name.lower().replace("_", "-")
        default_model_id = env_name.lower().replace("_", "-")
        model_configs[name] = {
            "name": name,
            "label": label,
            "default_model_id": default_model_id,
            "api_key": api_key,
            "base_url": base_url,
            "params": params,
            "env_prefix": env_name
        }
        label_to_name[label] = name

    # ---------- 4. 构建重跑任务列表 ----------
    tasks_to_run = []
    # 缓存每个 (task, pid) 的原始数据，避免重复读取
    file_cache = {}          # key: (task, pid) -> list of items
    file_cache_lock = threading.Lock()

    for model_label, task, pid, item_id in failed_items:
        # 应用过滤器
        if model_filter and model_label not in [label_to_name.get(m, m) for m in model_filter]:
            continue
        if task_filter and task not in task_filter:
            continue
        model_name = label_to_name.get(model_label)
        if not model_name:
            print(f"⚠️ Unknown model label: {model_label}, skipping.")
            continue

        cache_key = (task, pid)
        with file_cache_lock:
            if cache_key not in file_cache:
                input_path = os.path.join(ROOT_DIR, "context_data", task, f"{pid}.jsonl")
                if not os.path.exists(input_path):
                    print(f"⚠️ Original data file not found: {input_path}, skipping {pid} {item_id}")
                    continue
                items = []
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            items.append(json.loads(line))
                file_cache[cache_key] = items
            else:
                items = file_cache[cache_key]

        # 查找指定 id 的条目
        target_item = next((it for it in items if it["id"] == item_id), None)
        if not target_item:
            print(f"⚠️ Item id {item_id} not found in {task}/{pid}.jsonl, skipping.")
            continue

        tasks_to_run.append({
            "task_type": task,
            "pid": pid,
            "id": item_id,
            "messages": target_item["messages"],
            "ground_truth": target_item["ground_truth"],
            "llm_name": model_name
        })

    print(f"🚀 Retrying {len(tasks_to_run)} tasks after filtering.")

    if not tasks_to_run:
        print("No tasks to run.")
        return

    # ---------- 5. 并发执行，实时更新输出文件 ----------
    file_write_locks = defaultdict(threading.Lock)   # key: full output path
    token_usage = defaultdict(lambda: defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0}))
    new_failures = defaultdict(list)                  # key: (model_label, task)

    run_dir = os.path.join(ROOT_DIR, "run_llm")

    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
        future_map = {
            executor.submit(call_llm_task, t, model_configs[t['llm_name']]): t
            for t in tasks_to_run
        }

        for future in tqdm(as_completed(future_map), total=len(tasks_to_run), desc="Retrying Failed Tasks"):
            res = future.result()
            if res['status'] == 'success':
                # 更新 token 统计
                if res.get('usage'):
                    model = res['llm_label']
                    task = res['task_type']
                    token_usage[model][task]["prompt"] += res['usage']['prompt_tokens']
                    token_usage[model][task]["completion"] += res['usage']['completion_tokens']
                    token_usage[model][task]["total"] += res['usage']['total_tokens']

                # 更新输出文件
                task = res['task_type']
                model_label = res['llm_label']
                pid = res['pid']
                out_dir = os.path.join(run_dir, task, model_label)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{pid}.jsonl")

                new_entry = {
                    "id": res['id'],
                    "prediction": res['prediction'],
                    "ground_truth": res['ground_truth']
                }

                # 加锁，读取现有文件，更新对应 id，再写回
                with file_write_locks[out_path]:
                    existing = {}
                    if os.path.exists(out_path):
                        with open(out_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    item = json.loads(line)
                                    existing[item["id"]] = item
                    existing[res['id']] = new_entry
                    sorted_items = sorted(existing.values(), key=lambda x: x['id'])
                    with open(out_path, 'w', encoding='utf-8') as f:
                        for item in sorted_items:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                # 重跑仍然失败，记录到新失败列表
                model_label = res['llm_label']
                task = res['task_type']
                new_failures[(model_label, task)].append({
                    "pid": res['pid'],
                    "id": res['id'],
                    "error": res['error']
                })

    # ---------- 6. 保存重跑后的 token 使用情况 ----------
    stats_dir = os.path.join(ROOT_DIR, "agents", "llm", "usage")
    os.makedirs(stats_dir, exist_ok=True)
    for model, tasks in token_usage.items():
        for task, tokens in tasks.items():
            safe_model = model.replace("/", "_")
            safe_task = task.replace("/", "_")
            filename = f"{safe_model}_{safe_task}_usage_retry.json"
            filepath = os.path.join(stats_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": model,
                    "task": task,
                    "prompt_tokens": tokens["prompt"],
                    "completion_tokens": tokens["completion"],
                    "total_tokens": tokens["total"]
                }, f, indent=2)
            print(f"📊 Token usage (retry) saved: {filepath}")

    # ---------- 7. 保存新的失败记录 ----------
    retry_failed_dir = os.path.join(ROOT_DIR, "agents", "llm", "failed_retry")
    os.makedirs(retry_failed_dir, exist_ok=True)
    if new_failures:
        for (model_label, task), failures in new_failures.items():
            safe_model = model_label.replace("/", "_")
            safe_task = task.replace("/", "_")
            filename = f"{safe_model}_{safe_task}_failed_retry.json"
            filepath = os.path.join(retry_failed_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(failures, f, indent=2, ensure_ascii=False)
            print(f"❌ New failures during retry: {len(failures)} items saved to {filepath}")
    else:
        print("✅ All failed tasks retried successfully.")

    print(f"\n✅ Retry done. Updated results in '{run_dir}', token stats in '{stats_dir}', new failures in '{retry_failed_dir}'.")

def main():
    config = load_config(CONFIG_PATH)

    # 判断是否进入重跑模式
    if config.get("resume_failed", False):
        run_resume_failed(config)
        return

    # 以下是正常模式（原代码保持不变）
    raw_models = config["models"]
    task_types = config["tasks"]
    demo_n = config["demo_n"]
    max_workers = config["max_workers"]
    specific_patients = config.get("specific_patients")

    # Process model names, add thinking param for _THINKING suffix
    model_list = []
    for m in raw_models:
        if m.endswith("_THINKING"):
            base_name = m[:-9]
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

    # Build valid model configs
    model_configs = {}
    valid_models = []
    for m in model_list:
        name = m["name"]
        env_name = m["env_name"]
        params = m["params"]

        api_key = os.getenv(f"{env_name}_API_KEY")
        base_url = os.getenv(f"{env_name}_BASE_URL")
        if not api_key:
            print(f"⚠️ Skipping {name} (env name: {env_name}): API key missing.")
            continue

        label = name.lower().replace("_", "-")
        default_model_id = env_name.lower().replace("_", "-")

        model_configs[name] = {
            "name": name,
            "label": label,
            "default_model_id": default_model_id,
            "api_key": api_key,
            "base_url": base_url,
            "params": params,
            "env_prefix": env_name
        }
        valid_models.append(name)

    if not valid_models:
        print("❌ No valid models. Exiting.")
        return

    # Build task list
    all_tasks = []
    expected_counts = defaultdict(int)  # key: (task, model_label, pid) -> number of items

    for task in task_types:
        input_dir = os.path.join(ROOT_DIR, "context_data", task)
        if not os.path.exists(input_dir):
            print(f"⚠️ Directory not found: {input_dir}")
            continue

        files = sorted(glob.glob(os.path.join(input_dir, "P*.jsonl")))
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

    # Shared data structures and lock for real-time saving
    results = {}
    token_usage = {}
    completed_counts = defaultdict(int)
    lock = threading.Lock()
    run_dir = os.path.join(ROOT_DIR, "run_llm")

    # Group failed tasks by (model, task)
    failed_by_model_task = defaultdict(list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(call_llm_task, t, model_configs[t['llm_name']]): t
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
                        token_usage.setdefault(model, {}).setdefault(task, {"prompt":0, "completion":0, "total":0})
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

    # Handle any remaining results
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

    # Print token usage summary
    print("\n🔢 Token Usage Summary:")
    for model, tasks in token_usage.items():
        for task, tokens in tasks.items():
            print(f"  {model} | {task}: prompt={tokens['prompt']}, completion={tokens['completion']}, total={tokens['total']}")

    # Save token usage to files
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

    # Save failed tasks logs
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