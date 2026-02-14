import os
import json
import glob
import dotenv
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Project root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parents[2]
# Load environment variables from the root .env file
dotenv.load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# ================= Configuration for cost estimation =================
# Model price per 1K tokens (in USD). Update as needed.
PRICE_PER_1K = {
    "qwen-turbo": {"prompt": 0.0005, "completion": 0.0015},   # example
    "gpt-5-mini": {"prompt": 0.00015, "completion": 0.0006},
    "deepseek-v3-2": {"prompt": 0.00014, "completion": 0.00028},
    "default": {"prompt": 0.0, "completion": 0.0}
}
# ====================================================================

def calculate_cost(model_label, prompt_tokens, completion_tokens):
    """Estimate cost in USD based on token usage and predefined prices."""
    model_key = model_label.lower()
    price = PRICE_PER_1K.get(model_key, PRICE_PER_1K["default"])
    cost_prompt = (prompt_tokens / 1000) * price["prompt"]
    cost_completion = (completion_tokens / 1000) * price["completion"]
    return cost_prompt + cost_completion

def call_llm_task(task_item, model_config):
    """
    Execute a single LLM call. Returns result with token usage if successful.
    """
    try:
        # Create client directly from config
        client = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['base_url']
        )
        
        # Allow overriding the actual model name via .env (e.g., QWEN_TURBO_MODEL_ID)
        model_id = os.getenv(f"{task_item['llm_name'].upper()}_MODEL_ID", model_config['label'])

        completion = client.chat.completions.create(
            model=model_id,
            messages=task_item['messages'],
            temperature=0.1
        )
        response = completion.choices[0].message.content
        usage = completion.usage  # Extract usage information

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
        return {
            "status": "error", 
            "pid": task_item['pid'], 
            "llm": model_config['label'], 
            "error": str(e)
        }

def main():
    # ================= Configuration =================
    LLM_LIST = ["QWEN_TURBO", "GPT5_MINI", "DEEPSEEK_V3_2"]
    LLM_LIST = LLM_LIST[:1]  # for quick testing, only run the first model
    TASK_LIST = ["trajectory_sorting", "visit_cloze"] 
    DEMO_N = 5               # set to None to process all patients
    # ==================================================

    # Build model configs and filter out those without API key
    llm_configs = {}
    valid_llm_names = []
    for name in LLM_LIST:
        api_key = os.getenv(f"{name}_API_KEY")
        base_url = os.getenv(f"{name}_BASE_URL")
        if not api_key:
            print(f"⚠️ Skipping {name}: API Key not found in environment.")
            continue
        llm_configs[name] = {
            "label": name.lower().replace("_", "-"), 
            "api_key": api_key,
            "base_url": base_url
        }
        valid_llm_names.append(name)

    if not valid_llm_names:
        print("❌ No valid LLM configurations. Exiting.")
        return

    # Scan and build task pool
    all_tasks = []
    for task_type in TASK_LIST:
        # Folder name mapping
        input_dir = os.path.join(ROOT_DIR, "context_data", task_type)
        
        if not os.path.exists(input_dir):
            print(f"⚠️ Directory not found, skipping: {input_dir}")
            continue

        files = glob.glob(os.path.join(input_dir, "P*.jsonl"))
        if DEMO_N is not None:
            files = files[:DEMO_N]
            
        for fpath in files:
            pid = os.path.basename(fpath).replace(".jsonl", "")
            with open(fpath, 'r', encoding='utf-8') as f:
                task_lines = [json.loads(line) for line in f if line.strip()]
            
            for item in task_lines:
                for llm_name in valid_llm_names:   # use only valid models
                    all_tasks.append({
                        "task_type": task_type,
                        "pid": pid,
                        "id": item['id'],
                        "messages": item['messages'],
                        "ground_truth": item['ground_truth'],
                        "llm_name": llm_name
                    })

    print(f"🚀 Benchmarking {len(valid_llm_names)} models on {len(TASK_LIST)} tasks.")
    print(f"📦 Total Parallel Requests: {len(all_tasks)}")

    # Concurrent execution
    results_collector = {}
    token_usage = {}  # key: model_label, value: {"prompt": 0, "completion": 0, "total": 0}
    max_workers = int(os.getenv("MAX_WORKERS", 10))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(call_llm_task, t, llm_configs[t['llm_name']]): t 
            for t in all_tasks
        }
        
        for future in tqdm(as_completed(future_to_task), total=len(all_tasks), desc="Processing Tasks"):
            res = future.result()
            if res['status'] == "success":
                key = (res['task_type'], res['llm_label'], res['pid'])
                if key not in results_collector:
                    results_collector[key] = []
                results_collector[key].append(res)

                # Accumulate token usage
                if res.get('usage'):
                    model = res['llm_label']
                    if model not in token_usage:
                        token_usage[model] = {"prompt": 0, "completion": 0, "total": 0}
                    token_usage[model]["prompt"] += res['usage']['prompt_tokens']
                    token_usage[model]["completion"] += res['usage']['completion_tokens']
                    token_usage[model]["total"] += res['usage']['total_tokens']
            else:
                print(f"❌ Failed: PID {res.get('pid')} on LLM {res.get('llm')} - {res.get('error')}")

    # Save results to run_llm/ under the project root
    final_run_dir = os.path.join(ROOT_DIR, "run_llm")
    
    for (t_name, m_label, pid), data_list in results_collector.items():
        output_dir = os.path.join(final_run_dir, t_name, m_label)
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure entries are sorted by id
        data_list.sort(key=lambda x: x['id'])
        
        out_path = os.path.join(output_dir, f"{pid}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for entry in data_list:
                f_out.write(json.dumps({
                    "id": entry['id'],
                    "prediction": entry['prediction'],
                    "ground_truth": entry['ground_truth']
                }, ensure_ascii=False) + "\n")

    # Print token usage summary and calculate cost
    print("\n💰 Token Usage and Cost Summary:")
    total_cost = 0.0
    for model, tokens in token_usage.items():
        cost = calculate_cost(model, tokens["prompt"], tokens["completion"])
        total_cost += cost
        print(f"  {model}: prompt={tokens['prompt']}, completion={tokens['completion']}, total={tokens['total']}, cost=${cost:.4f}")
    print(f"  Total cost: ${total_cost:.4f}")

    # Save token usage stats to a JSON file
    stats_path = os.path.join(final_run_dir, "token_usage.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            "token_usage": token_usage,
            "total_cost_usd": total_cost
        }, f, indent=2)
    print(f"📊 Token usage stats saved to {stats_path}")

    print(f"\n✅ All tests done. Check '{final_run_dir}' for results.")

if __name__ == "__main__":
    main()