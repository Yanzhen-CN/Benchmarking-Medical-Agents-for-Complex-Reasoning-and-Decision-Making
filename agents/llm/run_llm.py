import os
import json
import glob
import dotenv
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# 彻底干掉 mem0，直接用官方轻量级客户端
from openai import OpenAI

# 1. 锁定项目的绝对根目录 (llm 文件夹的上一级)
ROOT_DIR = Path(__file__).resolve().parent.parent
# 加载根目录的环境变量
dotenv.load_dotenv(ROOT_DIR / ".env", override=True)

def call_llm_task(task_item, model_config):
    """
    具体的执行单元：完全不污染全局环境变量，安全并发
    """
    try:
        # 直接把配置喂给 Client，不需要碰 os.environ
        client = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['base_url']
        )
        
        # 允许在 .env 中指定真实的模型名称，如果没有就用 label
        model_id = os.getenv(f"{task_item['llm_name'].upper()}_MODEL_ID", model_config['label'])

        completion = client.chat.completions.create(
            model=model_id,
            messages=task_item['messages'],
            temperature=0.1
        )
        response = completion.choices[0].message.content
        
        return {
            "status": "success",
            "llm_label": model_config['label'],
            "task_type": task_item['task_type'],
            "pid": task_item['pid'],
            "id": task_item['id'],
            "prediction": response,
            "ground_truth": task_item['ground_truth']
        }
    except Exception as e:
        return {
            "status": "error", 
            "pid": task_item['pid'], 
            "llm": model_config['label'], 
            "error": str(e)
        }

def main():
    # ================= 配置区 =================
    LLM_LIST = ["QWEN_TURBO", "GPT5_MINI", "DEEPSEEK_V3_2"]
    LLM_LIST = LLM_LIST[:1]  # 测试时只跑第一个
    TASK_LIST = ["trajectory_sorting", "visit_cloze"] 
    # ==========================================
    DEMO_N = 5 # 全量运行时设为 None

    # 1. 构建模型映射表
    llm_configs = {}
    for name in LLM_LIST:
        llm_configs[name] = {
            "label": name.lower().replace("_", "-"), 
            "api_key": os.getenv(f"{name}_API_KEY"),
            "base_url": os.getenv(f"{name}_BASE_URL")
        }
        if not llm_configs[name]["api_key"]:
            print(f"⚠️ Warning: {name} API Key not found in .env")

    # 2. 扫描并构建任务池
    all_tasks = []
    for task_type in TASK_LIST:
        # 适配文件夹名称
        sub_dir = task_type if "sorting" in task_type else f"visit_{task_type}"
        # 回到根目录找 context_data
        input_dir = os.path.join(ROOT_DIR, "context_data", sub_dir)
        
        if not os.path.exists(input_dir):
            print(f"⚠️ 找不到目录跳过: {input_dir}")
            continue

        files = glob.glob(os.path.join(input_dir, "P*.jsonl"))
        if DEMO_N is not None:
            files = files[:DEMO_N]
            
        for fpath in files:
            pid = os.path.basename(fpath).replace(".jsonl", "")
            with open(fpath, 'r', encoding='utf-8') as f:
                task_lines = [json.loads(line) for line in f if line.strip()]
            
            for item in task_lines:
                for llm_name in LLM_LIST:
                    all_tasks.append({
                        "task_type": task_type,
                        "pid": pid,
                        "id": item['id'],
                        "messages": item['messages'],
                        "ground_truth": item['ground_truth'],
                        "llm_name": llm_name
                    })

    print(f"🚀 Benchmarking {len(LLM_LIST)} models on {len(TASK_LIST)} tasks.")
    print(f"📦 Total Parallel Requests: {len(all_tasks)}")

    # 3. 高并发执行
    results_collector = {}
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
            else:
                print(f"❌ Failed: PID {res.get('pid')} on LLM {res.get('llm')} - {res.get('error')}")

    # 4. 结果聚合持久化 (直接输出到根目录的 run_llm)
    final_run_dir = os.path.join(ROOT_DIR, "run_llm")
    
    for (t_name, m_label, pid), data_list in results_collector.items():
        output_dir = os.path.join(final_run_dir, t_name, m_label)
        os.makedirs(output_dir, exist_ok=True)
        
        # 确保同一个 PID 下的 ID 是顺序的
        data_list.sort(key=lambda x: x['id'])
        
        out_path = os.path.join(output_dir, f"{pid}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for entry in data_list:
                f_out.write(json.dumps({
                    "id": entry['id'],
                    "prediction": entry['prediction'],
                    "ground_truth": entry['ground_truth']
                }, ensure_ascii=False) + "\n")

    print(f"\n✅ All tests done. Check '{final_run_dir}' for results.")

if __name__ == "__main__":
    main()