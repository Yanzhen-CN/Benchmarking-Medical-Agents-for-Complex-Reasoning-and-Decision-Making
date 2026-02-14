import os
import json
import glob
import dotenv
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# 导入你的 Provider
from agents.mem0_agent import OpenAICompatibleLLMProvider

# 1. 加载环境变量
dotenv.load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

def call_llm_task(task_item, model_config):
    """
    具体的执行单元
    """
    # 在线程内局部设置环境变量，防止 Key 冲突
    # 注意：Provider 初始化必须在环境变量设置之后
    os.environ["OPENAI_API_KEY"] = model_config['api_key']
    os.environ["OPENAI_BASE_URL"] = model_config['base_url']
    
    try:
        # 实例化 Provider
        llm = OpenAICompatibleLLMProvider()
        response = llm.generate_response(task_item['messages'])
        
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
    # 这里的列表元素必须对应 .env 中的前缀
    LLM_LIST = ["QWEN_TURBO", "GPT5_MINI", "DEEPSEEK_V3_2"]
    TASK_LIST = ["trajectory_sorting", "visit_cloze"] 
    # ==========================================

    # 1. 构建模型映射表
    llm_configs = {}
    for name in LLM_LIST:
        llm_configs[name] = {
            "label": name.lower().replace("_", "-"), # 文件夹命名更美观
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
        input_dir = os.path.join("context_data", sub_dir)
        
        files = glob.glob(os.path.join(input_dir, "P*.jsonl"))
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

    # 4. 结果聚合持久化
    for (t_name, m_label, pid), data_list in results_collector.items():
        # 结果存放在 run_llm / task / model / pid.jsonl
        output_dir = os.path.join("run_llm", t_name, m_label)
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

    print(f"\n✅ All tests done. Check 'run_llm/' for results.")

if __name__ == "__main__":
    main()