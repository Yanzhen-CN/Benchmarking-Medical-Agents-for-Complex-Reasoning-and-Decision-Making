import os
import json
import glob
import argparse
from tqdm import tqdm

import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 假设你的 Provider 类在这个路径下
from agents.mem0_agent import OpenAICompatibleLLMProvider

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input context_data path (e.g., context_data/trajectory_sorting)")
    parser.add_argument("--task_name", type=str, required=True, help="Task directory name (e.g., sorting or cloze)")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for sub-directory")
    args = parser.parse_args()

    # 初始化 LLM Provider
    llm = OpenAICompatibleLLMProvider()
    
    # 确定模型名称用于创建文件夹
    # 优先使用命令行参数，其次是环境变量，最后默认
    raw_model_name = args.model_name or os.getenv("OPENAI_MODEL_NAME") or "default_model"
    # 替换路径非法字符
    safe_model_name = raw_model_name.replace("/", "_").replace(":", "_")

    # 结果存放在根目录下的 run_llm / task_name / model_name
    output_base = os.path.join("run_llm", args.task_name, safe_model_name)
    os.makedirs(output_base, exist_ok=True)
    
    # 获取目录下所有病人的 context 文件
    files = glob.glob(os.path.join(args.input_dir, "P*.jsonl"))
    
    for fpath in tqdm(files, desc=f"LLM Benchmarking [{safe_model_name}]"):
        pid = os.path.basename(fpath).replace(".jsonl", "")
        results = []
        
        with open(fpath, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if not line.strip():
                    continue
                item = json.loads(line)
                try:
                    # 调用 Provider 发送完整的 Messages 数组
                    response = llm.generate_response(item['messages'])
                    
                    results.append({
                        "id": item['id'],
                        "prediction": response,
                        "ground_truth": item['ground_truth']
                    })
                except Exception as e:
                    print(f"Error processing {pid} item {item.get('id')}: {e}")

        # 按照要求：out_path = os.path.join(output_base, f"{pid}.jsonl")
        if results:
            out_path = os.path.join(output_base, f"{pid}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f_out:
                for res in results:
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()