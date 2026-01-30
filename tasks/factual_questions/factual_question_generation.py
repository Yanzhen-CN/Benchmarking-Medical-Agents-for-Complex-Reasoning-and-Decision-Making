"""
Medical Agent Benchmark - Factual Question Generator
====================================================

Purpose:
    该脚本旨在自动生成用于测试医疗 AI Agent 长程记忆能力的基准数据集 (Benchmark)。
    它通过分析患者的时序电子病历 (EHR) 数据，生成考察 "精确检索" 和 "模糊检索" 能力的问答对。

Directory Structure:
    - Input : ./bench_data/patients_sequence/*_sequenced.json (原始时序数据)
    - Inter : ./bench_data/patients_summary/*_summary.json (中间生成的骨架摘要)
    - Output: ./bench_data/patients_questions/*_questions.json (最终生成的单个病人问答对)

Usage Examples:
    python factual_question_generation.py test
    python factual_question_generation.py summary --num 5
    python factual_question_generation.py generate --num 5
    python factual_question_generation.py all --num 1
    num 可选 如果不输入生成 
"""

"""
Medical Agent Benchmark - Factual Question Generator (Final Complete Version)
===========================================================================
Log of Changes:
1. [Restored] test_connection: 恢复了完整的 API 测试逻辑，不再是占位符。
2. [Logic] Batch Processing: 按 Visit 隔离，并在 Visit 内部按 BATCH_SIZE 分块。
3. [Logic] Summary: 包含 patient_info, admission, discharge, event 全量骨架。
4. [Logic] Key Facts: 生成问题时增加关键得分点提取。
"""

from pathlib import Path
import openai
import os
import argparse
import json
import math
from collections import defaultdict

# ================= 配置区域 =================
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ROOT = Path(__file__).resolve().parent
PATIENTS_DIR = ROOT / "bench_data" / "patients_sequence"
SUM_DIR = ROOT / "bench_data" / "patients_summary"
QUESTIONS_DIR = ROOT / "bench_data" / "patients_questions"

SUM_DIR.mkdir(parents=True, exist_ok=True)
QUESTIONS_DIR.mkdir(parents=True, exist_ok=True) 

# [参数] 批量处理大小
BATCH_SIZE = 5 

# ================= 工具函数 =================

def get_chat_response(prompt, model="gpt-4o", as_json=False):
    """通用请求函数，带超时和异常处理"""
    try:
        response_format = {"type": "json_object"} if as_json else None
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional medical data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, 
            response_format=response_format,
            timeout=90.0 # Batch 处理文本量大，超时时间设长一点
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_connection():
    """
    [已恢复] 完整的 API 连接测试函数
    """
    import sys
    print("\n>>> [Test Mode] 开始 API 连接测试...")

    # 1. 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 严重错误: 未找到环境变量 OPENAI_API_KEY")
        print("   请确保你已经 export 了 key，或者在代码中 load_dotenv()")
        return
    else:
        print(f"✅ API Key 已检测到: {api_key[:8]}******")

    # 2. 检查 Client 对象
    try:
        if 'client' not in globals() or client is None:
             print("❌ 严重错误: 'client' 变量未定义。")
             return
        print(f"✅ Client 对象状态: Initialized")
    except Exception as e:
        print(f"❌ Client 检查出错: {e}")
        return

    # 3. 发起实际测试请求
    print("\n>>> 正在发送测试请求给 OpenAI (GPT-4o)...")
    test_prompt = "Hello! Please reply with the single word 'Connected' if you receive this."
    result = get_chat_response(test_prompt)
    
    print("-" * 30)
    if result:
        print(f"✅ 测试成功! 模型回复:\n{result}")
    else:
        print(f"❌ 测试失败。请检查上方的 Error 信息。")
    print("-" * 30)

# ================= Task 1: Summary (全量骨架) =================

def get_brief_summary(item_data):
    meta = item_data['metadata']
    section = meta['section']
    content = str(item_data['data'])
    
    if section == 'patient_info':
        instruction = "Summarize demographics (age, gender, race) in 10 words."
    elif 'admission' in section:
        instruction = "Summarize admission reason/type in 10 words."
    elif 'discharge' in section:
        instruction = "Summarize discharge location/disposition in 10 words."
    else:
        instruction = f"Summarize this {meta.get('event_type', 'event')} in 20 words. Focus on key findings."

    prompt = f"{instruction}\nDATA: {content}\nBRIEF SUMMARY:"
    return get_chat_response(prompt)

def run_summary_task(patient_id):
    seq_file = PATIENTS_DIR / f"{patient_id}_sequenced.json"
    sum_file = SUM_DIR / f"{patient_id}_summary.json"
    
    if sum_file.exists(): 
        print(f"Summary exists for {patient_id}, skipping.")
        return
    if not seq_file.exists(): return

    with open(seq_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    summary_list = []
    # 允许所有主要 section 进入骨架
    target_sections = ['patient_info', 'admission_info', 'discharge_info', 'event']
    items_to_process = [i for i in data_list if i['metadata']['section'] in target_sections]
    
    print(f"\n>>> [Summary] Processing {patient_id} ({len(items_to_process)} items)...")

    for i, item in enumerate(items_to_process, 1):
        meta = item['metadata']
        print(f"  ({i}/{len(items_to_process)}) Summarizing {meta['item_id']}...", end="\r")
        
        brief = get_brief_summary(item)
        summary_list.append({
            "item_id": meta['item_id'],
            "visit_id": meta['visit_id'], 
            "timestamp": meta['timestamp'],
            "section": meta['section'],
            "event_type": meta.get('event_type'),
            "brief": brief
        })

    with open(sum_file, 'w', encoding='utf-8') as f:
        json.dump(summary_list, f, indent=2, ensure_ascii=False)
    print(f"\n[完成] Summary saved.")

# ================= Task 2: Generate (Visit-Isolated Batching) =================

def run_generate_task(patient_id):
    seq_file = PATIENTS_DIR / f"{patient_id}_sequenced.json"
    sum_file = SUM_DIR / f"{patient_id}_summary.json"
    out_file = QUESTIONS_DIR / f"{patient_id}_questions.json"
    
    if not seq_file.exists() or not sum_file.exists():
        return []

    with open(seq_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    with open(sum_file, 'r', encoding='utf-8') as f:
        skeleton = json.load(f)

    # 1. 确定目标 Visit (排除最后一个)
    all_visits = sorted(list(set(e['visit_id'] for e in skeleton if e['visit_id'])))
    target_visits = all_visits[:-1] if len(all_visits) > 1 else all_visits
    
    # 2. 按 Visit 分组 Event
    # 结构: { "V1": [event1, event2...], "V2": [...] }
    visit_groups = defaultdict(list)
    
    for e in full_data:
        meta = e['metadata']
        # 只处理属于目标 Visit 且 section 为 event 的数据
        if meta['section'] == 'event' and meta['visit_id'] in target_visits:
            visit_groups[meta['visit_id']].append(e)
            
    qa_pairs = []
    total_visits = len(visit_groups)
    print(f"\n>>> [Generate] Processing {patient_id} ({total_visits} target visits)...")

    # 3. 遍历每个 Visit
    for v_idx, (visit_id, events) in enumerate(visit_groups.items(), 1):
        num_events = len(events)
        print(f"\n  Visit {visit_id} ({num_events} events) - Batching...", end="")
        
        # 4. 在 Visit 内部按 Batch 切分 (逻辑核心)
        for i in range(0, num_events, BATCH_SIZE):
            batch_events = events[i : i + BATCH_SIZE]
            
            print(f".", end="", flush=True) # 进度点

            prompt = f"""
CONTEXT (Patient Timeline):
{json.dumps(skeleton, indent=2)}

TASK:
You are provided with a BATCH of medical events from Visit ID: {visit_id}.
Generate evaluation data for EACH event in the batch.

BATCH DATA ({len(batch_events)} items):
{json.dumps(batch_events, indent=2)}

REQUIREMENTS FOR EACH ITEM:
1. Exact Recall: Question using item_id and timestamp.
2. Loose Recall: Question using ONLY timeline context (NO ID/Time).
3. Answer: Concise factual answer.
4. Key Facts: List of 3-5 mandatory keywords/phrases (entities, numbers, units) from the answer.

OUTPUT FORMAT (JSON Object with "questions" list):
{{
  "questions": [
    {{
      "exact_recall": "...",
      "loose_recall": "...",
      "answer": "...",
      "key_facts": ["..."],
      "evidence": {{ "item_id": "...", "visit_id": "{visit_id}" }}
    }}
  ]
}}
"""
            res_str = get_chat_response(prompt, as_json=True)
            
            if res_str:
                try:
                    res_json = json.loads(res_str)
                    if "questions" in res_json:
                        qa_pairs.extend(res_json["questions"])
                except:
                    print(f"[Warn] JSON Parse Error in {visit_id}")

    if qa_pairs:
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        print(f"\n[完成] Saved {len(qa_pairs)} questions.")
    else:
        print(f"\n[提示] No questions generated.")

    return qa_pairs

# ================= 主程序 =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['summary', 'generate', 'all', 'test'])
    parser.add_argument("--num", type=int, default=None)
    args = parser.parse_args()

    # 模式选择逻辑
    if args.mode == 'test':
        test_connection()
        exit()

    all_seq_files = sorted([f.name for f in PATIENTS_DIR.glob("*_sequenced.json")])
    target_files = all_seq_files[:args.num] if args.num is not None else all_seq_files
    target_ids = [f.split('_')[0] for f in target_files]
    
    print(f"Target patients: {len(target_ids)}")

    for pid in target_ids:
        if args.mode in ['summary', 'all']:
            run_summary_task(pid)
        
        if args.mode in ['generate', 'all']:
            run_generate_task(pid)