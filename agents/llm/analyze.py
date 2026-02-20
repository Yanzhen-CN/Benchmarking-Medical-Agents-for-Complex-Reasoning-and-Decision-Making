#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LongMedBench 数据分析脚本
从原始序列数据、问题数据、测试数据和评分结果中提取统计信息，
生成可视化图表，保存到 analysis-data 文件夹。

依赖：pandas, matplotlib, seaborn, numpy, tqdm
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ========== 路径配置 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 假设脚本在 analysis/ 目录下
SEQ_DIR = PROJECT_ROOT / "bench_data" / "patients_sequence"          # 原始序列数据
QUESTION_DIR = PROJECT_ROOT / "question_data"                        # 问题数据
CONTEXT_DIR = PROJECT_ROOT / "context_data"                          # 测试数据（用于LLM的上下文）
SCORE_DIR = PROJECT_ROOT / "score_data"                              # 评分结果
ANALYSIS_DIR = PROJECT_ROOT / "analysis_data"                        # 输出目录
ANALYSIS_DIR.mkdir(exist_ok=True)

# 任务和模型定义
TASKS = ['trajectory_sorting', 'visit_cloze', 'visit_sorting']
MODELS = ['deepseek-v3.2', 'gpt-5-mini', 'deepseek-v3.2-thinking', 'qwen-turbo']

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ========== 辅助函数 ==========
def extract_text_from_option(option_value):
    """
    从选项值中提取可读文本，用于计算长度。
    - 对于排序任务，option_value 是字符串。
    - 对于 visit_cloze，option_value 是事件对象，包含 content 字段。
    """
    if isinstance(option_value, str):
        return option_value
    elif isinstance(option_value, dict):
        # 优先使用 content 字段
        content = option_value.get('content', '')
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        else:
            return str(content)
    else:
        return str(option_value)

def load_patient_sequence(pid):
    """加载患者的原始事件序列"""
    seq_file = SEQ_DIR / f"{pid}_sequenced.json"
    if not seq_file.exists():
        return None
    with open(seq_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def compute_patient_metadata(pid):
    """计算患者的住院次数、每个visit的事件数、平均住院日等"""
    events = load_patient_sequence(pid)
    if not events:
        return None
    visits = defaultdict(list)
    for ev in events:
        vref = ev.get('visit_ref')
        if vref and vref != 'V0':
            visits[vref].append(ev)
    num_visits = len(visits)
    events_per_visit = [len(evs) for evs in visits.values()]
    # 计算平均住院日（如果有admission和discharge）
    durations = []
    for vref, evs in visits.items():
        adm = next((e for e in evs if e.get('event_type') == 'ADMISSION'), None)
        dis = next((e for e in evs if e.get('event_type') == 'DISCHARGE'), None)
        if adm and dis and adm.get('timestamp') and dis.get('timestamp'):
            try:
                adm_ts = pd.to_datetime(adm['timestamp'])
                dis_ts = pd.to_datetime(dis['timestamp'])
                duration = (dis_ts - adm_ts).days
                durations.append(duration)
            except:
                pass
    return {
        'pid': pid,
        'num_visits': num_visits,
        'total_events': sum(events_per_visit),
        'events_per_visit_mean': np.mean(events_per_visit) if events_per_visit else 0,
        'visit_duration_mean': np.mean(durations) if durations else None
    }

def load_scores(task, model):
    """加载指定任务和模型的评分结果，返回 dict {sample_id: tau}"""
    score_dir = SCORE_DIR / task / model
    scores = {}
    if not score_dir.exists():
        return scores
    for patient_file in score_dir.glob("P*.jsonl"):
        with open(patient_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                scores[data['id']] = data['tau']
    return scores

# ========== 统计问题数据中的fact-question对 ==========
def count_fact_question_pairs(task):
    """从问题数据中统计每个患者的fact-question对数量"""
    task_dir = QUESTION_DIR / task
    if not task_dir.exists():
        return 0
    total_pairs = 0
    for patient_file in task_dir.glob("P*.jsonl"):
        with open(patient_file, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f]
        i = 0
        while i < len(lines):
            if lines[i]['type'] == 'fact':
                if i+1 < len(lines) and lines[i+1]['type'] == 'question':
                    total_pairs += 1
                    i += 2
                else:
                    i += 1
            else:
                i += 1
    return total_pairs

# ========== 处理每个任务，收集样本级信息 ==========
def process_task(task):
    """
    处理指定任务，收集每个模型每个样本的 τ 值和选项信息，
    返回一个 DataFrame，包含所有样本的统计信息。
    """
    all_rows = []
    # 首先，从问题文件中构建 sample_id -> 选项信息的映射
    question_task_dir = QUESTION_DIR / task
    if not question_task_dir.exists():
        print(f"Warning: Question directory not found: {question_task_dir}")
        return pd.DataFrame()

    sample_info = {}  # sample_id -> {'pid', 'num_options', 'option_lengths'}
    for patient_file in tqdm(list(question_task_dir.glob("P*.jsonl")), desc=f"Loading {task} questions"):
        pid = patient_file.stem
        with open(patient_file, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f]
        # 按顺序解析：fact 后紧跟 question
        i = 0
        while i < len(lines):
            if lines[i]['type'] == 'fact':
                fact_data = lines[i]['data']  # 选项字典
                if i+1 < len(lines) and lines[i+1]['type'] == 'question':
                    q = lines[i+1]
                    sample_id = q['id']
                    # 提取选项信息
                    if task == 'visit_cloze':
                        # 对于 cloze，选项在 question 的 data.options 中
                        if isinstance(q['data'], dict) and 'options' in q['data']:
                            options_dict = q['data']['options']
                        else:
                            options_dict = {}
                    else:
                        # 对于排序任务，fact_data 就是选项字典
                        options_dict = fact_data
                    # 计算选项数和选项长度
                    num_options = len(options_dict)
                    option_lengths = []
                    for opt_key, opt_val in options_dict.items():
                        text = extract_text_from_option(opt_val)
                        option_lengths.append(len(text))
                    sample_info[sample_id] = {
                        'pid': pid,
                        'num_options': num_options,
                        'option_lengths': option_lengths,
                        'option_lengths_str': json.dumps(option_lengths)
                    }
                    i += 2  # 跳过question
                else:
                    i += 1
            else:
                i += 1

    # 对于每个模型，加载评分并合并
    for model in MODELS:
        scores = load_scores(task, model)
        if not scores:
            continue
        for sample_id, tau in scores.items():
            if sample_id in sample_info:
                info = sample_info[sample_id]
                row = {
                    'sample_id': sample_id,
                    'pid': info['pid'],
                    'model': model,
                    'task': task,
                    'tau': tau,
                    'num_options': info['num_options'],
                    'option_lengths': info['option_lengths_str'],
                }
                all_rows.append(row)
            else:
                # 如果评分样本在问题文件中找不到，可能是问题文件缺失，记录警告
                print(f"Warning: sample_id {sample_id} not found in question data for {task}")

    return pd.DataFrame(all_rows)

# ========== 患者元数据 ==========
def collect_patient_metadata():
    """收集所有患者的住院次数等元数据"""
    patient_files = list(SEQ_DIR.glob("P*_sequenced.json"))
    rows = []
    for pf in tqdm(patient_files, desc="Collecting patient metadata"):
        pid = pf.stem.split('_')[0]
        meta = compute_patient_metadata(pid)
        if meta:
            rows.append(meta)
    return pd.DataFrame(rows)

# ========== 统计测试数据（context_data）中的样本量 ==========
def count_context_samples():
    """统计 context_data 中各任务下的样本数（即模型输入文件的行数）"""
    counts = {}
    for task in TASKS:
        task_dir = CONTEXT_DIR / task
        if not task_dir.exists():
            counts[task] = 0
            continue
        total = 0
        for patient_file in task_dir.glob("P*.jsonl"):
            with open(patient_file, 'r', encoding='utf-8') as f:
                total += sum(1 for _ in f)
        counts[task] = total
    return counts

# ========== 绘图函数 ==========
def plot_model_performance(df):
    """绘制各任务下模型的平均τ（带误差条）"""
    grouped = df.groupby(['task', 'model'])['tau'].agg(['mean', 'std', 'count']).reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='task', y='mean', hue='model', data=grouped, capsize=0.1,
                errwidth=1.5, errcolor='black')
    plt.title('Model Performance on LongMedBench Tasks')
    plt.ylabel("Kendall's τ")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'model_performance.pdf')
    plt.savefig(ANALYSIS_DIR / 'model_performance.png', dpi=300)
    plt.close()

def plot_option_complexity(df_cloze):
    """绘制选项数分布和选项数 vs τ 的关系（仅针对 visit_cloze）"""
    if df_cloze.empty:
        return
    # 解析 option_lengths 字符串为列表
    df_cloze['option_lengths'] = df_cloze['option_lengths'].apply(json.loads)
    df_cloze['avg_option_length'] = df_cloze['option_lengths'].apply(np.mean)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 选项数分布
    sns.histplot(df_cloze['num_options'], bins=20, ax=axes[0,0])
    axes[0,0].set_xlabel('Number of Options')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Option Counts (visit_cloze)')

    # 选项数 vs τ（按模型）
    sns.boxplot(x='num_options', y='tau', hue='model', data=df_cloze, ax=axes[0,1])
    axes[0,1].set_xlabel('Number of Options')
    axes[0,1].set_ylabel("Kendall's τ")
    axes[0,1].set_title('Performance vs. Option Count')
    axes[0,1].legend_.remove()

    # 选项平均长度分布
    sns.histplot(df_cloze['avg_option_length'], bins=30, ax=axes[1,0])
    axes[1,0].set_xlabel('Average Option Length (characters)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution of Average Option Length')

    # 选项平均长度 vs τ
    sns.scatterplot(x='avg_option_length', y='tau', hue='model', data=df_cloze, alpha=0.6, ax=axes[1,1])
    axes[1,1].set_xlabel('Average Option Length')
    axes[1,1].set_ylabel("Kendall's τ")
    axes[1,1].set_title('Performance vs. Option Length')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'option_complexity.pdf')
    plt.savefig(ANALYSIS_DIR / 'option_complexity.png', dpi=300)
    plt.close()

def plot_patient_visits_distribution(patient_df):
    """绘制患者住院次数分布"""
    plt.figure(figsize=(8, 5))
    sns.histplot(patient_df['num_visits'], bins=range(1, patient_df['num_visits'].max()+2),
                 discrete=True)
    plt.xlabel('Number of Visits per Patient')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Patient Visits')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'patient_visits.pdf')
    plt.savefig(ANALYSIS_DIR / 'patient_visits.png', dpi=300)
    plt.close()

def plot_sorting_difficulty_reduction(df):
    """对比 trajectory_sorting 和 visit_sorting 的τ"""
    sorting_df = df[df['task'].isin(['trajectory_sorting', 'visit_sorting'])]
    if sorting_df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='task', y='tau', hue='model', data=sorting_df)
    plt.xlabel('Task')
    plt.ylabel("Kendall's τ")
    plt.title('Difficulty Reduction: Trajectory Sorting vs. Simplified Version')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'sorting_reduction.pdf')
    plt.savefig(ANALYSIS_DIR / 'sorting_reduction.png', dpi=300)
    plt.close()

# ========== 主函数 ==========
def main():
    print("Starting analysis...")

    # 收集患者元数据
    patient_meta = collect_patient_metadata()
    patient_meta.to_csv(ANALYSIS_DIR / 'patient_metadata.csv', index=False)
    print(f"Patient metadata saved, {len(patient_meta)} patients.")

    # 统计问题数据中的 fact-question 对个数
    print("\nCounting fact-question pairs in question data:")
    pair_counts = {}
    for task in TASKS:
        count = count_fact_question_pairs(task)
        pair_counts[task] = count
        print(f"{task}: {count} pairs")
    with open(ANALYSIS_DIR / 'fact_question_counts.json', 'w') as f:
        json.dump(pair_counts, f, indent=2)

    # 统计测试数据（context_data）中的样本量
    context_counts = count_context_samples()
    print("\nContext data sample counts:")
    for task, cnt in context_counts.items():
        print(f"{task}: {cnt} samples")
    with open(ANALYSIS_DIR / 'context_sample_counts.json', 'w') as f:
        json.dump(context_counts, f, indent=2)

    # 处理每个任务，收集样本级数据
    all_dfs = []
    for task in TASKS:
        print(f"\nProcessing task: {task}")
        df = process_task(task)
        if not df.empty:
            all_dfs.append(df)
            # 保存每个任务的详细数据
            df.to_csv(ANALYSIS_DIR / f'{task}_samples.csv', index=False)
            print(f"Saved {len(df)} samples for {task}")

    if not all_dfs:
        print("No performance data collected. Exiting.")
        return

    # 合并所有任务的数据
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 绘制模型性能图
    plot_model_performance(combined_df)

    # 绘制选项复杂度图（仅针对 visit_cloze）
    cloze_df = combined_df[combined_df['task'] == 'visit_cloze']
    plot_option_complexity(cloze_df)

    # 绘制患者住院次数分布
    plot_patient_visits_distribution(patient_meta)

    # 绘制排序难度对比
    plot_sorting_difficulty_reduction(combined_df)

    print(f"\nAll analysis results saved to {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()