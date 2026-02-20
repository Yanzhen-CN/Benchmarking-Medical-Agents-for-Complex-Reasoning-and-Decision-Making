#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LongMedBench 数据分析脚本
- 支持患者子集分析（全部/前50）
- 支持仅统计或仅绘图模式
- 生成论文所需的统计数据和图表（仅输出PNG）
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')

# ========== 路径配置 ==========
# 注意：如果脚本位于 analysis/ 目录下，建议使用 parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 请根据实际目录结构调整
SEQ_DIR = PROJECT_ROOT / "bench_data" / "patients_sequence"
QUESTION_DIR = PROJECT_ROOT / "question_data"
CONTEXT_DIR = PROJECT_ROOT / "context_data"
SCORE_DIR = PROJECT_ROOT / "score_data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis_data"
ANALYSIS_DIR.mkdir(exist_ok=True)

# 任务和模型定义
TASKS = ['trajectory_sorting', 'visit_cloze', 'visit_sorting']
MODELS = ['deepseek-v3.2', 'gpt-5-mini', 'deepseek-v3.2-thinking', 'qwen-turbo']

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42   # 保留，但不再输出PDF
plt.rcParams['ps.fonttype'] = 42

# ========== 辅助函数 ==========
def extract_text_from_option(option_value):
    if isinstance(option_value, str):
        return option_value
    elif isinstance(option_value, dict):
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
    seq_file = SEQ_DIR / f"{pid}_sequenced.json"
    if not seq_file.exists():
        return None
    with open(seq_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def compute_patient_metadata(pid):
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

def count_fact_question_pairs(task):
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

def count_context_samples():
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

def process_task(task, patient_list=None):
    """
    处理指定任务，收集每个模型每个样本的 τ 值和选项信息，
    返回一个 DataFrame，包含所有样本的统计信息。
    如果 patient_list 不为空，只处理指定患者。
    """
    all_rows = []
    question_task_dir = QUESTION_DIR / task
    if not question_task_dir.exists():
        print(f"Warning: Question directory not found: {question_task_dir}")
        return pd.DataFrame()

    sample_info = {}
    for patient_file in tqdm(list(question_task_dir.glob("P*.jsonl")), desc=f"Loading {task} questions"):
        pid = patient_file.stem
        if patient_list is not None and pid not in patient_list:
            continue
        with open(patient_file, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f]
        i = 0
        while i < len(lines):
            if lines[i]['type'] == 'fact':
                fact_data = lines[i]['data']
                if i+1 < len(lines) and lines[i+1]['type'] == 'question':
                    q = lines[i+1]
                    sample_id = q['id']
                    if task == 'visit_cloze':
                        if isinstance(q['data'], dict) and 'options' in q['data']:
                            options_dict = q['data']['options']
                        else:
                            options_dict = {}
                    else:
                        options_dict = fact_data
                    num_options = len(options_dict)
                    option_lengths = []
                    for opt_val in options_dict.values():
                        text = extract_text_from_option(opt_val)
                        option_lengths.append(len(text))
                    sample_info[sample_id] = {
                        'pid': pid,
                        'num_options': num_options,
                        'option_lengths': option_lengths,
                        'option_lengths_str': json.dumps(option_lengths)
                    }
                    i += 2
                else:
                    i += 1
            else:
                i += 1

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
                # 可选：记录未匹配的样本，但通常不应发生
                pass

    return pd.DataFrame(all_rows)

def collect_patient_metadata(patient_list=None):
    patient_files = list(SEQ_DIR.glob("P*_sequenced.json"))
    rows = []
    for pf in tqdm(patient_files, desc="Collecting patient metadata"):
        pid = pf.stem.split('_')[0]
        if patient_list is not None and pid not in patient_list:
            continue
        meta = compute_patient_metadata(pid)
        if meta:
            rows.append(meta)
    return pd.DataFrame(rows)

def load_summaries():
    """读取所有模型的任务 summary，返回 DataFrame"""
    rows = []
    for task in TASKS:
        for model in MODELS:
            summary_file = SCORE_DIR / task / model / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                rows.append({
                    'task': task,
                    'model': model,
                    'mean_tau': data['mean_tau'],
                    'std_tau': data['std_tau'],
                    'total_samples': data['total_samples'],
                    'valid_samples': data['valid_samples']
                })
    return pd.DataFrame(rows)

# ========== 绘图函数（仅输出PNG） ==========
def plot_model_performance_bar(summary_df, output_suffix=""):
    """使用 summary 数据绘制柱状图（带误差条），仅保存PNG"""
    if summary_df.empty:
        print("No summary data to plot.")
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(x='task', y='mean_tau', hue='model', data=summary_df,
                capsize=0.1, errwidth=1.5, errcolor='black')
    plt.title('Model Performance on LongMedBench Tasks' + (" (First 50 Patients)" if output_suffix else ""))
    plt.ylabel("Kendall's τ")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f'model_performance{output_suffix}.png', dpi=300)
    plt.close()

def plot_option_complexity(df_cloze, output_suffix=""):
    if df_cloze.empty:
        return
    df_cloze['option_lengths'] = df_cloze['option_lengths'].apply(json.loads)
    df_cloze['avg_option_length'] = df_cloze['option_lengths'].apply(np.mean)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(df_cloze['num_options'], bins=20, ax=axes[0,0])
    axes[0,0].set_xlabel('Number of Options')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Option Counts (visit_cloze)')

    sns.boxplot(x='num_options', y='tau', hue='model', data=df_cloze, ax=axes[0,1])
    axes[0,1].set_xlabel('Number of Options')
    axes[0,1].set_ylabel("Kendall's τ")
    axes[0,1].set_title('Performance vs. Option Count')
    axes[0,1].legend_.remove()

    sns.histplot(df_cloze['avg_option_length'], bins=30, ax=axes[1,0])
    axes[1,0].set_xlabel('Average Option Length (characters)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution of Average Option Length')

    sns.scatterplot(x='avg_option_length', y='tau', hue='model', data=df_cloze, alpha=0.6, ax=axes[1,1])
    axes[1,1].set_xlabel('Average Option Length')
    axes[1,1].set_ylabel("Kendall's τ")
    axes[1,1].set_title('Performance vs. Option Length')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f'option_complexity{output_suffix}.png', dpi=300)
    plt.close()

def plot_patient_visits_distribution(patient_df, output_suffix=""):
    plt.figure(figsize=(8, 5))
    sns.histplot(patient_df['num_visits'], bins=range(1, patient_df['num_visits'].max()+2),
                 discrete=True)
    plt.xlabel('Number of Visits per Patient')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Patient Visits' + (" (First 50 Patients)" if output_suffix else ""))
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f'patient_visits{output_suffix}.png', dpi=300)
    plt.close()

def plot_sorting_difficulty_reduction(df, output_suffix=""):
    sorting_df = df[df['task'].isin(['trajectory_sorting', 'visit_sorting'])]
    if sorting_df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='task', y='tau', hue='model', data=sorting_df)
    plt.xlabel('Task')
    plt.ylabel("Kendall's τ")
    plt.title('Difficulty Reduction: Trajectory Sorting vs. Simplified Version' + (" (First 50 Patients)" if output_suffix else ""))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f'sorting_reduction{output_suffix}.png', dpi=300)
    plt.close()

# ========== 与 summary 对比验证样本级数据 ==========
def validate_with_summary(df, summary_df):
    print("\n=== Validation with grading summary ===")
    for task in TASKS:
        for model in MODELS:
            srow = summary_df[(summary_df['task'] == task) & (summary_df['model'] == model)]
            if srow.empty:
                continue
            s_mean = srow.iloc[0]['mean_tau']
            s_std = srow.iloc[0]['std_tau']
            subset = df[(df['task'] == task) & (df['model'] == model)]
            if subset.empty:
                print(f"  {task} - {model}: no samples in df (expected from summary)")
                continue
            calc_mean = subset['tau'].mean()
            calc_std = subset['tau'].std()
            diff_mean = abs(calc_mean - s_mean)
            diff_std = abs(calc_std - s_std)
            if diff_mean > 1e-6 or diff_std > 1e-6:
                print(f"  {task} - {model}: mean diff={diff_mean:.2e}, std diff={diff_std:.2e}")
            else:
                print(f"  {task} - {model}: ✓ matches summary")

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='LongMedBench数据分析')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--plot', action='store_true',
                       help='仅绘图模式（需先运行过统计）')
    group.add_argument('-a', '--analysis', action='store_true',
                       help='仅统计模式（不绘图）')
    parser.add_argument('--subset', type=str, default='all',
                        choices=['all', 'first50'],
                        help='患者子集：all（全部），first50（前50）')
    args = parser.parse_args()

    # 确定运行模式
    if args.plot:
        mode = 'plot_only'
    elif args.analysis:
        mode = 'stats_only'
    else:
        mode = 'full'

    print(f"Mode: {mode}, Subset: {args.subset}")

    # 获取患者列表
    all_patients = sorted([f.stem.split('_')[0] for f in SEQ_DIR.glob("P*_sequenced.json")])
    if args.subset == 'first50':
        patient_list = all_patients[:50]
        print(f"Using first 50 patients: {patient_list[:5]}...")
    else:
        patient_list = None
        print(f"Using all {len(all_patients)} patients.")

    # ===== 统计阶段 =====
    if mode in ['full', 'stats_only']:
        print("\n========== Statistics Phase ==========")

        # 收集患者元数据
        patient_meta = collect_patient_metadata(patient_list)
        patient_meta.to_csv(ANALYSIS_DIR / f'patient_metadata_{args.subset}.csv', index=False)
        print(f"Patient metadata saved, {len(patient_meta)} patients.")

        # 统计问题数据中的 fact-question 对个数（仅对全部患者有意义）
        if args.subset == 'all':
            print("\nCounting fact-question pairs in question data (all patients):")
            pair_counts = {}
            for task in TASKS:
                count = count_fact_question_pairs(task)
                pair_counts[task] = count
                print(f"{task}: {count} pairs")
            with open(ANALYSIS_DIR / 'fact_question_counts.json', 'w') as f:
                json.dump(pair_counts, f, indent=2)

            # 统计测试数据样本量
            context_counts = count_context_samples()
            print("\nContext data sample counts:")
            for task, cnt in context_counts.items():
                print(f"{task}: {cnt} samples")
            with open(ANALYSIS_DIR / 'context_sample_counts.json', 'w') as f:
                json.dump(context_counts, f, indent=2)

        # 处理每个任务，收集有评分的样本
        all_dfs = []
        for task in TASKS:
            print(f"\nProcessing task: {task}")
            df = process_task(task, patient_list)
            if not df.empty:
                all_dfs.append(df)
                df.to_csv(ANALYSIS_DIR / f'{task}_samples_{args.subset}.csv', index=False)
                print(f"Saved {len(df)} samples for {task}")

        # 加载 summary 数据（用于模型性能图）
        summary_df = load_summaries()
        if args.subset == 'all':
            # 保存全局 summary 供后续绘图使用
            summary_df.to_csv(ANALYSIS_DIR / 'model_summary_all.csv', index=False)
        else:
            # 对于子集，我们使用样本级数据计算子集性能并保存
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                subset_summary = combined_df.groupby(['task', 'model'])['tau'].agg(['mean', 'std', 'count']).reset_index()
                subset_summary = subset_summary.rename(columns={'mean': 'mean_tau', 'std': 'std_tau'})
                subset_summary.to_csv(ANALYSIS_DIR / f'model_summary_{args.subset}.csv', index=False)
                print(f"Saved subset model summary for {args.subset}")
            else:
                print("No sample-level data, cannot generate subset summary.")

        # 验证一致性（仅对全部患者）
        if args.subset == 'all' and all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            validate_with_summary(combined_df, summary_df)

        if mode == 'stats_only':
            print("\nStatistics completed. Exiting.")
            return

    # ===== 绘图阶段 =====
    if mode in ['full', 'plot_only']:
        print("\n========== Plotting Phase ==========")

        # 加载所需数据
        patient_meta_file = ANALYSIS_DIR / f'patient_metadata_{args.subset}.csv'
        if not patient_meta_file.exists():
            print(f"Error: Patient metadata file not found: {patient_meta_file}")
            return
        patient_meta = pd.read_csv(patient_meta_file)

        if args.subset == 'all':
            summary_file = ANALYSIS_DIR / 'model_summary_all.csv'
        else:
            summary_file = ANALYSIS_DIR / f'model_summary_{args.subset}.csv'
        if not summary_file.exists():
            print(f"Error: Model summary file not found: {summary_file}")
            return
        summary_df = pd.read_csv(summary_file)

        all_dfs = []
        for task in TASKS:
            sample_file = ANALYSIS_DIR / f'{task}_samples_{args.subset}.csv'
            if sample_file.exists():
                df = pd.read_csv(sample_file)
                all_dfs.append(df)
            else:
                print(f"Warning: Sample file not found: {sample_file}")
        if not all_dfs:
            print("No sample-level data found for plotting additional figures.")
        else:
            combined_df = pd.concat(all_dfs, ignore_index=True)

        # 绘制模型性能柱状图（仅PNG）
        plot_model_performance_bar(summary_df, f"_{args.subset}")

        # 绘制其他图表（需要样本级数据）
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            cloze_df = combined_df[combined_df['task'] == 'visit_cloze']
            plot_option_complexity(cloze_df, f"_{args.subset}")
            plot_patient_visits_distribution(patient_meta, f"_{args.subset}")
            plot_sorting_difficulty_reduction(combined_df, f"_{args.subset}")
        else:
            print("Skipping other plots due to missing sample data.")

        print(f"\nAll plots saved as PNG to {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()