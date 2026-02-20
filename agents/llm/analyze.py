#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LongMedBench 数据分析脚本
- 直接使用 score_data 中的 tau 值，确保与 grading 结果一致
- 从 question_data 中读取选项信息，通过样本 ID 合并
- 自动生成全部患者和前50名患者两套图表
- 支持仅统计或仅绘图模式
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 根据实际情况调整
SEQ_DIR = PROJECT_ROOT / "bench_data" / "patients_sequence"
QUESTION_DIR = PROJECT_ROOT / "question_data"
CONTEXT_DIR = PROJECT_ROOT / "context_data"
SCORE_DIR = PROJECT_ROOT / "score_data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis_data"
ANALYSIS_DIR.mkdir(exist_ok=True)

TASKS = ['trajectory_sorting', 'visit_cloze', 'visit_sorting']
MODELS = ['deepseek-v3.2', 'gpt-5-mini', 'deepseek-v3.2-thinking', 'qwen-turbo']

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ========== 从 question_data 读取选项信息 ==========
def load_option_info(task):
    """
    从 question_data 中读取每个样本的选项信息，返回字典 {sample_id: {pid, num_options, option_lengths}}
    """
    task_dir = QUESTION_DIR / task
    if not task_dir.exists():
        return {}
    
    info = {}
    for patient_file in tqdm(list(task_dir.glob("P*.jsonl")), desc=f"Loading {task} option info"):
        pid = patient_file.stem
        with open(patient_file, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f]
        i = 0
        while i < len(lines):
            if lines[i]['type'] == 'fact':
                fact_data = lines[i]['data']
                if i+1 < len(lines) and lines[i+1]['type'] == 'question':
                    q = lines[i+1]
                    sample_id = q['id']
                    # 提取选项字典
                    if task == 'visit_cloze':
                        # cloze 的选项在 question 的 data.options 中
                        if isinstance(q['data'], dict) and 'options' in q['data']:
                            options_dict = q['data']['options']
                        else:
                            options_dict = {}
                    else:
                        # 排序任务的选项在 fact_data 中
                        options_dict = fact_data
                    
                    num_options = len(options_dict)
                    option_lengths = []
                    for opt_val in options_dict.values():
                        # 提取文本长度
                        if isinstance(opt_val, str):
                            text = opt_val
                        elif isinstance(opt_val, dict):
                            text = opt_val.get('content', '')
                            if isinstance(text, dict):
                                text = json.dumps(text, ensure_ascii=False)
                        else:
                            text = str(opt_val)
                        option_lengths.append(len(text))
                    
                    info[sample_id] = {
                        'pid': pid,
                        'num_options': num_options,
                        'option_lengths': option_lengths
                    }
                    i += 2
                else:
                    i += 1
            else:
                i += 1
    return info

# ========== 从 score_data 读取 tau 值 ==========
def load_tau_data(task, model, patient_list=None):
    """
    读取指定任务和模型的评分数据，返回 DataFrame，包含 sample_id, pid, tau。
    如果 patient_list 不为 None，只保留指定患者的样本。
    """
    rows = []
    score_dir = SCORE_DIR / task / model
    if not score_dir.exists():
        return pd.DataFrame()
    
    for patient_file in score_dir.glob("P*.jsonl"):
        pid = patient_file.stem
        if patient_list is not None and pid not in patient_list:
            continue
        with open(patient_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                rows.append({
                    'sample_id': data['id'],
                    'pid': data['pid'],
                    'tau': data['tau']
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['model'] = model
        df['task'] = task
    return df

# ========== 构建完整样本级数据（合并 tau 和选项信息） ==========
def build_samples_data(task, patient_list=None):
    """
    为指定任务构建样本级数据，包含所有有 tau 的样本，并合并选项信息。
    返回 DataFrame，列包括：sample_id, pid, model, tau, num_options, option_lengths (列表字符串)
    """
    all_dfs = []
    option_info = load_option_info(task)  # 提前加载选项信息
    
    for model in MODELS:
        tau_df = load_tau_data(task, model, patient_list)
        if tau_df.empty:
            continue
        
        # 合并选项信息
        tau_df['num_options'] = tau_df['sample_id'].apply(
            lambda sid: option_info.get(sid, {}).get('num_options', np.nan)
        )
        tau_df['option_lengths'] = tau_df['sample_id'].apply(
            lambda sid: json.dumps(option_info.get(sid, {}).get('option_lengths', []))
        )
        all_dfs.append(tau_df)
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # 丢弃没有选项信息的样本（理论上不应该发生，除非问题数据缺失）
        combined = combined.dropna(subset=['num_options'])
        return combined
    else:
        return pd.DataFrame()

# ========== 患者元数据 ==========
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

# ========== 全局 summary 加载 ==========
def load_global_summaries():
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

# ========== 绘图函数 ==========
def plot_model_performance_bar(summary_df, subset_name):
    """模型性能柱状图，使用 summary 数据"""
    if summary_df.empty:
        print(f"No summary data for {subset_name}, skipping plot.")
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(x='task', y='mean_tau', hue='model', data=summary_df,
                capsize=0.1, errwidth=1.5, errcolor='black')
    title = 'Model Performance on LongMedBench Tasks'
    if subset_name == 'first50':
        title += ' (First 50 Patients)'
    plt.title(title)
    plt.ylabel("Kendall's τ")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f'model_performance_{subset_name}.png', dpi=300)
    plt.close()

def plot_option_complexity(df, subset_name):
    """绘制 visit_cloze 选项复杂度分析"""
    df_cloze = df[df['task'] == 'visit_cloze'].copy()
    if df_cloze.empty:
        print(f"No visit_cloze data for {subset_name}, skipping option complexity plot.")
        return
    
    # 解析选项长度列表
    df_cloze['option_lengths'] = df_cloze['option_lengths'].apply(json.loads)
    df_cloze['avg_option_length'] = df_cloze['option_lengths'].apply(np.mean)
    
    # 对选项数进行分箱，避免连续值导致箱线图稀疏
    df_cloze['opt_bin'] = pd.cut(df_cloze['num_options'], bins=range(0, 81, 5), right=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 选项数分布
    sns.histplot(df_cloze['num_options'], bins=20, ax=axes[0,0])
    axes[0,0].set_xlabel('Number of Options')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Option Counts')
    
    # 选项数 vs τ（分箱）
    # 过滤掉无数据的箱
    plot_df = df_cloze.dropna(subset=['opt_bin'])
    if not plot_df.empty:
        sns.boxplot(x='opt_bin', y='tau', hue='model', data=plot_df, ax=axes[0,1])
        axes[0,1].set_xlabel('Number of Options (binned)')
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
    plt.savefig(ANALYSIS_DIR / f'option_complexity_{subset_name}.png', dpi=300)
    plt.close()

def plot_patient_visits_distribution(patient_df, subset_name):
    plt.figure(figsize=(8, 5))
    sns.histplot(patient_df['num_visits'], bins=range(1, patient_df['num_visits'].max()+2),
                 discrete=True)
    plt.xlabel('Number of Visits per Patient')
    plt.ylabel('Number of Patients')
    title = 'Distribution of Patient Visits'
    if subset_name == 'first50':
        title += ' (First 50 Patients)'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f'patient_visits_{subset_name}.png', dpi=300)
    plt.close()

def plot_sorting_difficulty_reduction(df, subset_name):
    """对比 trajectory_sorting 和 visit_sorting 的 τ 分布"""
    sorting_df = df[df['task'].isin(['trajectory_sorting', 'visit_sorting'])].copy()
    if sorting_df.empty:
        print(f"No sorting data for {subset_name}, skipping reduction plot.")
        return
    
    # 移除没有数据的模型（例如 qwen-turbo 可能在 visit_sorting 上无数据）
    models_with_data = sorting_df.groupby('model')['tau'].count()
    models_with_data = models_with_data[models_with_data > 0].index.tolist()
    sorting_df = sorting_df[sorting_df['model'].isin(models_with_data)]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='task', y='tau', hue='model', data=sorting_df)
    plt.xlabel('Task')
    plt.ylabel("Kendall's τ")
    plt.ylim(-1, 1)  # τ 的完整范围
    title = 'Difficulty Reduction: Trajectory Sorting vs. Simplified Version'
    if subset_name == 'first50':
        title += ' (First 50 Patients)'
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f'sorting_reduction_{subset_name}.png', dpi=300)
    plt.close()

# ========== 验证与 summary 的一致性 ==========
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
                print(f"  {task} - {model}: no samples in df")
                continue
            calc_mean = subset['tau'].mean()
            calc_std = subset['tau'].std()
            n_samples = len(subset)
            print(f"  {task} - {model}: summary mean={s_mean:.3f}, calc mean={calc_mean:.3f} (n={n_samples})")
            if abs(calc_mean - s_mean) > 0.01:
                print(f"    ⚠️  WARNING: mean diff={abs(calc_mean - s_mean):.3f}")

# ========== 处理单个子集 ==========
def process_subset(subset_name, patient_list, mode):
    print(f"\n========== Processing {subset_name} subset ==========")
    
    if mode in ['full', 'stats_only']:
        # 收集患者元数据
        patient_meta = collect_patient_metadata(patient_list)
        patient_meta.to_csv(ANALYSIS_DIR / f'patient_metadata_{subset_name}.csv', index=False)
        print(f"Patient metadata saved, {len(patient_meta)} patients.")
        
        # 对于全部患者，统计总样本量（可选）
        if subset_name == 'all':
            # 统计 question_data 中的 fact-question 对数量（仅用于信息）
            print("\nCounting fact-question pairs in question data (all patients):")
            pair_counts = {}
            for task in TASKS:
                # 简单计数，不依赖选项信息
                task_dir = QUESTION_DIR / task
                count = 0
                if task_dir.exists():
                    for f in task_dir.glob("P*.jsonl"):
                        with open(f) as fp:
                            for line in fp:
                                data = json.loads(line)
                                if data.get('type') == 'question':
                                    count += 1
                pair_counts[task] = count
                print(f"{task}: {count} pairs")
            with open(ANALYSIS_DIR / 'fact_question_counts.json', 'w') as f:
                json.dump(pair_counts, f, indent=2)
            
            # context_data 样本量
            context_counts = {}
            for task in TASKS:
                task_dir = CONTEXT_DIR / task
                if not task_dir.exists():
                    context_counts[task] = 0
                else:
                    total = 0
                    for f in task_dir.glob("P*.jsonl"):
                        with open(f) as fp:
                            total += sum(1 for _ in fp)
                    context_counts[task] = total
            print("\nContext data sample counts:")
            for task, cnt in context_counts.items():
                print(f"{task}: {cnt} samples")
            with open(ANALYSIS_DIR / 'context_sample_counts.json', 'w') as f:
                json.dump(context_counts, f, indent=2)
        
        # 构建所有任务的样本级数据
        all_dfs = []
        for task in TASKS:
            print(f"\nBuilding samples for task: {task}")
            df = build_samples_data(task, patient_list)
            if not df.empty:
                all_dfs.append(df)
                df.to_csv(ANALYSIS_DIR / f'{task}_samples_{subset_name}.csv', index=False)
                print(f"Saved {len(df)} samples for {task}")
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv(ANALYSIS_DIR / f'all_samples_{subset_name}.csv', index=False)
            
            # 生成模型性能摘要（用于子集）
            if subset_name == 'first50':
                # 对于前50患者，没有现成 summary，从样本级数据计算
                subset_summary = combined_df.groupby(['task', 'model'])['tau'].agg(['mean', 'std', 'count']).reset_index()
                subset_summary = subset_summary.rename(columns={'mean': 'mean_tau', 'std': 'std_tau'})
                subset_summary.to_csv(ANALYSIS_DIR / f'model_summary_{subset_name}.csv', index=False)
                print(f"Saved subset model summary for {subset_name}")
        else:
            print("No sample data collected.")
        
        if mode == 'stats_only':
            return
    
    # 绘图阶段
    if mode in ['full', 'plot_only']:
        # 加载患者元数据
        patient_meta_file = ANALYSIS_DIR / f'patient_metadata_{subset_name}.csv'
        if not patient_meta_file.exists():
            print(f"Error: Patient metadata file not found: {patient_meta_file}")
            return
        patient_meta = pd.read_csv(patient_meta_file)
        
        # 加载样本级数据
        samples_file = ANALYSIS_DIR / f'all_samples_{subset_name}.csv'
        if not samples_file.exists():
            print(f"Error: Samples file not found: {samples_file}")
            return
        combined_df = pd.read_csv(samples_file)
        
        # 对于全部患者，模型性能图使用全局 summary
        if subset_name == 'all':
            summary_df = load_global_summaries()
            # 确保 summary_df 与 combined_df 中出现的模型一致（可能有些模型无数据）
            models_in_data = combined_df['model'].unique()
            summary_df = summary_df[summary_df['model'].isin(models_in_data)]
        else:
            # 前50患者，使用之前计算的子集 summary
            summary_file = ANALYSIS_DIR / f'model_summary_{subset_name}.csv'
            if not summary_file.exists():
                print(f"Error: Model summary file not found: {summary_file}")
                return
            summary_df = pd.read_csv(summary_file)
        
        # 绘图
        plot_model_performance_bar(summary_df, subset_name)
        plot_option_complexity(combined_df, subset_name)
        plot_patient_visits_distribution(patient_meta, subset_name)
        plot_sorting_difficulty_reduction(combined_df, subset_name)
        
        print(f"Plots for {subset_name} saved.")

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='LongMedBench数据分析')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--plot', action='store_true',
                       help='仅绘图模式（需先运行过统计）')
    group.add_argument('-a', '--analysis', action='store_true',
                       help='仅统计模式（不绘图）')
    args = parser.parse_args()
    
    if args.plot:
        mode = 'plot_only'
    elif args.analysis:
        mode = 'stats_only'
    else:
        mode = 'full'
    
    print(f"Mode: {mode}")
    
    # 获取所有患者ID
    all_patients = sorted([f.stem.split('_')[0] for f in SEQ_DIR.glob("P*_sequenced.json")])
    first50 = all_patients[:50]
    
    # 处理全部患者
    process_subset('all', None, mode)
    
    # 处理前50名患者
    process_subset('first50', first50, mode)
    
    print(f"\nAll done. Results saved to {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()