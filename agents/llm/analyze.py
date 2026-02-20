#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LongMedBench 数据分析脚本（最终版）
- 直接从 score_data 读取 tau，确保与 grading 一致
- 左连接保留所有 tau 样本，仅对需要选项信息的图进行过滤
- 自动生成全部患者和前50名患者两套图表
- 输出验证信息，对比样本级均值与 summary
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

# ========== 路径配置 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

# ========== 从 score_data 读取 tau ==========
def load_tau_from_score(task, model, patient_list=None):
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

# ========== 从 question_data 读取选项信息 ==========
def load_option_info_from_question(task):
    task_dir = QUESTION_DIR / task
    if not task_dir.exists():
        return {}
    info = {}
    for patient_file in tqdm(list(task_dir.glob("P*.jsonl")), desc=f"Loading {task} options"):
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
                        'num_options': num_options,
                        'option_lengths': option_lengths
                    }
                    i += 2
                else:
                    i += 1
            else:
                i += 1
    return info

# ========== 构建完整的样本级数据（左连接保留所有 tau）==========
def build_full_samples(patient_list=None):
    all_dfs = []
    # 预先加载所有任务的选项信息
    option_info_all = {task: load_option_info_from_question(task) for task in TASKS}
    for task in TASKS:
        print(f"\nLoading tau for task: {task}")
        option_info = option_info_all[task]
        for model in MODELS:
            tau_df = load_tau_from_score(task, model, patient_list)
            if tau_df.empty:
                continue
            # 左连接：保留所有 tau 样本，选项信息缺失则设为 NaN
            tau_df['num_options'] = tau_df['sample_id'].map(
                lambda sid: option_info.get(sid, {}).get('num_options', np.nan)
            )
            tau_df['option_lengths'] = tau_df['sample_id'].map(
                lambda sid: json.dumps(option_info.get(sid, {}).get('option_lengths', []))
            )
            all_dfs.append(tau_df)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# ========== 患者元数据 ==========
def load_patient_sequence(pid):
    seq_file = SEQ_DIR / f"{pid}_sequenced.json"
    if not seq_file.exists():
        return None
    with open(seq_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_patient_metadata(pid):
    events = load_patient_sequence(pid)
    if not events:
        return None
    visits = {}
    for ev in events:
        vref = ev.get('visit_ref')
        if vref and vref != 'V0':
            visits.setdefault(vref, []).append(ev)
    num_visits = len(visits)
    events_per_visit = [len(evs) for evs in visits.values()]
    durations = []
    for evs in visits.values():
        adm = next((e for e in evs if e.get('event_type') == 'ADMISSION'), None)
        dis = next((e for e in evs if e.get('event_type') == 'DISCHARGE'), None)
        if adm and dis and adm.get('timestamp') and dis.get('timestamp'):
            try:
                adm_ts = pd.to_datetime(adm['timestamp'])
                dis_ts = pd.to_datetime(dis['timestamp'])
                durations.append((dis_ts - adm_ts).days)
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
    for pf in tqdm(patient_files, desc="Patient metadata"):
        pid = pf.stem.split('_')[0]
        if patient_list is not None and pid not in patient_list:
            continue
        meta = compute_patient_metadata(pid)
        if meta:
            rows.append(meta)
    return pd.DataFrame(rows)

# ========== 加载全局 summary 用于验证 ==========
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

# ========== 验证样本级数据与 summary 的一致性 ==========
def validate_with_summary(samples_df):
    print("\n=== Validation with grading summary ===")
    summary_df = load_global_summaries()
    for task in TASKS:
        for model in MODELS:
            srow = summary_df[(summary_df['task'] == task) & (summary_df['model'] == model)]
            if srow.empty:
                continue
            s_mean = srow.iloc[0]['mean_tau']
            s_std = srow.iloc[0]['std_tau']
            s_total = srow.iloc[0]['total_samples']
            subset = samples_df[(samples_df['task'] == task) & (samples_df['model'] == model)]
            if subset.empty:
                print(f"  {task} - {model}: no samples in samples_df (expected {s_total})")
                continue
            calc_mean = subset['tau'].mean()
            calc_std = subset['tau'].std()
            n_samples = len(subset)
            print(f"  {task} - {model}: summary mean={s_mean:.3f} (n={s_total}), calc mean={calc_mean:.3f} (n={n_samples})")
            if abs(calc_mean - s_mean) > 0.01 or n_samples != s_total:
                print(f"    ⚠️  WARNING: diff mean={abs(calc_mean - s_mean):.3f}, sample count diff={s_total - n_samples}")

# ========== 绘图函数 ==========
def plot_model_performance_bar(samples_df, subset_name):
    grouped = samples_df.groupby(['task', 'model'])['tau'].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped.rename(columns={'mean': 'mean_tau', 'std': 'std_tau'})
    if grouped.empty:
        print(f"No data for {subset_name} model performance plot.")
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(x='task', y='mean_tau', hue='model', data=grouped,
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

def plot_option_complexity(samples_df, subset_name):
    df = samples_df[samples_df['task'] == 'visit_cloze'].copy()
    if df.empty:
        print(f"No visit_cloze data for {subset_name}, skipping option complexity plot.")
        return
    # 过滤掉没有选项信息的样本（理论上应该都有，但以防万一）
    df = df.dropna(subset=['num_options'])
    df['option_lengths'] = df['option_lengths'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    df['avg_option_length'] = df['option_lengths'].apply(lambda x: np.mean(x) if x else np.nan)
    df = df.dropna(subset=['avg_option_length'])
    
    df['opt_bin'] = pd.cut(df['num_options'], bins=range(0, 81, 5), right=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.histplot(df['num_options'], bins=20, ax=axes[0,0])
    axes[0,0].set_xlabel('Number of Options')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Option Counts')
    
    if not df['opt_bin'].isna().all():
        sns.boxplot(x='opt_bin', y='tau', hue='model', data=df, ax=axes[0,1])
        axes[0,1].set_xlabel('Number of Options (binned)')
        axes[0,1].set_ylabel("Kendall's τ")
        axes[0,1].set_title('Performance vs. Option Count')
        axes[0,1].legend_.remove()
    
    sns.histplot(df['avg_option_length'], bins=30, ax=axes[1,0])
    axes[1,0].set_xlabel('Average Option Length (characters)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution of Average Option Length')
    
    sns.scatterplot(x='avg_option_length', y='tau', hue='model', data=df, alpha=0.6, ax=axes[1,1])
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

def plot_sorting_difficulty_reduction(samples_df, subset_name):
    df = samples_df[samples_df['task'].isin(['trajectory_sorting', 'visit_sorting'])].copy()
    if df.empty:
        print(f"No sorting data for {subset_name}, skipping reduction plot.")
        return
    # 移除没有数据的模型（比如 qwen-turbo 可能没有 visit_sorting 样本）
    models_present = df['model'].unique()
    df = df[df['model'].isin(models_present)]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='task', y='tau', hue='model', data=df)
    plt.xlabel('Task')
    plt.ylabel("Kendall's τ")
    plt.ylim(-1, 1)
    title = 'Difficulty Reduction: Trajectory Sorting vs. Simplified Version'
    if subset_name == 'first50':
        title += ' (First 50 Patients)'
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f'sorting_reduction_{subset_name}.png', dpi=300)
    plt.close()

# ========== 处理单个子集 ==========
def process_subset(subset_name, patient_list, mode):
    print(f"\n========== Processing {subset_name} subset ==========")
    
    if mode in ['full', 'stats_only']:
        # 患者元数据
        patient_meta = collect_patient_metadata(patient_list)
        patient_meta.to_csv(ANALYSIS_DIR / f'patient_metadata_{subset_name}.csv', index=False)
        print(f"Patient metadata saved, {len(patient_meta)} patients.")
        
        # 构建样本级数据（包含所有 tau）
        samples_df = build_full_samples(patient_list)
        if not samples_df.empty:
            samples_df.to_csv(ANALYSIS_DIR / f'all_samples_{subset_name}.csv', index=False)
            print(f"Total samples: {len(samples_df)}")
            
            # 验证与 summary 的一致性（仅对全部患者有意义，因为 summary 是全局的）
            if subset_name == 'all':
                validate_with_summary(samples_df)
        else:
            print("No samples collected.")
        
        if mode == 'stats_only':
            return
    
    if mode in ['full', 'plot_only']:
        # 加载数据
        samples_file = ANALYSIS_DIR / f'all_samples_{subset_name}.csv'
        if not samples_file.exists():
            print(f"Error: Samples file not found: {samples_file}")
            return
        samples_df = pd.read_csv(samples_file)
        
        patient_meta_file = ANALYSIS_DIR / f'patient_metadata_{subset_name}.csv'
        if not patient_meta_file.exists():
            print(f"Error: Patient metadata file not found: {patient_meta_file}")
            return
        patient_meta = pd.read_csv(patient_meta_file)
        
        # 绘图
        plot_model_performance_bar(samples_df, subset_name)
        plot_option_complexity(samples_df, subset_name)
        plot_patient_visits_distribution(patient_meta, subset_name)
        plot_sorting_difficulty_reduction(samples_df, subset_name)
        print(f"Plots for {subset_name} saved.")

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='LongMedBench数据分析')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--plot', action='store_true', help='仅绘图模式')
    group.add_argument('-a', '--analysis', action='store_true', help='仅统计模式')
    args = parser.parse_args()
    
    mode = 'full'
    if args.plot:
        mode = 'plot_only'
    elif args.analysis:
        mode = 'stats_only'
    
    print(f"Mode: {mode}")
    
    # 获取所有患者ID
    all_patients = sorted([f.stem.split('_')[0] for f in SEQ_DIR.glob("P*_sequenced.json")])
    first50 = all_patients[:50]
    
    # 处理全部患者
    process_subset('all', None, mode)
    # 处理前50患者
    process_subset('first50', first50, mode)
    
    print(f"\nAll done. Results saved to {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()