#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LongMedBench 数据分析脚本（最终版）
- 直接从 score_data 读取 tau，确保与 grading 一致
- 模型性能、选项复杂度、难度对比图均基于全量样本
- 患者住院次数分布图分别基于全量和前50患者绘制
- 输出验证信息，确保样本数与 summary 一致
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
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings('ignore')

# ========== 路径配置 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = PROJECT_ROOT / "bench_data" / "patients_sequence"
QUESTION_DIR = PROJECT_ROOT / "question_data"
CONTEXT_DIR = PROJECT_ROOT / "context_data"
SCORE_DIR = PROJECT_ROOT / "score_data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis_data"
ANALYSIS_DIR.mkdir(exist_ok=True)

TASKS = ['joint_sorting', 'visit_cloze', 'visit_sorting']
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
    option_info_all = {task: load_option_info_from_question(task) for task in TASKS}
    for task in TASKS:
        print(f"\nLoading tau for task: {task}")
        option_info = option_info_all[task]
        for model in MODELS:
            tau_df = load_tau_from_score(task, model, patient_list)
            if tau_df.empty:
                continue
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
            s_total = srow.iloc[0]['total_samples']
            subset = samples_df[(samples_df['task'] == task) & (samples_df['model'] == model)]
            if subset.empty:
                print(f"  {task} - {model}: no samples in samples_df (expected {s_total})")
                continue
            calc_mean = subset['tau'].mean()
            n_samples = len(subset)
            print(f"  {task} - {model}: summary mean={s_mean:.3f} (n={s_total}), calc mean={calc_mean:.3f} (n={n_samples})")
            if abs(calc_mean - s_mean) > 0.01 or n_samples != s_total:
                print(f"    ⚠️  WARNING: diff mean={abs(calc_mean - s_mean):.3f}, sample count diff={s_total - n_samples}")

# ========== 绘图函数（全量样本） ==========
def plot_model_performance_bar(samples_df):
    """基于全量样本绘制模型性能柱状图"""
    grouped = samples_df.groupby(['task', 'model'])['tau'].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped.rename(columns={'mean': 'mean_tau', 'std': 'std_tau'})
    if grouped.empty:
        print("No data for model performance plot.")
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(x='task', y='mean_tau', hue='model', data=grouped,
                capsize=0.1, errwidth=1.5, errcolor='black')
    plt.title('Model Performance on LongMedBench Tasks')
    plt.ylabel("Kendall's τ")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'model_performance_all.png', dpi=300)
    plt.close()

def plot_option_complexity(samples_df):
    """绘制选项数对性能的影响：分箱折线图"""
    df = samples_df[samples_df['task'] == 'visit_cloze'].copy()
    if df.empty:
        print("No visit_cloze data, skipping option complexity plot.")
        return
    
    df = df.dropna(subset=['num_options'])
    
    # 对选项数进行分箱（每5个一组）
    bins = range(0, 81, 5)  # 0-5,5-10,...,75-80
    labels = [f"{i}-{i+4}" for i in bins[:-1]]
    df['opt_group'] = pd.cut(df['num_options'], bins=bins, labels=labels, right=False)
    
    # 计算每个选项组内各模型的平均τ
    grouped = df.groupby(['opt_group', 'model'])['tau'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='opt_group', y='tau', hue='model', data=grouped, marker='o')
    plt.xlabel('Number of Options')
    plt.ylabel("Mean Kendall's τ")
    plt.title('Model Performance vs. Number of Options (visit_cloze)')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', frameon=True)  # 放在右上角内侧，带边框
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'option_complexity_all.png', dpi=300)
    plt.close()

def plot_sorting_difficulty_reduction(samples_df, subset_name=None):
    df = samples_df[samples_df['task'].isin(['joint_sorting', 'visit_sorting'])].copy()
    if df.empty:
        return
    
    plt.figure(figsize=(10, 6))
    # 小提琴图（半透明）
    sns.violinplot(x='task', y='tau', hue='model', data=df,
                   inner=None, linewidth=1, palette='Set2', alpha=0.5)
    # 散点（使用 stripplot 避免 swarmplot 的避让计算，适合较多数据）
    sns.stripplot(x='task', y='tau', hue='model', data=df,
                  dodge=True, size=2, palette='Set2', alpha=0.7, jitter=0.2)
    
    # 去重图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.5, frameon=True)
    
    plt.xlabel('Task')
    plt.ylabel("Kendall's τ")
    plt.ylim(-1, 1)
    title = 'Difficulty Reduction: Trajectory Sorting vs. Simplified Version'
    if subset_name == 'first50':
        title += ' (First 50 Patients)'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f'sorting_reduction_{subset_name or "all"}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_patient_visits_distribution(patient_df, suffix):
    """根据患者数据绘制住院次数分布图，suffix 为 '_all' 或 '_first50'"""
    plt.figure(figsize=(8, 5))
    sns.histplot(patient_df['num_visits'], bins=range(1, patient_df['num_visits'].max()+2),
                 discrete=True)
    plt.xlabel('Number of Visits per Patient')
    plt.ylabel('Number of Patients')
    title = 'Distribution of Patient Visits'
    if suffix == '_first50':
        title += ' (First 50 Patients)'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f'patient_visits{suffix}.png', dpi=300)
    plt.close()

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
    
    # ===== 统计阶段（生成两个子集的样本数据）=====
    if mode in ['full', 'stats_only']:
        # 全量患者
        print("\n--- Processing all patients (for metadata) ---")
        patient_meta_all = collect_patient_metadata(None)
        patient_meta_all.to_csv(ANALYSIS_DIR / 'patient_metadata_all.csv', index=False)
        print(f"All patients metadata saved, {len(patient_meta_all)} patients.")
        
        # 前50患者
        print("\n--- Processing first 50 patients (for metadata) ---")
        patient_meta_50 = collect_patient_metadata(first50)
        patient_meta_50.to_csv(ANALYSIS_DIR / 'patient_metadata_first50.csv', index=False)
        print(f"First 50 patients metadata saved, {len(patient_meta_50)} patients.")
        
        # 构建全量样本数据（用于所有分析图）
        print("\n--- Building full samples (all patients) ---")
        samples_all = build_full_samples(None)
        if not samples_all.empty:
            samples_all.to_csv(ANALYSIS_DIR / 'all_samples_all.csv', index=False)
            print(f"Total full samples: {len(samples_all)}")
            validate_with_summary(samples_all)
        else:
            print("No full samples collected.")
        
        # 前50患者样本数据（仅用于验证，不用于主要绘图）
        print("\n--- Building first50 samples (for verification) ---")
        samples_50 = build_full_samples(first50)
        if not samples_50.empty:
            samples_50.to_csv(ANALYSIS_DIR / 'all_samples_first50.csv', index=False)
            print(f"Total first50 samples: {len(samples_50)}")
        else:
            print("No first50 samples collected.")
        
        if mode == 'stats_only':
            print("\nStatistics completed. Exiting.")
            return
    
    # ===== 绘图阶段 =====
    if mode in ['full', 'plot_only']:
        samples_all_file = ANALYSIS_DIR / 'all_samples_all.csv'
        if not samples_all_file.exists():
            print(f"Error: Full samples file not found: {samples_all_file}")
            return
        samples_all = pd.read_csv(samples_all_file)
        
        # plot_model_performance_bar(samples_all)
        # plot_option_complexity(samples_all)
        plot_sorting_difficulty_reduction(samples_all)
        
        # 绘制患者住院次数分布图
        patient_meta_all_file = ANALYSIS_DIR / 'patient_metadata_all.csv'
        patient_meta_50_file = ANALYSIS_DIR / 'patient_metadata_first50.csv'
        if patient_meta_all_file.exists():
            patient_all = pd.read_csv(patient_meta_all_file)
            plot_patient_visits_distribution(patient_all, '_all')
        if patient_meta_50_file.exists():
            patient_50 = pd.read_csv(patient_meta_50_file)
            plot_patient_visits_distribution(patient_50, '_first50')
        
        print(f"\nAll plots saved to {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()