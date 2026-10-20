import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_all_visualizations(csv_file='all_experiments_results.csv', output_dir='evaluation_results'):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)

    # Style configuration
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 1.0

    # ============================================================
    # Plot 1: Baseline SIA Vulnerability vs Non-IID Skew (Alpha)
    # ============================================================
    df_base = df[df['defense'] == 'none'].copy()
    if not df_base.empty:
        agg_base = df_base.groupby('alpha').agg({
            'attack_acc': ['mean', 'std'],
            'test_acc': ['mean', 'std']
        }).reset_index()
        agg_base.columns = ['alpha', 'attack_mean', 'attack_std', 'test_mean', 'test_std']
        agg_base = agg_base.sort_values('alpha')

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        x = np.arange(len(agg_base))
        width = 0.35

        rects1 = ax.bar(x - width/2, agg_base['attack_mean'], width, yerr=agg_base['attack_std'].fillna(0),
                        label='SIA Attack Success Rate (ASR)', color='#e63946', alpha=0.9, capsize=4)
        rects2 = ax.bar(x + width/2, agg_base['test_mean'], width, yerr=agg_base['test_std'].fillna(0),
                        label='Model Test Accuracy', color='#457b9d', alpha=0.9, capsize=4)

        ax.axhline(10.0, color='#666666', linestyle='--', linewidth=1.5, label='Random Guess Baseline (10%)')

        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Baseline SIA Vulnerability across Non-IID Dirichlet Skew (α)', fontsize=12, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"α = {a}\n(Extreme Non-IID)" if a==0.1 else f"α = {a}\n(Mild Non-IID)" if a==1.0 else f"α = {a}\n(Near IID)" for a in agg_base['alpha']])
        ax.set_ylim(0, 105)
        ax.legend(frameon=True, loc='upper right')

        # Add bar labels
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot1_path = os.path.join(output_dir, 'plot1_baseline_non_iid.png')
        plt.savefig(plot1_path)
        plt.close()
        print(f"Generated: {plot1_path}")

    # ============================================================
    # Plot 2: Differential Privacy Trade-off (Noise vs Accuracy vs ASR)
    # ============================================================
    df_dp = df[df['defense'] == 'dp'].copy()
    if not df_dp.empty:
        agg_dp = df_dp.groupby('dp_noise').agg({
            'attack_acc': 'mean',
            'test_acc': 'mean'
        }).reset_index().sort_values('dp_noise')

        fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)

        color_asr = '#d90429'
        color_acc = '#2b2d42'

        ax1.set_xlabel('DP Gaussian Noise Multiplier (σ)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('SIA Attack Success Rate (%)', color=color_asr, fontsize=11, fontweight='bold')
        line1 = ax1.plot([str(n) for n in agg_dp['dp_noise']], agg_dp['attack_acc'],
                         color=color_asr, marker='o', linewidth=2.5, markersize=7, label='Attack Success Rate (ASR)')
        ax1.tick_params(axis='y', labelcolor=color_asr)
        ax1.axhline(10.0, color='#8d99ae', linestyle='--', linewidth=1.2, label='Random Guess (10%)')
        ax1.set_ylim(0, 105)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Model Test Accuracy (%)', color=color_acc, fontsize=11, fontweight='bold')
        line2 = ax2.plot([str(n) for n in agg_dp['dp_noise']], agg_dp['test_acc'],
                         color='#1d3557', marker='s', linewidth=2.5, markersize=7, linestyle='-.', label='Test Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)
        ax2.set_ylim(0, 105)

        plt.title('Differential Privacy Privacy-Utility Trade-off (α = 0.1)', fontsize=12, fontweight='bold', pad=12)
        fig.tight_layout()
        plot2_path = os.path.join(output_dir, 'plot2_dp_tradeoff.png')
        plt.savefig(plot2_path)
        plt.close()
        print(f"Generated: {plot2_path}")

    # ============================================================
    # Plot 3: Comprehensive Defense Comparison Bar Chart
    # ============================================================
    # Compare Baseline (alpha=0.1), DP (sigma=0.01), SecAgg (alpha=0.1), DP+SecAgg (alpha=0.1)
    comparisons = []
    
    # Baseline
    b = df[(df['defense'] == 'none') & (np.isclose(df['alpha'], 0.1))]
    if not b.empty:
        comparisons.append({
            'System': 'Baseline (No Defense)',
            'Attack_ASR': b['attack_acc'].mean(),
            'Test_Acc': b['test_acc'].mean(),
            'Color': '#e63946'
        })

    # DP (sigma=0.01)
    d = df[(df['defense'] == 'dp') & (np.isclose(df['dp_noise'], 0.01))]
    if not d.empty:
        comparisons.append({
            'System': 'Differential Privacy (σ=0.01)',
            'Attack_ASR': d['attack_acc'].mean(),
            'Test_Acc': d['test_acc'].mean(),
            'Color': '#f4a261'
        })

    # SecAgg
    s = df[df['defense'] == 'secagg']
    if not s.empty:
        comparisons.append({
            'System': 'Secure Aggregation (SecAgg)',
            'Attack_ASR': s['attack_acc'].mean(),
            'Test_Acc': s['test_acc'].mean(),
            'Color': '#2a9d8f'
        })

    # DP + SecAgg
    ds = df[df['defense'] == 'dp_secagg']
    if not ds.empty:
        comparisons.append({
            'System': 'Combined (DP + SecAgg)',
            'Attack_ASR': ds['attack_acc'].mean(),
            'Test_Acc': ds['test_acc'].mean(),
            'Color': '#264653'
        })

    if comparisons:
        df_comp = pd.DataFrame(comparisons)
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
        x = np.arange(len(df_comp))
        w = 0.35

        r1 = ax.bar(x - w/2, df_comp['Attack_ASR'], w, label='Attack Success Rate (Lower is Safer)', color='#e76f51', alpha=0.9)
        r2 = ax.bar(x + w/2, df_comp['Test_Acc'], w, label='Model Test Accuracy (Higher is Better)', color='#1d3557', alpha=0.9)

        ax.axhline(10.0, color='#666666', linestyle='--', linewidth=1.5, label='Random Guess Baseline (10%)')
        ax.set_xticks(x)
        ax.set_xticklabels(df_comp['System'], fontsize=10, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Comparative Defense Benchmark: Privacy Defense vs Model Utility (α = 0.1)', fontsize=12, fontweight='bold', pad=12)
        ax.set_ylim(0, 110)
        ax.legend(frameon=True, loc='upper right')

        for rect in r1:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        for rect in r2:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot3_path = os.path.join(output_dir, 'plot3_defense_comparison.png')
        plt.savefig(plot3_path)
        plt.close()
        print(f"Generated: {plot3_path}")

    # ============================================================
    # Summary Markdown Table Generation
    # ============================================================
    summary_path = os.path.join(output_dir, 'experiments_summary.md')
    summary_md = ["# Master Experimental Results Summary\n"]
    summary_md.append(f"**Random Guessing Baseline**: 10.0% ($K = 10$ clients)\n")
    summary_md.append("| Phase | Defense Mechanism | Configuration | Test Accuracy (%) | SIA Attack Accuracy (%) | Privacy Status |")
    summary_md.append("|---|---|---|---|---|---|")

    for _, row in df.iterrows():
        cfg = f"α={row['alpha']}, Seed={int(row['seed'])}"
        if row['defense'] in ['dp', 'dp_secagg']:
            cfg += f", σ={row['dp_noise']}"
        status = "Vulnerable" if row['attack_acc'] > 40.0 else ("Partially Protected" if row['attack_acc'] > 20.0 else "Fully Protected (~Random Guess)")
        summary_md.append(f"| {row['phase']} | `{row['defense']}` | {cfg} | {row['test_acc']:.2f}% | **{row['attack_acc']:.2f}%** | {status} |")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_md))
    print(f"Generated: {summary_path}")

if __name__ == '__main__':
    generate_all_visualizations()
