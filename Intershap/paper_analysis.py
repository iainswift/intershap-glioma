"""
InterSHAP Paper Analysis
========================
Multi-seed robustness analysis, publication figures, and statistical tests.
"""
import sys
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Configuration
SEEDS = [42, 123, 2024]
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "MyData" / "results" / "intershap"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})


def run_fusion_analysis(fusion_type, seed):
    """Run InterSHAP analysis for a specific fusion type and seed."""
    save_dir = RESULTS_DIR / fusion_type / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if fusion_type == "early":
        script_path = BASE_DIR / "3_EarlyFusion" / "apply_intershap.py"
        checkpoint = BASE_DIR / "MyData" / "results" / "early_fusion" / "checkpoints" / "models" / "earlyfusion_model_survcox_fold0" / "model_dict_best.pt"
        data_path = BASE_DIR / "MyData" / "splits" / "early_test.csv"
        cmd = [
            "python", str(script_path),
            "--seed", str(seed),
            "--save_dir", str(save_dir),
            "--checkpoint", str(checkpoint),
            "--data_path", str(data_path)
        ]
    elif fusion_type == "late":
        # Use DIRECT method (analyzes actual R Cox Lasso, not proxy)
        script_path = BASE_DIR / "4_LateFusion" / "apply_intershap_direct.py"
        train_path = BASE_DIR / "MyData" / "results" / "late_fusion" / "scores_late_train.csv"
        test_path = BASE_DIR / "MyData" / "results" / "late_fusion" / "scores_late_test.csv"
        cmd = [
            "python", str(script_path),
            "--seed", str(seed),
            "--save_dir", str(save_dir),
            "--train_path", str(train_path),
            "--test_path", str(test_path)
        ]
    elif fusion_type == "joint":
        script_path = BASE_DIR / "5_JointFusion" / "apply_intershap.py"
        checkpoint = BASE_DIR / "MyData" / "results" / "joint_fusion" / "checkpoints" / "models" / "jointfusion_model_survcox_fold0" / "model_dict_best.pt"
        data_path = BASE_DIR / "MyData" / "splits" / "joint_test.csv"
        patch_path = BASE_DIR / "MyData" / "patches"
        cmd = [
            "python", str(script_path),
            "--seed", str(seed),
            "--save_dir", str(save_dir),
            "--checkpoint", str(checkpoint),
            "--data_path", str(data_path),
            "--patch_path", str(patch_path),
            "--use_cached"  # Use cached embeddings for speed
        ]
    
    print(f"  Running {fusion_type} fusion with seed {seed}...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    
    if result.returncode != 0:
        print(f"    Warning: {fusion_type} seed {seed} failed")
        print(f"    {result.stderr[:500]}")
        return None
    
    # Load results
    summary_path = save_dir / "global_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return None


def run_multi_seed_analysis():
    """Run all fusion types across multiple seeds."""
    print("=" * 60)
    print("MULTI-SEED ROBUSTNESS ANALYSIS")
    print("=" * 60)
    
    all_results = []
    
    for fusion_type in ["early", "late", "joint"]:
        print(f"\n{fusion_type.upper()} FUSION:")
        for seed in SEEDS:
            result = run_fusion_analysis(fusion_type, seed)
            if result is not None:
                result['Seed'] = seed
                result['Fusion'] = fusion_type.capitalize()
                all_results.append(result)
                print(f"    Seed {seed}: InterSHAP={result['InterSHAP'].values[0]:.2f}%")
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(RESULTS_DIR / "multi_seed_results.csv", index=False)
        return combined
    return None


def compute_statistics(results_df):
    """Compute statistical summaries and significance tests."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    
    stats_summary = []
    
    for fusion in results_df['Fusion'].unique():
        subset = results_df[results_df['Fusion'] == fusion]
        
        intershap_vals = subset['InterSHAP'].values
        wsi_vals = subset['WSI_Contribution'].values
        rna_vals = subset['RNA_Contribution'].values
        
        stats_summary.append({
            'Fusion': fusion,
            'InterSHAP_Mean': np.mean(intershap_vals),
            'InterSHAP_Std': np.std(intershap_vals),
            'InterSHAP_CI95_Low': np.mean(intershap_vals) - 1.96 * np.std(intershap_vals) / np.sqrt(len(intershap_vals)),
            'InterSHAP_CI95_High': np.mean(intershap_vals) + 1.96 * np.std(intershap_vals) / np.sqrt(len(intershap_vals)),
            'WSI_Mean': np.mean(wsi_vals),
            'WSI_Std': np.std(wsi_vals),
            'RNA_Mean': np.mean(rna_vals),
            'RNA_Std': np.std(rna_vals),
            'N_Seeds': len(intershap_vals)
        })
    
    stats_df = pd.DataFrame(stats_summary)
    
    # Pairwise comparisons (if we have variance)
    print("\n--- Summary Statistics ---")
    print(stats_df.to_string(index=False))
    
    # Statistical tests between fusion types
    print("\n--- Pairwise Significance Tests (Mann-Whitney U) ---")
    
    fusions = results_df['Fusion'].unique()
    pvalues = []
    
    for i, f1 in enumerate(fusions):
        for f2 in fusions[i+1:]:
            vals1 = results_df[results_df['Fusion'] == f1]['InterSHAP'].values
            vals2 = results_df[results_df['Fusion'] == f2]['InterSHAP'].values
            
            if len(vals1) > 1 and len(vals2) > 1:
                # Mann-Whitney U test (non-parametric)
                stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                # Also do t-test for comparison
                t_stat, t_pval = stats.ttest_ind(vals1, vals2)
                
                pvalues.append({
                    'Comparison': f"{f1} vs {f2}",
                    'Mann-Whitney_U': stat,
                    'MW_pvalue': pval,
                    'T_statistic': t_stat,
                    'T_pvalue': t_pval,
                    'Significant_0.05': 'Yes' if pval < 0.05 else 'No'
                })
                print(f"  {f1} vs {f2}: U={stat:.2f}, p={pval:.4f} {'*' if pval < 0.05 else ''}")
    
    pval_df = pd.DataFrame(pvalues) if pvalues else None
    
    # Test if InterSHAP > 0 for each fusion (one-sample t-test)
    print("\n--- One-Sample Tests (InterSHAP > 0) ---")
    for fusion in fusions:
        vals = results_df[results_df['Fusion'] == fusion]['InterSHAP'].values
        if len(vals) > 1 and np.std(vals) > 0:
            t_stat, pval = stats.ttest_1samp(vals, 0)
            pval_onesided = pval / 2 if t_stat > 0 else 1 - pval / 2
            print(f"  {fusion}: mean={np.mean(vals):.2f}%, t={t_stat:.2f}, p(one-sided)={pval_onesided:.4f} {'*' if pval_onesided < 0.05 else ''}")
        else:
            print(f"  {fusion}: mean={np.mean(vals):.2f}% (insufficient variance for test)")
    
    # Save statistics
    stats_df.to_csv(RESULTS_DIR / "statistical_summary.csv", index=False)
    if pval_df is not None:
        pval_df.to_csv(RESULTS_DIR / "pairwise_tests.csv", index=False)
    
    return stats_df, pval_df


def create_figures(results_df, stats_df):
    """Create publication-ready figures."""
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    
    # Figure 1: Bar chart with error bars
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    fusion_order = ['Early', 'Late', 'Joint']
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    x = np.arange(len(fusion_order))
    width = 0.6
    
    means = [stats_df[stats_df['Fusion'] == f]['InterSHAP_Mean'].values[0] for f in fusion_order]
    stds = [stats_df[stats_df['Fusion'] == f]['InterSHAP_Std'].values[0] for f in fusion_order]
    
    bars = ax1.bar(x, means, width, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('InterSHAP Score (%)', fontweight='bold')
    ax1.set_xlabel('Fusion Strategy', fontweight='bold')
    ax1.set_title('Cross-Modal Interaction (InterSHAP) by Fusion Strategy', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fusion_order)
    ax1.set_ylim(0, max(means) * 1.4 + 5)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax1.annotate(f'{mean:.1f}±{std:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + std),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig1.savefig(FIGURES_DIR / "fig1_intershap_comparison.png")
    fig1.savefig(FIGURES_DIR / "fig1_intershap_comparison.pdf")
    print(f"  Saved: fig1_intershap_comparison.png/pdf")
    
    # Figure 2: Stacked bar chart - Contribution breakdown
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    wsi_means = [stats_df[stats_df['Fusion'] == f]['WSI_Mean'].values[0] for f in fusion_order]
    rna_means = [stats_df[stats_df['Fusion'] == f]['RNA_Mean'].values[0] for f in fusion_order]
    inter_means = means  # InterSHAP
    
    ax2.bar(x, wsi_means, width, label='WSI (Histopathology)', color='#e74c3c', edgecolor='black')
    ax2.bar(x, rna_means, width, bottom=wsi_means, label='RNA (Gene Expression)', color='#f39c12', edgecolor='black')
    ax2.bar(x, inter_means, width, bottom=[w+r for w,r in zip(wsi_means, rna_means)], 
            label='Interaction (InterSHAP)', color='#1abc9c', edgecolor='black')
    
    ax2.set_ylabel('Contribution (%)', fontweight='bold')
    ax2.set_xlabel('Fusion Strategy', fontweight='bold')
    ax2.set_title('Modality Attribution Breakdown', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(fusion_order)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 115)
    
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "fig2_contribution_breakdown.png")
    fig2.savefig(FIGURES_DIR / "fig2_contribution_breakdown.pdf")
    print(f"  Saved: fig2_contribution_breakdown.png/pdf")
    
    # Figure 3: Box plot with individual points
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    # Prepare data for boxplot
    plot_data = []
    for fusion in fusion_order:
        subset = results_df[results_df['Fusion'] == fusion]['InterSHAP'].values
        for val in subset:
            plot_data.append({'Fusion': fusion, 'InterSHAP': val})
    plot_df = pd.DataFrame(plot_data)
    
    bp = ax3.boxplot([results_df[results_df['Fusion'] == f]['InterSHAP'].values for f in fusion_order],
                     labels=fusion_order, patch_artist=True, widths=0.5)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, fusion in enumerate(fusion_order):
        vals = results_df[results_df['Fusion'] == fusion]['InterSHAP'].values
        jitter = np.random.normal(0, 0.04, len(vals))
        ax3.scatter([i+1 + j for j in jitter], vals, color='black', s=50, zorder=5, alpha=0.8)
    
    ax3.set_ylabel('InterSHAP Score (%)', fontweight='bold')
    ax3.set_xlabel('Fusion Strategy', fontweight='bold')
    ax3.set_title('InterSHAP Distribution Across Seeds', fontweight='bold')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / "fig3_intershap_boxplot.png")
    fig3.savefig(FIGURES_DIR / "fig3_intershap_boxplot.pdf")
    print(f"  Saved: fig3_intershap_boxplot.png/pdf")
    
    # Figure 4: Risk group analysis heatmap
    fig4, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    risk_labels = ['High Risk', 'Medium-High', 'Medium-Low', 'Low Risk']
    
    for idx, fusion in enumerate(fusion_order):
        # Load risk group data from first seed
        risk_path = RESULTS_DIR / fusion.lower() / "risk_group_analysis.csv"
        if not risk_path.exists():
            risk_path = RESULTS_DIR / fusion.lower() / "seed_42" / "risk_group_analysis.csv"
        
        if risk_path.exists():
            risk_df = pd.read_csv(risk_path)
            
            # Create heatmap data
            if 'Mean_InterSHAP' in risk_df.columns:
                heatmap_data = risk_df[['Mean_InterSHAP', 'WSI_Mean', 'RNA_Mean']].values
            elif 'InterSHAP_Mean' in risk_df.columns:
                heatmap_data = risk_df[['InterSHAP_Mean', 'WSI_Mean', 'RNA_Mean']].values
            else:
                continue
            
            im = axes[idx].imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=60)
            
            axes[idx].set_xticks([0, 1, 2])
            axes[idx].set_xticklabels(['InterSHAP', 'WSI', 'RNA'], rotation=45)
            axes[idx].set_yticks(range(len(risk_labels)))
            axes[idx].set_yticklabels(risk_labels)
            axes[idx].set_title(f'{fusion} Fusion', fontweight='bold')
            
            # Add text annotations
            for i in range(heatmap_data.shape[0]):
                for j in range(heatmap_data.shape[1]):
                    val = heatmap_data[i, j]
                    color = 'white' if val > 30 else 'black'
                    axes[idx].text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=10)
    
    plt.colorbar(im, ax=axes, label='Contribution (%)', shrink=0.8)
    fig4.suptitle('Modality Contributions by Risk Group', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig4.savefig(FIGURES_DIR / "fig4_risk_group_heatmap.png")
    fig4.savefig(FIGURES_DIR / "fig4_risk_group_heatmap.pdf")
    print(f"  Saved: fig4_risk_group_heatmap.png/pdf")
    
    plt.close('all')
    return True


def generate_latex_table(stats_df):
    """Generate LaTeX table for paper."""
    print("\n" + "=" * 60)
    print("LATEX TABLE")
    print("=" * 60)
    
    latex = r"""
\begin{table}[h]
\centering
\caption{InterSHAP Analysis Results Across Fusion Strategies}
\label{tab:intershap_results}
\begin{tabular}{lccc}
\toprule
\textbf{Fusion Strategy} & \textbf{InterSHAP (\%)} & \textbf{WSI (\%)} & \textbf{RNA (\%)} \\
\midrule
"""
    for _, row in stats_df.iterrows():
        latex += f"{row['Fusion']} & {row['InterSHAP_Mean']:.2f} $\\pm$ {row['InterSHAP_Std']:.2f} & "
        latex += f"{row['WSI_Mean']:.2f} $\\pm$ {row['WSI_Std']:.2f} & "
        latex += f"{row['RNA_Mean']:.2f} $\\pm$ {row['RNA_Std']:.2f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item InterSHAP: Shapley Interaction Index measuring cross-modal synergy.
\item WSI: Histopathology whole-slide image contribution.
\item RNA: Gene expression contribution.
\item Values reported as mean $\pm$ std across """ + str(len(SEEDS)) + r""" random seeds.
\end{tablenotes}
\end{table}
"""
    
    print(latex)
    
    with open(RESULTS_DIR / "latex_table.tex", 'w') as f:
        f.write(latex)
    print(f"\n  Saved: latex_table.tex")
    
    return latex


def main():
    print("\n" + "=" * 70)
    print("INTERSHAP PUBLICATION ANALYSIS")
    print("=" * 70)
    
    # Step 1: Multi-seed analysis
    results_df = run_multi_seed_analysis()
    
    if results_df is None or len(results_df) == 0:
        print("\nError: No results generated. Check individual fusion scripts.")
        return
    
    # Step 2: Statistical analysis
    stats_df, pval_df = compute_statistics(results_df)
    
    # Step 3: Generate figures
    create_figures(results_df, stats_df)
    
    # Step 4: Generate LaTeX table
    generate_latex_table(stats_df)
    
    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("\nFiles generated:")
    print("  - multi_seed_results.csv")
    print("  - statistical_summary.csv")
    print("  - pairwise_tests.csv")
    print("  - latex_table.tex")
    print("  - fig1_intershap_comparison.png/pdf")
    print("  - fig2_contribution_breakdown.png/pdf")
    print("  - fig3_intershap_boxplot.png/pdf")
    print("  - fig4_risk_group_heatmap.png/pdf")


if __name__ == "__main__":
    main()
