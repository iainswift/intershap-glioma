"""
InterSHAP Analysis on FULL Dataset (Train + Test)
=================================================
Uses all ~575 patients instead of just 114 test patients.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("INTERSHAP ANALYSIS - FULL DATASET")
print("=" * 70)

# Load BOTH train and test data
train_csv = os.path.join(project_root, 'MyData', 'splits', 'early_train.csv')
test_csv = os.path.join(project_root, 'MyData', 'splits', 'early_test.csv')

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# Combine
df_all = pd.concat([df_train, df_test], ignore_index=True)
print(f"\nTotal samples: {len(df_all)} (Train: {len(df_train)}, Test: {len(df_test)})")

# Check for duplicates
unique_cases = df_all['case'].nunique()
print(f"Unique patients: {unique_cases}")

# Load model (fold 0)
model_path = os.path.join(project_root, 'MyData', 'results', 'early_fusion', 
                          'checkpoints', 'models', 'earlyfusion_model_survcox_fold0',
                          'model_dict_best.pt')

model = nn.Sequential(
    nn.Dropout(), nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout(),
    nn.Linear(2048, 200), nn.ReLU(), nn.Dropout(), nn.Linear(200, 1)
)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval().to(device)

# Extract features
features = df_all.iloc[:, 4:].values.astype(np.float32)
baseline = features.mean(axis=0)

# Compute InterSHAP for ALL samples
print(f"\nComputing InterSHAP for {len(df_all)} samples...")

intershap_values = []
wsi_values = []
rna_values = []

for i in range(len(df_all)):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(df_all)}")
    
    sample = features[i:i+1]
    rna = sample[:, :2048]
    wsi = sample[:, 2048:]
    rna_base = baseline[:2048].reshape(1, -1)
    wsi_base = baseline[2048:].reshape(1, -1)
    
    with torch.no_grad():
        v_empty = model(torch.from_numpy(np.concatenate([rna_base, wsi_base], axis=1)).float().to(device)).item()
        v_wsi = model(torch.from_numpy(np.concatenate([rna_base, wsi], axis=1)).float().to(device)).item()
        v_rna = model(torch.from_numpy(np.concatenate([rna, wsi_base], axis=1)).float().to(device)).item()
        v_full = model(torch.from_numpy(sample).float().to(device)).item()
    
    phi_wsi = 0.5 * (v_wsi - v_empty) + 0.5 * (v_full - v_rna)
    phi_rna = 0.5 * (v_rna - v_empty) + 0.5 * (v_full - v_wsi)
    phi_int = 0.5 * (v_full - v_wsi - v_rna + v_empty)
    
    total = abs(phi_wsi) + abs(phi_rna) + abs(phi_int)
    if total > 0:
        intershap_values.append(100 * abs(phi_int) / total)
        wsi_values.append(100 * abs(phi_wsi) / total)
        rna_values.append(100 * abs(phi_rna) / total)
    else:
        intershap_values.append(0)
        wsi_values.append(0)
        rna_values.append(0)

# Add to dataframe
df_all['intershap'] = intershap_values
df_all['wsi_contribution'] = wsi_values
df_all['rna_contribution'] = rna_values

# Summary statistics
print("\n" + "=" * 70)
print("FULL DATASET INTERSHAP RESULTS")
print("=" * 70)

print(f"\nResults (N={len(df_all)}):")
print(f"  InterSHAP: {np.mean(intershap_values):.2f}% +/- {np.std(intershap_values):.2f}%")
print(f"  WSI:       {np.mean(wsi_values):.2f}% +/- {np.std(wsi_values):.2f}%")
print(f"  RNA:       {np.mean(rna_values):.2f}% +/- {np.std(rna_values):.2f}%")

# Kaplan-Meier on FULL dataset
print("=" * 70)
print("KAPLAN-MEIER ANALYSIS (FULL DATASET)")
print("=" * 70)

survival_time = df_all['survival_months'].values
event = df_all['vital_status'].values

print(f"\nSurvival data: {len(survival_time)} patients")
print(f"Events (deaths): {event.sum()} / {len(event)} ({100*event.mean():.1f}%)")

# Median split
median_intershap = np.median(intershap_values)
high_interaction = np.array(intershap_values) >= median_intershap
low_interaction = np.array(intershap_values) < median_intershap

print(f"\nMedian InterSHAP: {median_intershap:.2f}%")
print(f"High interaction: {high_interaction.sum()} patients")
print(f"Low interaction: {low_interaction.sum()} patients")

# KM analysis
kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()

kmf_high.fit(survival_time[high_interaction], event[high_interaction], 
             label=f'High InterSHAP (≥{median_intershap:.1f}%)')
kmf_low.fit(survival_time[low_interaction], event[low_interaction], 
            label=f'Low InterSHAP (<{median_intershap:.1f}%)')

# Log-rank test
results = logrank_test(
    survival_time[high_interaction], survival_time[low_interaction],
    event[high_interaction], event[low_interaction]
)

print(f"\nSurvival by InterSHAP (N={len(df_all)}):")
print(f"  High InterSHAP (>={median_intershap:.1f}%): N={high_interaction.sum()}, Median survival={kmf_high.median_survival_time_:.1f} mo")
print(f"  Low InterSHAP (<{median_intershap:.1f}%):  N={low_interaction.sum()}, Median survival={kmf_low.median_survival_time_:.1f} mo")
print(f"  Log-rank p={results.p_value:.6f}")

# Spearman correlation
corr, p_corr = spearmanr(intershap_values, survival_time)
print(f"Spearman correlation (InterSHAP vs survival): r = {corr:.3f}, p = {p_corr:.4f}")

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: KM curves
ax1 = axes[0]
kmf_high.plot_survival_function(ax=ax1, ci_show=True, color='red')
kmf_low.plot_survival_function(ax=ax1, ci_show=True, color='blue')
ax1.set_xlabel('Time (months)', fontsize=12)
ax1.set_ylabel('Survival Probability', fontsize=12)
ax1.set_title(f'Survival by InterSHAP Level (N={len(df_all)})', fontsize=13, fontweight='bold')
ax1.legend(loc='lower left', fontsize=10)
ax1.text(0.95, 0.95, f'Log-rank p = {results.p_value:.4f}', 
         transform=ax1.transAxes, ha='right', va='top', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: Distribution
ax2 = axes[1]
ax2.hist(intershap_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax2.axvline(median_intershap, color='red', linestyle='--', linewidth=2, 
            label=f'Median = {median_intershap:.2f}%')
ax2.set_xlabel('InterSHAP (%)', fontsize=12)
ax2.set_ylabel('Number of Patients', fontsize=12)
ax2.set_title(f'InterSHAP Distribution (N={len(df_all)})', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)

# Plot 3: Scatter
ax3 = axes[2]
colors = ['red' if e == 1 else 'blue' for e in event]
ax3.scatter(intershap_values, survival_time, c=colors, alpha=0.5, s=30)
ax3.set_xlabel('InterSHAP (%)', fontsize=12)
ax3.set_ylabel('Survival Time (months)', fontsize=12)
ax3.set_title(f'InterSHAP vs Survival (r={corr:.3f}, p={p_corr:.4f})', fontsize=13, fontweight='bold')

# Legend for scatter
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Deceased'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Censored')]
ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'full_dataset_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f"\nFigure saved: {output_path}")

# Save detailed results
df_all[['case', 'survival_months', 'vital_status', 'survival_bin', 
        'intershap', 'wsi_contribution', 'rna_contribution']].to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'full_dataset_intershap.csv'), 
    index=False)
print("Results saved: full_dataset_intershap.csv")

print(f"\nLog-rank p = {results.p_value:.4f}")
