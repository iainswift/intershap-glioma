"""
Cross-Validation Robustness Analysis for InterSHAP

Verifies that InterSHAP results are stable across different train/test splits.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import glob

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("CROSS-VALIDATION ROBUSTNESS ANALYSIS")
print("=" * 70)

# Find all fold models
model_dir = os.path.join(project_root, 'MyData', 'results', 'early_fusion', 
                         'checkpoints', 'models')
fold_dirs = sorted(glob.glob(os.path.join(model_dir, 'earlyfusion_model_survcox_fold*')))

print(f"\nFound {len(fold_dirs)} fold models")

# Load test data
test_csv = os.path.join(project_root, 'MyData', 'splits', 'early_test.csv')
df = pd.read_csv(test_csv)
features = df.iloc[:, 4:].values.astype(np.float32)
baseline = features.mean(axis=0)

# Compute InterSHAP for each fold's model
fold_results = []

for fold_dir in fold_dirs:
    fold_name = os.path.basename(fold_dir)
    model_path = os.path.join(fold_dir, 'model_dict_best.pt')
    
    if not os.path.exists(model_path):
        print(f"  {fold_name}: model not found, skipping")
        continue
    
    # Load model
    model = nn.Sequential(
        nn.Dropout(), nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout(),
        nn.Linear(2048, 200), nn.ReLU(), nn.Dropout(), nn.Linear(200, 1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval().to(device)
    
    # Compute InterSHAP for all samples
    intershap_values = []
    wsi_values = []
    rna_values = []
    
    for i in range(len(features)):
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
    
    fold_results.append({
        'fold': fold_name,
        'intershap_mean': np.mean(intershap_values),
        'intershap_std': np.std(intershap_values),
        'wsi_mean': np.mean(wsi_values),
        'rna_mean': np.mean(rna_values)
    })
    
    print(f"  {fold_name}: InterSHAP = {np.mean(intershap_values):.2f}% ± {np.std(intershap_values):.2f}%")

# Summary
results_df = pd.DataFrame(fold_results)

print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(f"\nInterSHAP across {len(fold_results)} folds:")
print(f"  Mean: {results_df['intershap_mean'].mean():.2f}% +/- {results_df['intershap_mean'].std():.2f}%")
print(f"  Range: [{results_df['intershap_mean'].min():.2f}%, {results_df['intershap_mean'].max():.2f}%]")
print(f"  WSI: {results_df['wsi_mean'].mean():.2f}% +/- {results_df['wsi_mean'].std():.2f}%")
print(f"  RNA: {results_df['rna_mean'].mean():.2f}% +/- {results_df['rna_mean'].std():.2f}%")

# Coefficient of Variation
cv = results_df['intershap_mean'].std() / results_df['intershap_mean'].mean() * 100
print(f"Coefficient of Variation: {cv:.1f}%")

# Visualize
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))

folds = [r['fold'].replace('earlyfusion_model_survcox_', '') for r in fold_results]
intershap_means = [r['intershap_mean'] for r in fold_results]
intershap_stds = [r['intershap_std'] for r in fold_results]

x = np.arange(len(folds))
bars = ax.bar(x, intershap_means, yerr=intershap_stds, capsize=5, 
              color='steelblue', alpha=0.8, edgecolor='black')

ax.axhline(np.mean(intershap_means), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(intershap_means):.2f}%')
ax.fill_between([-0.5, len(folds)-0.5], 
                np.mean(intershap_means) - np.std(intershap_means),
                np.mean(intershap_means) + np.std(intershap_means),
                color='red', alpha=0.1, label='±1 std')

ax.set_xticks(x)
ax.set_xticklabels(folds, rotation=45, ha='right')
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('InterSHAP (%)', fontsize=12)
ax.set_title('InterSHAP Stability Across Cross-Validation Folds', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cv_robustness.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved: {output_path}")

# Save detailed results
results_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               'cv_robustness_results.csv'), index=False)
