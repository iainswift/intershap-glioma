"""
InterSHAP Analysis for Attention-Based Fusion Models

Compares InterSHAP (cross-modal synergy) between:
1. Original MLP (baseline)
2. Cross-Attention Fusion
3. Bilinear Fusion
4. Gated Fusion
"""

import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Intershap'))

import torch
import numpy as np
import pandas as pd
from scipy import stats

# Import models
from models_attention import get_model


def load_attention_model(model_type, model_path, device):
    """Load a trained attention model."""
    
    model_kwargs = {'dropout': 0.25}
    if model_type in ['cross_attention', 'gated']:
        model_kwargs['hidden_dim'] = 256
    if model_type == 'bilinear':
        model_kwargs['rank'] = 64
        
    model = get_model(model_type, **model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


class AttentionModelAdapter:
    """Adapter for attention models to work with InterSHAP."""
    
    def __init__(self, model, data_path, device):
        self.model = model
        self.device = device
        
        # Load test data
        df = pd.read_csv(data_path)
        self.patient_ids = df['case'].values
        # Features start at column 4 (case, survival_months, vital_status, survival_bin, feature_0...)
        self.features = df.iloc[:, 4:].values.astype(np.float32)
        
        # Compute baseline (mean of each feature across test set)
        self.baseline = self.features.mean(axis=0)
        
    def get_data(self):
        """Return [WSI features, RNA features] and patient IDs."""
        # CSV format: [RNA (0:2048), WSI (2048:4096)]
        rna_feats = self.features[:, :2048]
        wsi_feats = self.features[:, 2048:]
        
        # Return [WSI, RNA] for Shapley convention
        return [wsi_feats, rna_feats], self.patient_ids
    
    def predict(self, inputs):
        """Run model prediction with mean baseline masking."""
        wsi, rna = inputs  # inputs[0]=WSI, inputs[1]=RNA
        
        # Reconstruct combined features in correct order [RNA, WSI]
        combined = np.concatenate([rna, wsi], axis=1)
        
        with torch.no_grad():
            tensor = torch.from_numpy(combined).float().to(self.device)
            output = self.model(tensor).cpu().numpy()
        
        return output.squeeze()
    
    def get_baseline(self):
        """Return [WSI baseline, RNA baseline]."""
        rna_baseline = self.baseline[:2048]
        wsi_baseline = self.baseline[2048:]
        return [wsi_baseline, rna_baseline]


def run_intershap_for_model(model, model_name, data_path, device):
    """Run InterSHAP analysis for a single model."""
    
    adapter = AttentionModelAdapter(model, data_path, device)
    [wsi_data, rna_data], patient_ids = adapter.get_data()
    [wsi_baseline, rna_baseline] = adapter.get_baseline()
    
    n_samples = len(patient_ids)
    
    # Compute InterSHAP scores
    all_phi_wsi = []
    all_phi_rna = []
    all_phi_interaction = []
    all_phi_total = []
    
    for i in range(n_samples):
        wsi_i = wsi_data[i:i+1]
        rna_i = rna_data[i:i+1]
        
        # v({}) - empty coalition (both masked)
        combined_baseline = np.concatenate([rna_baseline.reshape(1, -1), 
                                            wsi_baseline.reshape(1, -1)], axis=1)
        with torch.no_grad():
            tensor = torch.from_numpy(combined_baseline).float().to(device)
            v_empty = model(tensor).cpu().numpy().item()
        
        # v({WSI}) - only WSI present
        combined_wsi_only = np.concatenate([rna_baseline.reshape(1, -1), wsi_i], axis=1)
        with torch.no_grad():
            tensor = torch.from_numpy(combined_wsi_only).float().to(device)
            v_wsi = model(tensor).cpu().numpy().item()
        
        # v({RNA}) - only RNA present
        combined_rna_only = np.concatenate([rna_i, wsi_baseline.reshape(1, -1)], axis=1)
        with torch.no_grad():
            tensor = torch.from_numpy(combined_rna_only).float().to(device)
            v_rna = model(tensor).cpu().numpy().item()
        
        # v({WSI, RNA}) - full model
        combined_full = np.concatenate([rna_i, wsi_i], axis=1)
        with torch.no_grad():
            tensor = torch.from_numpy(combined_full).float().to(device)
            v_full = model(tensor).cpu().numpy().item()
        
        # Shapley values (for M=2)
        # phi_WSI = 0.5 * [v({WSI}) - v({})] + 0.5 * [v({WSI,RNA}) - v({RNA})]
        phi_wsi = 0.5 * (v_wsi - v_empty) + 0.5 * (v_full - v_rna)
        phi_rna = 0.5 * (v_rna - v_empty) + 0.5 * (v_full - v_wsi)
        
        # Interaction (Shapley Interaction Index)
        # Delta_01 = v({WSI,RNA}) - v({WSI}) - v({RNA}) + v({})
        delta = v_full - v_wsi - v_rna + v_empty
        phi_interaction = 0.5 * delta
        
        all_phi_wsi.append(phi_wsi)
        all_phi_rna.append(phi_rna)
        all_phi_interaction.append(phi_interaction)
        all_phi_total.append(abs(phi_wsi) + abs(phi_rna) + abs(phi_interaction))
    
    # Convert to arrays
    phi_wsi = np.array(all_phi_wsi)
    phi_rna = np.array(all_phi_rna)
    phi_interaction = np.array(all_phi_interaction)
    phi_total = np.array(all_phi_total)
    
    # Compute percentage contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_wsi = 100 * np.abs(phi_wsi) / phi_total
        pct_rna = 100 * np.abs(phi_rna) / phi_total
        pct_interaction = 100 * np.abs(phi_interaction) / phi_total
        
        pct_wsi = np.nan_to_num(pct_wsi, nan=33.33)
        pct_rna = np.nan_to_num(pct_rna, nan=33.33)
        pct_interaction = np.nan_to_num(pct_interaction, nan=33.33)
    
    results = {
        'model': model_name,
        'patient_id': patient_ids,
        'phi_wsi': phi_wsi,
        'phi_rna': phi_rna,
        'phi_interaction': phi_interaction,
        'pct_wsi': pct_wsi,
        'pct_rna': pct_rna,
        'pct_interaction': pct_interaction
    }
    
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Models were saved relative to script directory
    model_dir = os.path.join(script_dir, 'MyData', 'results', 'attention_models')
    data_path = os.path.join(project_root, 'MyData', 'splits', 'early_test.csv')
    
    # Models to analyze
    models_to_test = ['cross_attention', 'bilinear', 'gated']
    
    # Check which models are available
    available_models = []
    for m in models_to_test:
        path = os.path.join(model_dir, f'{m}_best.pt')
        if os.path.exists(path):
            available_models.append(m)
        else:
            print(f"WARNING: {m} model not found at {path}")
    
    if not available_models:
        print("ERROR: No trained attention models found!")
        print(f"Please train models first using: python train_attention_models.py --model <model_type>")
        return
    
    print(f"\nAnalyzing {len(available_models)} models: {available_models}")
    print("=" * 70)
    
    all_results = {}
    
    for model_name in available_models:
        print(f"\n--- Analyzing {model_name.upper()} ---")
        
        model_path = os.path.join(model_dir, f'{model_name}_best.pt')
        model = load_attention_model(model_name, model_path, device)
        
        results = run_intershap_for_model(model, model_name, data_path, device)
        all_results[model_name] = results
        
        # Print summary for this model
        print(f"  Samples: {len(results['patient_id'])}")
        print(f"  InterSHAP: {np.mean(results['pct_interaction']):.2f}% ± {np.std(results['pct_interaction']):.2f}%")
        print(f"  WSI:       {np.mean(results['pct_wsi']):.2f}% ± {np.std(results['pct_wsi']):.2f}%")
        print(f"  RNA:       {np.mean(results['pct_rna']):.2f}% ± {np.std(results['pct_rna']):.2f}%")
    
    # Compare to original MLP results (from previous analysis)
    mlp_intershap = 4.82  # From our corrected analysis
    
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS: InterSHAP Across Architectures")
    print("=" * 70)
    print(f"\n{'Model':<20} {'InterSHAP %':>15} {'WSI %':>15} {'RNA %':>15}")
    print("-" * 70)
    print(f"{'Original MLP':<20} {mlp_intershap:>15.2f} {'42.24':>15} {'52.94':>15}")
    
    for model_name, results in all_results.items():
        intershap = np.mean(results['pct_interaction'])
        wsi = np.mean(results['pct_wsi'])
        rna = np.mean(results['pct_rna'])
        print(f"{model_name:<20} {intershap:>15.2f} {wsi:>15.2f} {rna:>15.2f}")
    
    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS: InterSHAP vs Original MLP (4.82%)")
    print("=" * 70)
    
    for model_name, results in all_results.items():
        intershap_values = results['pct_interaction']
        
        # One-sample t-test against MLP baseline
        t_stat, p_value = stats.ttest_1samp(intershap_values, mlp_intershap)
        mean_diff = np.mean(intershap_values) - mlp_intershap
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "NS"
        
        print(f"\n{model_name}:")
        print(f"  Mean InterSHAP: {np.mean(intershap_values):.2f}%")
        print(f"  Difference from MLP: {mean_diff:+.2f}%")
        print(f"  t-statistic: {t_stat:.2f}")
        print(f"  p-value: {p_value:.4f} {significance}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    max_intershap = max([np.mean(r['pct_interaction']) for r in all_results.values()])
    
    print(f"\nMax InterSHAP across attention models: {max_intershap:.2f}%")
    print(f"Baseline MLP InterSHAP: {mlp_intershap:.2f}%")
    
    # Save detailed results
    output_dir = os.path.join(project_root, 'Intershap')
    
    # Save per-patient results
    for model_name, results in all_results.items():
        df = pd.DataFrame({
            'patient_id': results['patient_id'],
            'phi_wsi': results['phi_wsi'],
            'phi_rna': results['phi_rna'],
            'phi_interaction': results['phi_interaction'],
            'pct_wsi': results['pct_wsi'],
            'pct_rna': results['pct_rna'],
            'pct_interaction': results['pct_interaction']
        })
        df.to_csv(os.path.join(output_dir, f'intershap_{model_name}.csv'), index=False)
    
    # Save summary comparison
    summary_path = os.path.join(output_dir, 'attention_comparison.csv')
    rows = [{'model': 'Original MLP', 'intershap': mlp_intershap, 'wsi': 42.24, 'rna': 52.94}]
    for model_name, results in all_results.items():
        rows.append({
            'model': model_name,
            'intershap': np.mean(results['pct_interaction']),
            'wsi': np.mean(results['pct_wsi']),
            'rna': np.mean(results['pct_rna'])
        })
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
