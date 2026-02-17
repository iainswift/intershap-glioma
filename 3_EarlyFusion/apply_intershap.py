"""
InterSHAP Analysis for Early Fusion (Feature Concatenation)

Measures cross-modal interactions between WSI and RNA modalities
at the feature level after concatenation.

NOTE: This model outputs a single Cox risk score (not 4-class classification).
For InterSHAP, we treat the risk score directly as the model's behavior metric.
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Add Intershap utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Intershap'))
from intershap_utils import (
    set_seed, EarlyFusionAdapter, analyze_by_risk_group
)


def get_early_fusion_model(input_dim=4096, num_classes=1):
    """
    Build early fusion model matching training architecture.
    
    Architecture from 2_EarlyFusion_train.py:
    - Dropout()
    - Linear(4096, 2048)
    - ReLU()
    - Dropout()
    - Linear(2048, 200)
    - ReLU()
    - Dropout()
    - Linear(200, num_classes)
    
    For survcox task: num_classes=1 (single risk score output)
    """
    model = nn.Sequential(
        nn.Dropout(),           # 0
        nn.Linear(input_dim, 2048),  # 1
        nn.ReLU(),              # 2
        nn.Dropout(),           # 3
        nn.Linear(2048, 200),   # 4
        nn.ReLU(),              # 5
        nn.Dropout(),           # 6
        nn.Linear(200, num_classes),  # 7
    )
    return model


class ModalityMaskerCox:
    """Handles masking for Cox regression models (single output)."""
    
    def __init__(self, baseline_data, mask_type='mean'):
        self.mask_type = mask_type
        self.baselines = []
        self.modality_dims = []
        
        for mod_data in baseline_data:
            self.modality_dims.append(mod_data.shape[1])
            if mask_type == 'mean':
                self.baselines.append(np.mean(mod_data, axis=0))
            elif mask_type == 'zero':
                self.baselines.append(np.zeros(mod_data.shape[1]))
            elif mask_type == 'noise':
                self.baselines.append(np.random.randn(mod_data.shape[1]) * np.std(mod_data, axis=0))
    
    def mask_sample(self, sample, coalition):
        masked = sample.copy()
        cum_dim = 0
        for mod_idx, (baseline, mod_dim) in enumerate(zip(self.baselines, self.modality_dims)):
            if mod_idx not in coalition:
                masked[cum_dim:cum_dim + mod_dim] = baseline
            cum_dim += mod_dim
        return masked


def compute_shapley_interaction_cox(model, sample, masker, device):
    """
    Compute Shapley Interaction Index for Cox regression model.
    
    For Cox models (single risk score output), we use the raw risk score
    as the model behavior metric instead of class probabilities.
    """
    model.eval()
    
    def get_risk_score(coalition):
        masked = masker.mask_sample(sample, coalition)
        with torch.no_grad():
            x = torch.tensor(masked, dtype=torch.float32).unsqueeze(0).to(device)
            return model(x).cpu().item()
    
    # Coalition outputs
    f_both = get_risk_score({0, 1})
    f_wsi = get_risk_score({0})
    f_rna = get_risk_score({1})
    f_none = get_risk_score(set())
    
    # Shapley Interaction Index (Equation 2-3)
    nabla_01 = f_both - f_wsi - f_rna + f_none
    phi_01 = 0.5 * nabla_01  # Weight for M=2
    
    # Shapley values (Equation 1)
    phi_wsi = 0.5 * ((f_wsi - f_none) + (f_both - f_rna))
    phi_rna = 0.5 * ((f_rna - f_none) + (f_both - f_wsi))
    
    # Self-contributions (Equation 4)
    phi_wsi_self = phi_wsi - phi_01
    phi_rna_self = phi_rna - phi_01
    
    return {
        'phi_01': phi_01,
        'phi_wsi_self': phi_wsi_self,
        'phi_rna_self': phi_rna_self,
        'risk_both': f_both,
        'risk_wsi': f_wsi,
        'risk_rna': f_rna,
        'risk_none': f_none
    }


def compute_global_intershap_cox(model, dataset, device, n_samples=None, mask_type='mean'):
    """Global InterSHAP for Cox regression models."""
    modality_data, labels = dataset.get_data()
    
    if n_samples is not None and n_samples < len(labels):
        indices = np.random.choice(len(labels), n_samples, replace=False)
    else:
        indices = np.arange(len(labels))
    
    masker = ModalityMaskerCox(modality_data, mask_type=mask_type)
    
    all_interactions = []
    all_phi_wsi_self = []
    all_phi_rna_self = []
    
    print(f"Computing InterSHAP for {len(indices)} samples...")
    for i, idx in enumerate(indices):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(indices)}")
        
        sample = np.concatenate([mod[idx] for mod in modality_data])
        result = compute_shapley_interaction_cox(model, sample, masker, device)
        
        all_interactions.append(np.abs(result['phi_01']))
        all_phi_wsi_self.append(np.abs(result['phi_wsi_self']))
        all_phi_rna_self.append(np.abs(result['phi_rna_self']))
    
    Phi_interaction = np.mean(all_interactions)
    Phi_wsi = np.mean(all_phi_wsi_self)
    Phi_rna = np.mean(all_phi_rna_self)
    
    total_behavior = Phi_wsi + Phi_rna + Phi_interaction
    
    if total_behavior > 1e-10:
        intershap_score = (Phi_interaction / total_behavior) * 100
        wsi_contribution = (Phi_wsi / total_behavior) * 100
        rna_contribution = (Phi_rna / total_behavior) * 100
    else:
        intershap_score = wsi_contribution = rna_contribution = 0.0
    
    return {
        'InterSHAP': intershap_score,
        'WSI_Contribution': wsi_contribution,
        'RNA_Contribution': rna_contribution,
        'Total_Behavior': total_behavior,
        'n_samples': len(indices)
    }


def compute_local_intershap_cox(model, dataset, device, mask_type='mean'):
    """Per-sample InterSHAP for Cox regression models."""
    modality_data, labels = dataset.get_data()
    masker = ModalityMaskerCox(modality_data, mask_type=mask_type)
    
    results = []
    print(f"Computing local InterSHAP for {len(labels)} samples...")
    
    for idx in range(len(labels)):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(labels)}")
        
        sample = np.concatenate([mod[idx] for mod in modality_data])
        result = compute_shapley_interaction_cox(model, sample, masker, device)
        
        interaction = np.abs(result['phi_01'])
        wsi_self = np.abs(result['phi_wsi_self'])
        rna_self = np.abs(result['phi_rna_self'])
        
        total = interaction + wsi_self + rna_self
        
        if total > 1e-10:
            local_intershap = (interaction / total) * 100
            wsi_pct = (wsi_self / total) * 100
            rna_pct = (rna_self / total) * 100
        else:
            local_intershap = wsi_pct = rna_pct = 0.0
        
        results.append({
            'sample_idx': idx,
            'survival_bin': labels[idx],
            'local_intershap': local_intershap,
            'wsi_contribution': wsi_pct,
            'rna_contribution': rna_pct,
            'risk_score': result['risk_both']
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InterSHAP analysis for Early Fusion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="../MyData/results/intershap/early",
                        help="Directory to save results")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples for global InterSHAP (None for all)")
    parser.add_argument("--checkpoint", type=str, 
                        default="../MyData/results/early_fusion/checkpoints/models/earlyfusion_model_survcox_fold0/model_dict_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="../MyData/splits/early_test.csv",
                        help="Path to test data CSV")
    parser.add_argument("--mask_type", type=str, default="mean", choices=["mean", "zero", "noise"],
                        help="Masking strategy for ablation")
    parser.add_argument("--num_classes", type=int, default=1,
                        help="Number of output classes (1 for Cox)")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    model = get_early_fusion_model(input_dim=4096, num_classes=args.num_classes)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"[OK] Loaded checkpoint: {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    model.to(device).eval()
    
    # Load data
    test_ds = EarlyFusionAdapter(
        args.data_path,
        wsi_dim=2048,
        rna_dim=2048,
        device=device
    )
    print(f"[OK] Loaded dataset: {len(test_ds)} samples")
    
    # 1. Compute Global InterSHAP
    print("\n" + "="*50)
    print("COMPUTING GLOBAL INTERSHAP (Cox Model)")
    print("="*50)
    
    global_results = compute_global_intershap_cox(
        model, test_ds, device,
        n_samples=args.n_samples,
        mask_type=args.mask_type
    )
    
    print(f"\n--- Global Results ---")
    print(f"InterSHAP Score:    {global_results['InterSHAP']:.2f}%")
    print(f"WSI Contribution:   {global_results['WSI_Contribution']:.2f}%")
    print(f"RNA Contribution:   {global_results['RNA_Contribution']:.2f}%")
    print(f"Total Behavior:     {global_results['Total_Behavior']:.6f}")
    
    # 2. Compute Local InterSHAP (per-patient)
    print("\n" + "="*50)
    print("COMPUTING LOCAL INTERSHAP (Per-Sample)")
    print("="*50)
    
    local_results = compute_local_intershap_cox(
        model, test_ds, device,
        mask_type=args.mask_type
    )
    
    # 3. Risk Group Analysis
    print("\n" + "="*50)
    print("RISK GROUP ANALYSIS")
    print("="*50)
    
    risk_analysis = analyze_by_risk_group(local_results, args.save_dir)
    print(risk_analysis.to_string(index=False))
    
    # 4. Save all results
    global_summary = pd.DataFrame([{
        'Fusion_Type': 'Early',
        'Seed': args.seed,
        'Mask_Type': args.mask_type,
        **global_results
    }])
    global_summary.to_csv(os.path.join(args.save_dir, 'global_summary.csv'), index=False)
    
    local_results.to_csv(os.path.join(args.save_dir, 'local_results.csv'), index=False)
    
    print(f"\n[OK] Results saved to {args.save_dir}")
    print(f"  - global_summary.csv")
    print(f"  - local_results.csv")
    print(f"  - risk_group_analysis.csv")