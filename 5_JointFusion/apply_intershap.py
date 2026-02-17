"""
InterSHAP Analysis for Joint Fusion (End-to-End Multimodal)
============================================================

Cox Regression Version: Model outputs single risk score (num_classes=1).

This is the most meaningful InterSHAP application since the model
learns to jointly process both modalities from raw inputs.

The analysis extracts embeddings from the trained joint model and
applies InterSHAP to the fusion head.

InterSHAP measures the Shapley Interaction Index between WSI and RNA
modalities. For Cox regression, we use the raw risk score as the
value function.
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

# Add Intershap utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Intershap'))
from intershap_utils import set_seed

from models import BagHistopathologyRNAModel
from resnet import resnet50
from datasets import PatchBagRNADataset


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized patch bags"""
    patch_bags = [item['patch_bag'] for item in batch]
    rna_data = torch.stack([item['rna_data'] for item in batch])
    survival_months = torch.tensor([item['survival_months'] for item in batch])
    vital_status = torch.tensor([item['vital_status'] for item in batch])
    
    batch_dict = {
        'patch_bag': patch_bags,
        'rna_data': rna_data,
        'survival_months': survival_months,
        'vital_status': vital_status,
    }
    
    if 'survival_bin' in batch[0]:
        batch_dict['survival_bin'] = torch.tensor([item['survival_bin'] for item in batch])
        
    return batch_dict


class ModalityMaskerCox:
    """
    Masking utility for Cox regression embeddings (output dim = 1).
    
    WSI embeddings: indices [0:2048]
    RNA embeddings: indices [2048:4096]
    """
    def __init__(self, embeddings, mask_type='mean'):
        self.mask_type = mask_type
        self.wsi_dim = 2048
        self.rna_dim = 2048
        
        # Compute reference values for masking
        self.wsi_ref = embeddings[:, :self.wsi_dim].mean(axis=0)
        self.rna_ref = embeddings[:, self.wsi_dim:].mean(axis=0)
        
        if mask_type == 'zero':
            self.wsi_ref = np.zeros(self.wsi_dim)
            self.rna_ref = np.zeros(self.rna_dim)
    
    def mask(self, x, mask_wsi=False, mask_rna=False):
        """Mask specified modalities."""
        x_masked = x.copy()
        if mask_wsi:
            x_masked[:, :self.wsi_dim] = self.wsi_ref
        if mask_rna:
            x_masked[:, self.wsi_dim:] = self.rna_ref
        return x_masked


def compute_shapley_interaction_cox(model, x, masker, device):
    """
    Compute Shapley Interaction Index for Cox regression.
    
    For Cox regression, we use the raw risk score as the value function.
    
    Args:
        model: Cox head model (outputs single risk score)
        x: Single sample embedding [1, 4096]
        masker: ModalityMaskerCox instance
        device: torch device
        
    Returns:
        dict with phi_wsi, phi_rna, phi_01, total_behavior
    """
    model.eval()
    
    # Create all 4 coalition inputs
    x_both = x                                           # f({0,1})
    x_wsi_only = masker.mask(x, mask_wsi=False, mask_rna=True)   # f({0})
    x_rna_only = masker.mask(x, mask_wsi=True, mask_rna=False)   # f({1})
    x_empty = masker.mask(x, mask_wsi=True, mask_rna=True)       # f({})
    
    with torch.no_grad():
        f_both = model(torch.tensor(x_both, dtype=torch.float32).to(device)).cpu().numpy().squeeze()
        f_wsi = model(torch.tensor(x_wsi_only, dtype=torch.float32).to(device)).cpu().numpy().squeeze()
        f_rna = model(torch.tensor(x_rna_only, dtype=torch.float32).to(device)).cpu().numpy().squeeze()
        f_empty = model(torch.tensor(x_empty, dtype=torch.float32).to(device)).cpu().numpy().squeeze()
    
    # Shapley values for M=2 case (closed form)
    # phi_0 = (f({0}) - f({})) + 0.5*(f({0,1}) - f({0}) - f({1}) + f({}))
    # phi_1 = (f({1}) - f({})) + 0.5*(f({0,1}) - f({0}) - f({1}) + f({}))
    nabla_01 = f_both - f_wsi - f_rna + f_empty  # Discrete derivative
    
    phi_wsi = (f_wsi - f_empty) + 0.5 * nabla_01
    phi_rna = (f_rna - f_empty) + 0.5 * nabla_01
    phi_01 = 0.5 * nabla_01  # Shapley Interaction Index
    
    total = f_both - f_empty  # Total change from baseline
    
    return {
        'phi_wsi': phi_wsi,
        'phi_rna': phi_rna,
        'phi_01': phi_01,
        'total_behavior': total,
        'f_both': f_both,
        'f_wsi': f_wsi,
        'f_rna': f_rna,
        'f_empty': f_empty
    }


def compute_global_intershap_cox(model, embeddings, labels, masker, device, n_samples=None):
    """
    Compute Global InterSHAP for Cox regression.
    
    Aggregates local InterSHAP scores across all samples.
    """
    n = len(embeddings) if n_samples is None else min(n_samples, len(embeddings))
    indices = np.random.choice(len(embeddings), n, replace=False) if n_samples else np.arange(len(embeddings))
    
    all_phi_wsi = []
    all_phi_rna = []
    all_phi_01 = []
    all_total = []
    
    print(f"Computing InterSHAP for {n} samples...")
    for i, idx in enumerate(indices):
        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/{n} samples")
        
        x = embeddings[idx:idx+1]  # [1, 4096]
        result = compute_shapley_interaction_cox(model, x, masker, device)
        
        all_phi_wsi.append(result['phi_wsi'])
        all_phi_rna.append(result['phi_rna'])
        all_phi_01.append(result['phi_01'])
        all_total.append(result['total_behavior'])
    
    # Aggregate using absolute values for attribution
    mean_abs_phi_wsi = np.mean(np.abs(all_phi_wsi))
    mean_abs_phi_rna = np.mean(np.abs(all_phi_rna))
    mean_abs_phi_01 = np.mean(np.abs(all_phi_01))
    total_behavior = mean_abs_phi_wsi + mean_abs_phi_rna + mean_abs_phi_01
    
    if total_behavior > 1e-10:
        intershap_pct = (mean_abs_phi_01 / total_behavior) * 100
        wsi_pct = (mean_abs_phi_wsi / total_behavior) * 100
        rna_pct = (mean_abs_phi_rna / total_behavior) * 100
    else:
        intershap_pct = wsi_pct = rna_pct = 0.0
    
    return {
        'InterSHAP': intershap_pct,
        'WSI_Contribution': wsi_pct,
        'RNA_Contribution': rna_pct,
        'Total_Behavior': total_behavior,
        'Mean_Phi_WSI': np.mean(all_phi_wsi),
        'Mean_Phi_RNA': np.mean(all_phi_rna),
        'Mean_Phi_Interaction': np.mean(all_phi_01),
        'Std_Phi_Interaction': np.std(all_phi_01)
    }


def compute_local_intershap_cox(model, embeddings, labels, masker, device):
    """
    Compute per-sample (local) InterSHAP for Cox regression.
    """
    results = []
    
    print(f"Computing local InterSHAP for {len(embeddings)} samples...")
    for i in range(len(embeddings)):
        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/{len(embeddings)} samples")
        
        x = embeddings[i:i+1]
        result = compute_shapley_interaction_cox(model, x, masker, device)
        
        # Local InterSHAP percentage
        total = np.abs(result['phi_wsi']) + np.abs(result['phi_rna']) + np.abs(result['phi_01'])
        if total > 1e-10:
            local_intershap = (np.abs(result['phi_01']) / total) * 100
            local_wsi = (np.abs(result['phi_wsi']) / total) * 100
            local_rna = (np.abs(result['phi_rna']) / total) * 100
        else:
            local_intershap = local_wsi = local_rna = 0.0
        
        results.append({
            'sample_idx': i,
            'survival_bin': labels[i],
            'phi_wsi': result['phi_wsi'],
            'phi_rna': result['phi_rna'],
            'phi_01': result['phi_01'],
            'local_intershap_pct': local_intershap,
            'local_wsi_pct': local_wsi,
            'local_rna_pct': local_rna,
            'risk_score': result['f_both']
        })
    
    return pd.DataFrame(results)


def analyze_by_risk_group(local_df, save_dir):
    """Analyze InterSHAP by survival risk group."""
    risk_labels = {0: 'High Risk', 1: 'Medium-High', 2: 'Medium-Low', 3: 'Low Risk'}
    local_df['risk_group'] = local_df['survival_bin'].map(risk_labels)
    
    summary = local_df.groupby('risk_group').agg({
        'local_intershap_pct': ['mean', 'std', 'count'],
        'local_wsi_pct': 'mean',
        'local_rna_pct': 'mean',
        'phi_01': 'mean'
    }).round(2)
    
    summary.columns = ['InterSHAP_Mean', 'InterSHAP_Std', 'N_Samples', 
                       'WSI_Mean', 'RNA_Mean', 'Interaction_Raw']
    summary = summary.reset_index()
    summary.to_csv(os.path.join(save_dir, 'risk_group_analysis.csv'), index=False)
    
    return summary


def extract_embeddings(model, dataset, device, batch_size=1):
    """
    Pre-extract WSI and RNA embeddings from the full joint model.
    
    Args:
        model: BagHistopathologyRNAModel
        dataset: PatchBagRNADataset
        device: torch device
        batch_size: batch size for extraction
    
    Returns:
        embeddings: numpy array [N, 4096] (WSI 2048 + RNA 2048)
        labels: numpy array [N]
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    
    all_embeddings = []
    all_labels = []
    
    print(f"Extracting embeddings from {len(dataset)} samples...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(loader)} batches")
            
            # Extract WSI embedding via ResNet
            patches = batch['patch_bag'][0].to(device).unsqueeze(0)
            wsi_feats = model.resnet.forward_extract(patches.squeeze(0))
            wsi_embedding = wsi_feats.view(1, -1, 2048).mean(dim=1)  # [1, 2048]
            
            # Extract RNA embedding via MLP
            rna_data = batch['rna_data'].to(device)
            rna_embedding = model.rna_mlp(rna_data)  # [1, 2048]
            
            # Concatenate
            combined = torch.cat([wsi_embedding, rna_embedding], dim=1)
            all_embeddings.append(combined.cpu().numpy())
            all_labels.append(batch['survival_bin'].item())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    
    print(f"[OK] Extracted {len(labels)} embeddings with shape {embeddings.shape}")
    return embeddings, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InterSHAP analysis for Joint Fusion (Cox)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="../MyData/results/intershap/joint",
                        help="Directory to save results")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit samples for InterSHAP (None for all)")
    parser.add_argument("--checkpoint", type=str,
                        default="../MyData/results/joint_fusion/checkpoints/models/jointfusion_model_survcox_fold0/model_dict_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="../MyData/splits/joint_test.csv",
                        help="Path to test data CSV")
    parser.add_argument("--patch_path", type=str, default="../MyData/patches",
                        help="Path to patch images")
    parser.add_argument("--bag_size", type=int, default=40, help="Patches per bag")
    parser.add_argument("--mask_type", type=str, default="mean", choices=["mean", "zero", "noise"],
                        help="Masking strategy for ablation")
    parser.add_argument("--use_cached", action="store_true",
                        help="Use cached embeddings if available")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check for cached embeddings
    cache_path = os.path.join(args.save_dir, 'embeddings_cache.npz')
    
    # Build full joint model (Cox version: num_classes=1)
    print("\nBuilding joint model (Cox regression, output=1)...")
    resnet = resnet50(pretrained=False)
    rna_mlp = nn.Sequential(
        nn.Dropout(),
        nn.Linear(12778, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 2048)
    )
    # Cox regression: final output is 1 (risk score)
    final_mlp = nn.Sequential(
        nn.Dropout(0.8),
        nn.Linear(4096, 1)  # Cox: single risk score output
    )
    full_model = BagHistopathologyRNAModel(resnet, rna_mlp, final_mlp)
    
    if os.path.exists(args.checkpoint):
        full_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[OK] Loaded checkpoint: {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    full_model.to(device).eval()
    
    if args.use_cached and os.path.exists(cache_path):
        print(f"[OK] Loading cached embeddings from {cache_path}")
        cached = np.load(cache_path)
        embeddings = cached['embeddings']
        labels = cached['labels']
    else:
        # Load raw data
        print("\nLoading dataset...")
        tsfm = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        test_raw = PatchBagRNADataset(
            args.patch_path,
            args.data_path,
            224,
            transforms=tsfm,
            bag_size=args.bag_size
        )
        
        # Extract embeddings
        print("\n" + "="*50)
        print("EXTRACTING EMBEDDINGS")
        print("="*50)
        embeddings, labels = extract_embeddings(full_model, test_raw, device)
        
        # Cache embeddings
        np.savez(cache_path, embeddings=embeddings, labels=labels)
        print(f"[OK] Cached embeddings to {cache_path}")
    
    # Create masker for InterSHAP
    masker = ModalityMaskerCox(embeddings, mask_type=args.mask_type)
    
    # Build fusion head model (just the final_mlp)
    head_model = nn.Sequential(
        nn.Dropout(0.8),
        nn.Linear(4096, 1)  # Cox: single risk score output
    )
    head_model.load_state_dict(full_model.final_mlp.state_dict())
    head_model.to(device).eval()
    
    # Compute Global InterSHAP
    print("\n" + "="*50)
    print("COMPUTING GLOBAL INTERSHAP (Cox Regression)")
    print("="*50)
    
    global_results = compute_global_intershap_cox(
        head_model, embeddings, labels, masker, device,
        n_samples=args.n_samples
    )
    
    print(f"\n--- Global Results ---")
    print(f"InterSHAP Score:    {global_results['InterSHAP']:.2f}%")
    print(f"WSI Contribution:   {global_results['WSI_Contribution']:.2f}%")
    print(f"RNA Contribution:   {global_results['RNA_Contribution']:.2f}%")
    print(f"Total Behavior:     {global_results['Total_Behavior']:.6f}")
    
    # Compute Local InterSHAP
    print("\n" + "="*50)
    print("COMPUTING LOCAL INTERSHAP (Per-Sample)")
    print("="*50)
    
    local_results = compute_local_intershap_cox(
        head_model, embeddings, labels, masker, device
    )
    
    # Risk Group Analysis
    print("\n" + "="*50)
    print("RISK GROUP ANALYSIS")
    print("="*50)
    
    risk_analysis = analyze_by_risk_group(local_results, args.save_dir)
    print(risk_analysis.to_string(index=False))
    
    # Save all results
    global_summary = pd.DataFrame([{
        'Fusion_Type': 'Joint',
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
    print(f"  - embeddings_cache.npz")