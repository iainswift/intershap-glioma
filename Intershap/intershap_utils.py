"""
Unified InterSHAP utilities for all fusion strategies.
Implements proper Shapley Interaction Index computation at the modality level.

Based on: "Measuring Cross-Modal Interactions in Multimodal Models" (AAAI-25)
          Wenderoth et al., University of Cambridge
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# CORE INTERSHAP IMPLEMENTATION (Based on Paper Section 3)
# =============================================================================

class ModalityMasker:
    """Handles masking strategies for modality ablation."""
    
    def __init__(self, baseline_data, mask_type='mean'):
        """
        Args:
            baseline_data: List of [modality1_data, modality2_data, ...] arrays
            mask_type: 'mean', 'zero', or 'noise'
        """
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
                # Use Gaussian noise with dataset statistics
                self.baselines.append(np.random.randn(mod_data.shape[1]) * np.std(mod_data, axis=0))
    
    def mask_sample(self, sample, coalition):
        """
        Mask modalities not in the coalition.
        
        Args:
            sample: concatenated feature vector
            coalition: set of modality indices to KEEP (e.g., {0}, {1}, {0,1})
        
        Returns:
            masked sample with baseline values for excluded modalities
        """
        masked = sample.copy()
        cum_dim = 0
        
        for mod_idx, (baseline, mod_dim) in enumerate(zip(self.baselines, self.modality_dims)):
            if mod_idx not in coalition:
                masked[cum_dim:cum_dim + mod_dim] = baseline
            cum_dim += mod_dim
            
        return masked


def compute_shapley_interaction_index(model, sample, masker, device, num_modalities=2):
    """
    Compute the Shapley Interaction Index for a single sample.
    
    Based on Equation (2) and (3) from the InterSHAP paper:
    
    For M=2 modalities, Equation (2) simplifies to:
    φ_01 = 0.5 * ∇_01({}, f)
    
    Where ∇_01 (Equation 3) = f({0,1}) - f({0}) - f({1}) + f({})
    
    The 0.5 coefficient comes from: |S|!(M-|S|-2)! / 2(M-1)! = 0!*0!/2*1! = 0.5
    
    Args:
        model: PyTorch model that takes concatenated features
        sample: numpy array of concatenated features
        masker: ModalityMasker instance
        device: torch device
        num_modalities: number of modalities (default 2)
    
    Returns:
        dict with interaction values and modality contributions
    """
    model.eval()
    
    def get_output(coalition):
        """Get model output for a given coalition of modalities."""
        masked = masker.mask_sample(sample, coalition)
        with torch.no_grad():
            x = torch.tensor(masked, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(x)
            # Use softmax probabilities
            probs = torch.softmax(out, dim=1)
            return probs.cpu().numpy()[0]
    
    # For M=2 modalities (i=0, j=1):
    f_both = get_output({0, 1})      # Both modalities present
    f_wsi = get_output({0})          # Only WSI (modality 0)
    f_rna = get_output({1})          # Only RNA (modality 1)
    f_none = get_output(set())       # Neither (baseline)
    
    # ∇_01 (Equation 3): Raw interaction effect for empty coalition S
    nabla_01 = f_both - f_wsi - f_rna + f_none
    
    # φ_01 (Equation 2): Shapley Interaction Index with proper weighting
    # For M=2: coefficient = |S|!(M-|S|-2)! / 2(M-1)! = 0!*0! / 2*1! = 0.5
    phi_01 = 0.5 * nabla_01
    
    # Shapley values for each modality (Equation 1)
    # φ_i = Σ_{S⊆M\{i}} [|S|!(M-|S|-1)!/M!] * [f(S∪{i}) - f(S)]
    # For M=2, i=0: φ_0 = 0.5*[(f({0})-f({})) + (f({0,1})-f({1}))]
    phi_wsi = 0.5 * ((f_wsi - f_none) + (f_both - f_rna))
    phi_rna = 0.5 * ((f_rna - f_none) + (f_both - f_wsi))
    
    # Self-contribution (Equation 4): φ_ii = φ_i - Σ_{j≠i} φ_ij
    # This is the unique contribution of each modality (excluding interactions)
    phi_wsi_self = phi_wsi - phi_01
    phi_rna_self = phi_rna - phi_01
    
    return {
        'phi_01': phi_01,              # Shapley Interaction Index (cross-modal)
        'phi_wsi': phi_wsi,            # Total WSI Shapley value
        'phi_rna': phi_rna,            # Total RNA Shapley value
        'phi_wsi_self': phi_wsi_self,  # WSI unique contribution (excluding interaction)
        'phi_rna_self': phi_rna_self,  # RNA unique contribution (excluding interaction)
        'nabla_01': nabla_01,          # Raw interaction (before weighting)
        'f_both': f_both,
        'f_wsi': f_wsi,
        'f_rna': f_rna,
        'f_none': f_none
    }


def compute_global_intershap(model, dataset, device, n_samples=None, mask_type='mean'):
    """
    Compute Global InterSHAP score across a dataset.
    
    Based on Equation (5)-(8) from the paper:
    - Φ_ij = mean(|φ_ij|) across samples
    - Interactions = Σ Φ_ij for i≠j
    - Behavior = Σ Φ_ij for all i,j
    - InterSHAP = Interactions / Behavior
    
    Args:
        model: PyTorch model
        dataset: Dataset with get_data() method returning [mod1, mod2], labels
        device: torch device
        n_samples: number of samples to use (None for all)
        mask_type: baseline masking strategy ('mean', 'zero', 'noise')
    
    Returns:
        dict with InterSHAP score and modality contributions as percentages
    """
    modality_data, labels = dataset.get_data()
    
    if n_samples is not None and n_samples < len(labels):
        indices = np.random.choice(len(labels), n_samples, replace=False)
    else:
        indices = np.arange(len(labels))
    
    masker = ModalityMasker(modality_data, mask_type=mask_type)
    
    # Collect per-sample results
    all_interactions = []
    all_phi_wsi_self = []
    all_phi_rna_self = []
    
    print(f"Computing InterSHAP for {len(indices)} samples...")
    for i, idx in enumerate(indices):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(indices)}")
        
        # Concatenate modalities for this sample
        sample = np.concatenate([mod[idx] for mod in modality_data])
        
        result = compute_shapley_interaction_index(model, sample, masker, device)
        
        # Take absolute mean across classes (as per paper Equation 5)
        all_interactions.append(np.abs(result['phi_01']).mean())
        all_phi_wsi_self.append(np.abs(result['phi_wsi_self']).mean())
        all_phi_rna_self.append(np.abs(result['phi_rna_self']).mean())
    
    # Global aggregation (Equation 5-7)
    Phi_interaction = np.mean(all_interactions)
    Phi_wsi = np.mean(all_phi_wsi_self)
    Phi_rna = np.mean(all_phi_rna_self)
    
    # Total behavior = sum of all contributions
    total_behavior = Phi_wsi + Phi_rna + Phi_interaction
    
    # InterSHAP score (Equation 8)
    if total_behavior > 1e-10:
        intershap_score = (Phi_interaction / total_behavior) * 100
        wsi_contribution = (Phi_wsi / total_behavior) * 100
        rna_contribution = (Phi_rna / total_behavior) * 100
    else:
        intershap_score = 0.0
        wsi_contribution = 0.0
        rna_contribution = 0.0
    
    return {
        'InterSHAP': intershap_score,
        'WSI_Contribution': wsi_contribution,
        'RNA_Contribution': rna_contribution,
        'Interaction_Raw': Phi_interaction,
        'WSI_Raw': Phi_wsi,
        'RNA_Raw': Phi_rna,
        'Total_Behavior': total_behavior,
        'n_samples': len(indices)
    }


def compute_local_intershap(model, dataset, device, mask_type='mean'):
    """
    Compute per-sample InterSHAP scores for clinical analysis.
    
    Based on Equation (10) from the paper:
    I_a = (Σ |φ_ij| for i≠j) / (Σ |φ_ij| for all i,j)
    
    Args:
        model: PyTorch model
        dataset: Dataset with get_data() method
        device: torch device
        mask_type: baseline masking strategy
    
    Returns:
        DataFrame with per-sample interaction scores
    """
    modality_data, labels = dataset.get_data()
    masker = ModalityMasker(modality_data, mask_type=mask_type)
    
    results = []
    
    print(f"Computing local InterSHAP for {len(labels)} samples...")
    for idx in range(len(labels)):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(labels)}")
            
        sample = np.concatenate([mod[idx] for mod in modality_data])
        result = compute_shapley_interaction_index(model, sample, masker, device)
        
        # Local InterSHAP (Equation 10)
        interaction = np.abs(result['phi_01']).mean()
        wsi_self = np.abs(result['phi_wsi_self']).mean()
        rna_self = np.abs(result['phi_rna_self']).mean()
        
        total = interaction + wsi_self + rna_self
        
        if total > 1e-10:
            local_intershap = (interaction / total) * 100
            wsi_pct = (wsi_self / total) * 100
            rna_pct = (rna_self / total) * 100
        else:
            local_intershap = 0.0
            wsi_pct = 0.0
            rna_pct = 0.0
        
        results.append({
            'sample_idx': idx,
            'survival_bin': labels[idx],
            'local_intershap': local_intershap,
            'wsi_contribution': wsi_pct,
            'rna_contribution': rna_pct,
            'interaction_raw': interaction,
            'wsi_raw': wsi_self,
            'rna_raw': rna_self,
            'total_behavior': total
        })
    
    return pd.DataFrame(results)


def analyze_by_risk_group(local_results_df, save_path=None):
    """
    Stratify InterSHAP scores by survival bin for clinical interpretation.
    
    Survival bins: 0 = shortest survival (high risk), 3 = longest (low risk)
    
    Args:
        local_results_df: DataFrame from compute_local_intershap
        save_path: optional path to save results
    
    Returns:
        Summary DataFrame grouped by survival bin
    """
    summary = local_results_df.groupby('survival_bin').agg({
        'local_intershap': ['mean', 'std', 'count'],
        'wsi_contribution': ['mean', 'std'],
        'rna_contribution': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary.columns = ['Survival_Bin', 'InterSHAP_Mean', 'InterSHAP_Std', 'N_Samples',
                       'WSI_Mean', 'WSI_Std', 'RNA_Mean', 'RNA_Std']
    
    # Add clinical interpretation
    risk_map = {
        0: 'High Risk (Shortest Survival)',
        1: 'Medium-High Risk',
        2: 'Medium-Low Risk',
        3: 'Low Risk (Longest Survival)'
    }
    summary['Risk_Level'] = summary['Survival_Bin'].map(risk_map)
    
    # Rename for consistency with aggregation
    summary = summary.rename(columns={
        'Survival_Bin': 'Risk_Group',
        'InterSHAP_Mean': 'Mean_InterSHAP',
        'InterSHAP_Std': 'Std_InterSHAP'
    })
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        summary.to_csv(os.path.join(save_path, 'risk_group_analysis.csv'), index=False)
    
    return summary


# =============================================================================
# ADAPTER CLASSES FOR EACH FUSION TYPE
# =============================================================================

class EarlyFusionAdapter(Dataset):
    """Adapter for pre-extracted concatenated features (Early Fusion).
    
    IMPORTANT: Feature order in early_*.csv is [RNA (0:2048), WSI (2048:4096)]
    This is because 1_Concat2Features.py uses:
        merged_df = rna_df.merge(pathology_df, on='case')
    which places RNA features BEFORE WSI features.
    
    Despite the CSV column order, we return [WSI, RNA] for consistency with
    the modality ordering in InterSHAP (modality 0 = WSI, modality 1 = RNA).
    """
    
    def __init__(self, data, labels=None, wsi_dim=2048, rna_dim=2048, device='cpu'):
        """
        Args:
            data: either path to CSV or numpy array of features
            labels: numpy array of labels (required if data is numpy array)
            wsi_dim: dimension of WSI features
            rna_dim: dimension of RNA features
            device: torch device
        """
        self.device = device
        self.wsi_dim = wsi_dim
        self.rna_dim = rna_dim
        
        if isinstance(data, str):
            # Load from CSV
            df = pd.read_csv(data)
            feature_cols = [c for c in df.columns if 'feature_' in c]
            self.features = df[feature_cols].values.astype(np.float32)
            self.labels = df['survival_bin'].values.astype(int)
            self.case_ids = df['case'].values if 'case' in df.columns else np.arange(len(df))
        else:
            # Use numpy arrays directly
            self.features = data.astype(np.float32)
            self.labels = labels.astype(int)
            self.case_ids = np.arange(len(labels))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32).to(self.device),
                torch.tensor(self.labels[idx], dtype=torch.long).to(self.device))
    
    def get_data(self):
        """Return modality-separated data for InterSHAP.
        
        CRITICAL: The CSV has feature order [RNA, WSI], but we return [WSI, RNA]
        for semantic consistency (modality 0 = WSI, modality 1 = RNA).
        
        CSV columns: feature_0 to feature_2047 = RNA
                     feature_2048 to feature_4095 = WSI
        
        Returns: [WSI_features, RNA_features] = [features[2048:4096], features[0:2048]]
        """
        # CORRECTED: Extract in semantic order [WSI, RNA], not CSV order [RNA, WSI]
        rna_feats = self.features[:, :self.rna_dim]          # feature_0 to 2047 = RNA
        wsi_feats = self.features[:, self.rna_dim:self.rna_dim + self.wsi_dim]  # feature_2048 to 4095 = WSI
        return [wsi_feats, rna_feats], self.labels  # Return [WSI, RNA] for semantic consistency
    
    def get_modality_shapes(self):
        return [self.wsi_dim, self.rna_dim]


class LateFusionAdapter(Dataset):
    """Adapter for late fusion (score-level combination)."""
    
    def __init__(self, csv_path, device='cpu'):
        """
        Args:
            csv_path: path to CSV with path_score, rna_score columns
            device: torch device
        """
        self.device = device
        df = pd.read_csv(csv_path)
        
        # Filter valid samples
        df = df[df['survival_months'] > 0].reset_index(drop=True)
        
        self.wsi_scores = df['path_score'].values.astype(np.float32).reshape(-1, 1)
        self.rna_scores = df['rna_score'].values.astype(np.float32).reshape(-1, 1)
        self.labels = df['survival_bin'].values.astype(int)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = np.concatenate([self.wsi_scores[idx], self.rna_scores[idx]])
        return (torch.tensor(x, dtype=torch.float32).to(self.device),
                torch.tensor(self.labels[idx], dtype=torch.long).to(self.device))
    
    def get_data(self):
        return [self.wsi_scores, self.rna_scores], self.labels
    
    def get_modality_shapes(self):
        return [1, 1]


class JointFusionAdapter(Dataset):
    """Adapter for joint fusion (pre-extracted embeddings from joint model)."""
    
    def __init__(self, embeddings, labels, wsi_dim=2048, rna_dim=2048, device='cpu'):
        """
        Args:
            embeddings: numpy array of concatenated [wsi_emb, rna_emb]
            labels: numpy array of survival bin labels
            wsi_dim: dimension of WSI embeddings
            rna_dim: dimension of RNA embeddings
            device: torch device
        """
        self.device = device
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(int)
        self.wsi_dim = wsi_dim
        self.rna_dim = rna_dim
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.embeddings[idx], dtype=torch.float32).to(self.device),
                torch.tensor(self.labels[idx], dtype=torch.long).to(self.device))
    
    def get_data(self):
        wsi = self.embeddings[:, :self.wsi_dim]
        rna = self.embeddings[:, self.wsi_dim:self.wsi_dim + self.rna_dim]
        return [wsi, rna], self.labels
    
    def get_modality_shapes(self):
        return [self.wsi_dim, self.rna_dim]
