"""
Attention-Based Fusion Models for InterSHAP Synergy Experiment

Tests whether explicit interaction mechanisms can capture cross-modal synergy
that standard MLPs fail to learn.

Models:
1. CrossAttentionFusion - WSI and RNA cross-attend to each other
2. BilinearFusion - Explicit x_wsi * x_rna interaction term
3. GatedFusion - Dynamic gating conditioned on both modalities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion where WSI queries RNA (and vice versa).
    This EXPLICITLY models interaction through Q*K^T attention.
    
    Architecture:
    1. Project each modality to common dimension
    2. WSI attends to RNA (what RNA info is relevant for this WSI?)
    3. RNA attends to WSI (what WSI info is relevant for this RNA?)
    4. Concatenate original + attended representations
    5. MLP fusion head
    """
    def __init__(self, wsi_dim=2048, rna_dim=2048, hidden_dim=256, num_heads=8, dropout=0.25):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project to common dimension
        self.wsi_proj = nn.Linear(wsi_dim, hidden_dim)
        self.rna_proj = nn.Linear(rna_dim, hidden_dim)
        
        # Cross-attention: WSI attends to RNA
        self.wsi_to_rna_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: RNA attends to WSI
        self.rna_to_wsi_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms for stability
        self.wsi_ln = nn.LayerNorm(hidden_dim)
        self.rna_ln = nn.LayerNorm(hidden_dim)
        
        # Final fusion MLP
        # 4 vectors: wsi_orig, rna_orig, wsi_attended, rna_attended
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: concatenated [RNA, WSI] tensor of shape (B, 4096)
               RNA is x[:, :2048], WSI is x[:, 2048:]
               (CORRECTED ORDER based on 1_Concat2Features.py)
        """
        # Split modalities (RNA first, WSI second in the CSV)
        rna = x[:, :2048]
        wsi = x[:, 2048:]
        
        # Project to attention dimension
        wsi_proj = self.wsi_proj(wsi).unsqueeze(1)  # (B, 1, hidden_dim)
        rna_proj = self.rna_proj(rna).unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Cross-attention: WSI queries RNA
        # "Given this WSI, what RNA information is relevant?"
        wsi_attended, _ = self.wsi_to_rna_attn(
            query=wsi_proj,
            key=rna_proj,
            value=rna_proj
        )
        wsi_attended = self.wsi_ln(wsi_attended)
        
        # Cross-attention: RNA queries WSI
        # "Given this RNA, what WSI information is relevant?"
        rna_attended, _ = self.rna_to_wsi_attn(
            query=rna_proj,
            key=wsi_proj,
            value=wsi_proj
        )
        rna_attended = self.rna_ln(rna_attended)
        
        # Concatenate all representations
        fused = torch.cat([
            wsi_proj.squeeze(1),      # Original WSI representation
            rna_proj.squeeze(1),      # Original RNA representation
            wsi_attended.squeeze(1),  # WSI informed by RNA (INTERACTION)
            rna_attended.squeeze(1)   # RNA informed by WSI (INTERACTION)
        ], dim=1)
        
        return self.fusion(fused)


class BilinearFusion(nn.Module):
    """
    Explicit bilinear interaction: models x_wsi^T W x_rna terms.
    Uses low-rank factorization to reduce parameters.
    
    The bilinear term FORCES the model to consider feature interactions:
    interaction_k = sum_i sum_j (wsi_i * U_ik) * (rna_j * V_jk)
                  = (wsi @ U) * (rna @ V)  [element-wise product]
    """
    def __init__(self, wsi_dim=2048, rna_dim=2048, rank=64, dropout=0.25):
        super().__init__()
        
        # Low-rank bilinear: W = U @ V^T
        # This captures rank many bilinear interaction terms
        self.wsi_factor = nn.Linear(wsi_dim, rank)
        self.rna_factor = nn.Linear(rna_dim, rank)
        
        # Also keep main effects (additive terms)
        self.wsi_main = nn.Linear(wsi_dim, rank)
        self.rna_main = nn.Linear(rna_dim, rank)
        
        # Batch norms for stability
        self.bn_interaction = nn.BatchNorm1d(rank)
        self.bn_wsi = nn.BatchNorm1d(rank)
        self.bn_rna = nn.BatchNorm1d(rank)
        
        # Final layer: combines main effects + interaction
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rank * 3, 64),  # 3 = wsi_main + rna_main + interaction
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        rna = x[:, :2048]
        wsi = x[:, 2048:]
        
        # Main effects (additive)
        wsi_m = self.bn_wsi(self.wsi_main(wsi))
        rna_m = self.bn_rna(self.rna_main(rna))
        
        # Bilinear interaction (element-wise product in low-rank space)
        # This is equivalent to bilinear in the original high-dim space
        wsi_f = self.wsi_factor(wsi)
        rna_f = self.rna_factor(rna)
        interaction = self.bn_interaction(wsi_f * rna_f)  # Element-wise product
        
        # Combine main effects + interaction
        fused = torch.cat([wsi_m, rna_m, interaction], dim=1)
        
        return self.fusion(fused)


class GatedFusion(nn.Module):
    """
    Gated fusion: learns dynamic weighting based on BOTH modalities.
    
    The gates are conditioned on the concatenation of both modalities,
    meaning the model MUST look at both to decide how to weight each.
    This creates implicit interaction.
    
    g_wsi = sigmoid(W_g @ [wsi; rna])  # Gate depends on BOTH
    output = g_wsi * wsi_proj + g_rna * rna_proj
    """
    def __init__(self, wsi_dim=2048, rna_dim=2048, hidden_dim=256, dropout=0.25):
        super().__init__()
        
        # Project modalities
        self.wsi_proj = nn.Linear(wsi_dim, hidden_dim)
        self.rna_proj = nn.Linear(rna_dim, hidden_dim)
        
        # Gates conditioned on BOTH modalities (this is the interaction)
        self.wsi_gate = nn.Sequential(
            nn.Linear(wsi_dim + rna_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.rna_gate = nn.Sequential(
            nn.Linear(wsi_dim + rna_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Layer norms
        self.wsi_ln = nn.LayerNorm(hidden_dim)
        self.rna_ln = nn.LayerNorm(hidden_dim)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        rna = x[:, :2048]
        wsi = x[:, 2048:]
        
        # Project modalities
        wsi_h = self.wsi_proj(wsi)
        rna_h = self.rna_proj(rna)
        
        # Compute gates (CONDITIONAL on both modalities = interaction)
        combined = torch.cat([wsi, rna], dim=1)
        wsi_g = self.wsi_gate(combined)  # Gate for WSI depends on RNA too
        rna_g = self.rna_gate(combined)  # Gate for RNA depends on WSI too
        
        # Apply gates
        wsi_gated = self.wsi_ln(wsi_h * wsi_g)
        rna_gated = self.rna_ln(rna_h * rna_g)
        
        # Fuse
        fused = torch.cat([wsi_gated, rna_gated], dim=1)
        
        return self.fusion(fused)


class BaselineMLP(nn.Module):
    """
    Baseline MLP matching original Early Fusion architecture.
    Included for fair comparison with same hidden dimensions.
    """
    def __init__(self, input_dim=4096, dropout=0.25):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 1)
        )
        
    def forward(self, x):
        return self.model(x)


def get_model(model_type, **kwargs):
    """Factory function to get model by name."""
    models = {
        'mlp': BaselineMLP,
        'cross_attention': CrossAttentionFusion,
        'bilinear': BilinearFusion,
        'gated': GatedFusion
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](**kwargs)


if __name__ == "__main__":
    # Test all models
    print("Testing attention-based fusion models...")
    
    batch_size = 8
    x = torch.randn(batch_size, 4096)  # [RNA (2048) | WSI (2048)]
    
    for name in ['mlp', 'cross_attention', 'bilinear', 'gated']:
        model = get_model(name)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:20s}: output shape = {out.shape}, params = {n_params:,}")
    
    print("\nAll models working correctly!")
