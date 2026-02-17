"""
InterSHAP Analysis for Late Fusion (Score Combination)

NOTE: Late fusion operates at the decision level, not feature level.
InterSHAP here measures how much the combination rule benefits from
having both scores vs. individual scores.

A proxy model is trained to map the two scores to survival bins,
then InterSHAP is applied to this proxy.
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add Intershap utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Intershap'))
from intershap_utils import (
    set_seed, LateFusionAdapter, compute_global_intershap,
    compute_local_intershap, analyze_by_risk_group
)


class LateFusionProxy(nn.Module):
    """
    Proxy model that learns to combine WSI and RNA scores.
    Uses a small MLP to capture potential non-linear interactions.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def train_proxy(model, train_ds, device, epochs=100, lr=0.01):
    """Train proxy model to mimic late fusion behavior."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    return model


def validate_proxy(model, test_ds, device):
    """Check if proxy model is valid for InterSHAP analysis."""
    model.eval()
    loader = DataLoader(test_ds, batch_size=32)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = torch.argmax(model(x), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InterSHAP analysis for Late Fusion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="../MyData/results/intershap/late",
                        help="Directory to save results")
    parser.add_argument("--train_path", type=str, 
                        default="../MyData/results/late_fusion/combined_scores_train.csv",
                        help="Path to training scores CSV")
    parser.add_argument("--test_path", type=str,
                        default="../MyData/results/late_fusion/combined_scores_test.csv",
                        help="Path to test scores CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Proxy training epochs")
    parser.add_argument("--mask_type", type=str, default="mean", choices=["mean", "zero", "noise"],
                        help="Masking strategy for ablation")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    train_ds = LateFusionAdapter(args.train_path, device)
    test_ds = LateFusionAdapter(args.test_path, device)
    print(f"[OK] Loaded data: Train={len(train_ds)}, Test={len(test_ds)} samples")
    
    # Train proxy model
    print("\n" + "="*50)
    print("TRAINING PROXY MODEL")
    print("="*50)
    
    model = LateFusionProxy(num_classes=4).to(device)
    model = train_proxy(model, train_ds, device, epochs=args.epochs)
    
    # Validate proxy
    acc = validate_proxy(model, test_ds, device)
    print(f"\n[OK] Proxy Model Test Accuracy: {acc:.4f}")
    
    if acc < 0.35:
        print("[!] WARNING: Proxy accuracy is near random (0.25).")
        print("  InterSHAP results may be unreliable.")
        print("  Consider: more epochs, different architecture, or check data quality")
    
    model.eval()
    
    # Compute Global InterSHAP
    print("\n" + "="*50)
    print("COMPUTING GLOBAL INTERSHAP")
    print("="*50)
    
    global_results = compute_global_intershap(
        model, test_ds, device,
        n_samples=None,  # Use all samples (late fusion is fast)
        mask_type=args.mask_type
    )
    
    print(f"\n--- Global Results ---")
    print(f"InterSHAP Score:       {global_results['InterSHAP']:.2f}%")
    print(f"WSI Score Contribution: {global_results['WSI_Contribution']:.2f}%")
    print(f"RNA Score Contribution: {global_results['RNA_Contribution']:.2f}%")
    
    # Compute Local InterSHAP
    print("\n" + "="*50)
    print("COMPUTING LOCAL INTERSHAP (Per-Sample)")
    print("="*50)
    
    local_results = compute_local_intershap(
        model, test_ds, device,
        mask_type=args.mask_type
    )
    
    # Risk Group Analysis
    print("\n" + "="*50)
    print("RISK GROUP ANALYSIS")
    print("="*50)
    
    risk_analysis = analyze_by_risk_group(local_results, args.save_dir)
    print(risk_analysis.to_string(index=False))
    
    # Save all results
    global_summary = pd.DataFrame([{
        'Fusion_Type': 'Late',
        'Seed': args.seed,
        'Proxy_Accuracy': acc,
        'Mask_Type': args.mask_type,
        **global_results
    }])
    global_summary.to_csv(os.path.join(args.save_dir, 'global_summary.csv'), index=False)
    
    local_results.to_csv(os.path.join(args.save_dir, 'local_results.csv'), index=False)
    
    # Save proxy model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'proxy_model.pt'))
    
    print(f"\n[OK] Results saved to {args.save_dir}")
    print(f"  - global_summary.csv")
    print(f"  - local_results.csv")
    print(f"  - risk_group_analysis.csv")
    print(f"  - proxy_model.pt")