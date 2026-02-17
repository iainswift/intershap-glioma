"""
Train Attention-Based Fusion Models for InterSHAP Synergy Experiment

This script trains Cross-Attention, Bilinear, and Gated fusion models to test
whether explicit interaction mechanisms can capture more cross-modal synergy.

Usage:
    python train_attention_models.py --model cross_attention --epochs 80
    python train_attention_models.py --model bilinear --epochs 80
    python train_attention_models.py --model gated --epochs 80
    python train_attention_models.py --model mlp --epochs 80  # baseline
"""

import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from lifelines.utils import concordance_index

# Import our new models
from models_attention import get_model, CrossAttentionFusion, BilinearFusion, GatedFusion, BaselineMLP
from datasets import featureDataset


def cox_loss(risk_scores, survival_times, events):
    """
    Cox proportional hazards partial log-likelihood loss.
    
    risk_scores: predicted risk (higher = worse prognosis)
    survival_times: time to event/censoring
    events: 1 if event occurred, 0 if censored
    """
    # Sort by survival time descending
    sorted_indices = torch.argsort(survival_times, descending=True)
    sorted_risks = risk_scores[sorted_indices]
    sorted_events = events[sorted_indices]
    
    # Compute log-partial likelihood
    log_cumsum_exp = torch.logcumsumexp(sorted_risks, dim=0)
    
    # Only count uncensored events in the loss
    log_likelihood = sorted_risks - log_cumsum_exp
    censored_likelihood = log_likelihood * sorted_events
    
    # Negative mean (we want to maximize likelihood)
    n_events = sorted_events.sum()
    if n_events > 0:
        return -censored_likelihood.sum() / n_events
    else:
        return torch.tensor(0.0, device=risk_scores.device)


def evaluate(model, dataloader, device):
    """Evaluate model and return C-index and loss."""
    model.eval()
    
    all_risks = []
    all_times = []
    all_events = []
    all_ids = []
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['feature_data'].to(device)
            times = batch['survival_months'].to(device).float()
            events = batch['vital_status'].to(device).float()
            
            risks = model(features).squeeze()
            
            loss = cox_loss(risks, times, events)
            total_loss += loss.item()
            n_batches += 1
            
            all_risks.append(risks.cpu().numpy())
            all_times.append(times.cpu().numpy())
            all_events.append(events.cpu().numpy())
            all_ids.extend(batch['case'])
    
    # Aggregate by patient ID (for multiple patches per patient)
    all_risks = np.concatenate(all_risks)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)
    
    # Aggregate by unique patient
    unique_ids = sorted(set(all_ids))
    patient_risks = []
    patient_times = []
    patient_events = []
    
    for pid in unique_ids:
        mask = [i for i, x in enumerate(all_ids) if x == pid]
        patient_risks.append(np.mean(all_risks[mask]))
        patient_times.append(all_times[mask[0]])
        patient_events.append(all_events[mask[0]])
    
    patient_risks = np.array(patient_risks)
    patient_times = np.array(patient_times)
    patient_events = np.array(patient_events)
    
    # Compute C-index (higher risk = lower survival, so negate)
    c_index = concordance_index(patient_times, -patient_risks, patient_events)
    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    
    return c_index, avg_loss


def train_one_epoch(model, dataloader, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        features = batch['feature_data'].to(device)
        times = batch['survival_months'].to(device).float()
        events = batch['vital_status'].to(device).float()
        
        optimizer.zero_grad()
        risks = model(features).squeeze()
        
        loss = cox_loss(risks, times, events)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Train Attention-Based Fusion Models')
    parser.add_argument('--model', type=str, default='cross_attention',
                        choices=['mlp', 'cross_attention', 'bilinear', 'gated'],
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for attention models')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_csv', type=str, default='MyData/splits/early_train.csv',
                        help='Path to training CSV')
    parser.add_argument('--test_csv', type=str, default='MyData/splits/early_test.csv',
                        help='Path to test CSV')
    parser.add_argument('--output_dir', type=str, default='MyData/results/attention_models',
                        help='Output directory for models')
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(project_root, args.train_csv)
    test_csv = os.path.join(project_root, args.test_csv)
    
    # Load datasets
    print(f"Loading training data from: {train_csv}")
    print(f"Loading test data from: {test_csv}")
    
    train_dataset = featureDataset(train_csv)
    test_dataset = featureDataset(test_csv)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(test_dataset),
        num_workers=0
    )
    
    # Create model
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} model")
    print(f"{'='*60}")
    
    model_kwargs = {'dropout': args.dropout}
    if args.model in ['cross_attention', 'gated']:
        model_kwargs['hidden_dim'] = args.hidden_dim
    if args.model == 'bilinear':
        model_kwargs['rank'] = 64
    
    model = get_model(args.model, **model_kwargs)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    best_val_ci = 0
    best_epoch = 0
    history = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"LR: {args.lr}, Weight Decay: {args.weight_decay}")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_ci, _ = evaluate(model, train_loader, device)
        test_ci, test_loss = evaluate(model, test_loader, device)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ci': train_ci,
            'test_loss': test_loss,
            'test_ci': test_ci
        })
        
        if test_ci > best_val_ci:
            best_val_ci = test_ci
            best_epoch = epoch
            # Save best model
            save_path = os.path.join(args.output_dir, f'{args.model}_best.pt')
            torch.save(model.state_dict(), save_path)
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Train CI: {train_ci:.4f} | Test CI: {test_ci:.4f}")
    
    # Save last model
    save_path = os.path.join(args.output_dir, f'{args.model}_last.pt')
    torch.save(model.state_dict(), save_path)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.output_dir, f'{args.model}_history.csv'), index=False)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'{args.model}_best.pt')))
    final_train_ci, _ = evaluate(model, train_loader, device)
    final_test_ci, _ = evaluate(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE: {args.model.upper()}")
    print("=" * 60)
    print(f"Best Epoch: {best_epoch}")
    print(f"Final Train C-Index: {final_train_ci:.4f}")
    print(f"Final Test C-Index:  {final_test_ci:.4f}")
    print(f"Model saved to: {args.output_dir}/{args.model}_best.pt")
    
    # Save summary
    summary = {
        'model': args.model,
        'best_epoch': best_epoch,
        'train_ci': final_train_ci,
        'test_ci': final_test_ci,
        'n_params': n_params,
        'epochs': args.epochs,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(args.output_dir, f'{args.model}_summary.txt')
    with open(summary_path, 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    
    return final_test_ci


if __name__ == '__main__':
    main()
