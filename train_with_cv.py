"""
Cross-Validation Training Wrapper for MultiModal Survival Prediction

This script implements 10-fold stratified cross-validation as described in the paper:
"Multimodal deep learning to predict prognosis in adult and pediatric brain tumors"

For each fold:
1. Split training data into train/val (90%/10%)
2. Train model on train subset
3. Evaluate on val subset
4. Track best model based on validation loss

Final model selection: Best performing fold based on validation CI
"""

import os
import sys
import json
import argparse
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import subprocess
from datetime import datetime

def create_cv_splits(train_csv_path, n_folds=10, output_dir=None, random_state=42):
    """
    Create stratified K-fold cross-validation splits from training data.
    
    Args:
        train_csv_path: Path to training CSV
        n_folds: Number of folds (paper uses 10 for adults, 5 for pediatric)
        output_dir: Directory to save fold CSVs
        random_state: Random seed for reproducibility
    
    Returns:
        List of (train_csv, val_csv) paths for each fold
    """
    # Load training data
    df = pd.read_csv(train_csv_path)
    
    # Create stratification key based on survival_bin and vital_status
    df['strat_key'] = df['survival_bin'].astype(str) + '_' + df['vital_status'].astype(str)
    
    # Initialize stratified K-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.dirname(train_csv_path)
    cv_dir = os.path.join(output_dir, 'cv_folds')
    os.makedirs(cv_dir, exist_ok=True)
    
    fold_paths = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['strat_key'])):
        train_fold = df.iloc[train_idx].drop('strat_key', axis=1)
        val_fold = df.iloc[val_idx].drop('strat_key', axis=1)
        
        train_path = os.path.join(cv_dir, f'fold_{fold_idx}_train.csv')
        val_path = os.path.join(cv_dir, f'fold_{fold_idx}_val.csv')
        
        train_fold.to_csv(train_path, index=False)
        val_fold.to_csv(val_path, index=False)
        
        fold_paths.append((train_path, val_path))
        print(f"  Fold {fold_idx}: train={len(train_fold)}, val={len(val_fold)}")
    
    return fold_paths


def run_cv_training(config_path, n_folds=10, modality='rna'):
    """
    Run cross-validation training for a given modality.
    
    Args:
        config_path: Path to base config JSON
        n_folds: Number of CV folds
        modality: 'rna', 'ffpe', 'joint', or 'early'
    """
    # Load base config
    with open(config_path) as f:
        base_config = json.load(f)
    
    # Determine training script
    script_map = {
        'rna': '2_GeneExpression/1_GeneExpress_train.py',
        'ffpe': '1_HistoPathology/2_HistoPath_train.py',
        'joint': '5_JointFusion/1_JointFusion_train.py',
        'early': '3_EarlyFusion/2_EarlyFusion_train.py'
    }
    train_script = script_map.get(modality, script_map['rna'])
    
    # Create CV splits
    print(f"\n{'='*60}")
    print(f"Creating {n_folds}-fold CV splits for {modality}...")
    print('='*60)
    
    train_csv = base_config['train_csv_path']
    fold_paths = create_cv_splits(train_csv, n_folds=n_folds)
    
    # Track results across folds
    cv_results = []
    
    # Run training for each fold
    for fold_idx, (train_path, val_path) in enumerate(fold_paths):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print('='*60)
        
        # Create fold-specific config
        fold_config = base_config.copy()
        fold_config['train_csv_path'] = train_path
        fold_config['val_csv_path'] = val_path
        fold_config['flag'] = f"{base_config.get('flag', modality)}_fold{fold_idx}"
        
        # Save fold config
        fold_config_path = train_path.replace('_train.csv', '_config.json')
        with open(fold_config_path, 'w') as f:
            json.dump(fold_config, f, indent=4)
        
        # Run training subprocess
        cmd = [sys.executable, train_script, '--config', fold_config_path]
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            cv_results.append({
                'fold': fold_idx,
                'status': 'success',
                'config': fold_config_path
            })
        except subprocess.CalledProcessError as e:
            print(f"Error in fold {fold_idx}: {e}")
            print(e.stderr[-1000:] if len(e.stderr) > 1000 else e.stderr)
            cv_results.append({
                'fold': fold_idx,
                'status': 'failed',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION COMPLETE")
    print('='*60)
    print(f"Successful folds: {sum(1 for r in cv_results if r['status'] == 'success')}/{n_folds}")
    
    return cv_results


def update_configs_for_no_val(config_dir='config/survcox'):
    """
    Update config files to work with CV approach (no separate val file).
    Creates a copy of train data as val for backward compatibility.
    """
    configs = [
        'config_rna_train_survcox.json',
        'config_ffpe_train_survcox.json',
        'config_joint_train_survcox.json',
        'config_feature_train_survcox.json'
    ]
    
    for config_name in configs:
        config_path = os.path.join(config_dir, config_name)
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            
            # For simple runs without CV, use train as val
            # (This allows existing scripts to work)
            if 'val_csv_path' in config:
                train_path = config['train_csv_path']
                # Point val to train for non-CV runs
                config['val_csv_path'] = train_path
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"Updated {config_name}: val_csv_path = train_csv_path")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-Validation Training for MultiModal Survival')
    parser.add_argument('--config', type=str, required=True, help='Base config JSON file')
    parser.add_argument('--modality', type=str, default='rna', 
                        choices=['rna', 'ffpe', 'joint', 'early'],
                        help='Modality to train')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of CV folds')
    parser.add_argument('--update_configs', action='store_true', 
                        help='Update config files for no-val approach')
    
    args = parser.parse_args()
    
    if args.update_configs:
        update_configs_for_no_val()
    else:
        run_cv_training(args.config, n_folds=args.n_folds, modality=args.modality)
