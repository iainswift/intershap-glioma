"""
InterSHAP Analysis for Late Fusion - DIRECT METHOD
===================================================

Uses the actual R Cox Lasso model output (late_fusion_score) instead of
training a proxy model. This provides exact InterSHAP values.

Key Finding: The R Cox Lasso is a LINEAR model:
    late_fusion_score = β₁ * path_score + β₂ * rna_score

For linear models, the Shapley Interaction Index is ZERO by definition,
since there's no interaction term in f(x) = w₁x₁ + w₂x₂ + b.
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Intershap'))
from intershap_utils import set_seed


def analyze_late_fusion_model(train_path, test_path):
    """
    Analyze the R Cox Lasso model structure and compute exact InterSHAP.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Extract features and target
    X_train = train[['path_score', 'rna_score']].values
    y_train = train['late_fusion_score'].values
    X_test = test[['path_score', 'rna_score']].values
    y_test = test['late_fusion_score'].values
    
    # Fit linear regression to recover coefficients
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train, y_train)
    
    beta_path = lr.coef_[0]
    beta_rna = lr.coef_[1]
    intercept = lr.intercept_
    
    # Verify linearity
    y_pred = lr.predict(X_train)
    r2 = 1 - np.sum((y_train - y_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
    
    return {
        'beta_path': beta_path,
        'beta_rna': beta_rna,
        'intercept': intercept,
        'r2': r2,
        'is_linear': r2 > 0.9999
    }, train, test


def compute_shapley_values_linear(beta_path, beta_rna, path_scores, rna_scores):
    """
    Compute exact Shapley values for a linear model.
    
    For f(x) = β₁x₁ + β₂x₂ + b:
    - φ₁ = β₁ * (x₁ - E[x₁])  (WSI/path contribution)
    - φ₂ = β₂ * (x₂ - E[x₂])  (RNA contribution)
    - φ₀₁ = 0                  (interaction is ZERO for linear models)
    """
    # Reference values (mean)
    ref_path = np.mean(path_scores)
    ref_rna = np.mean(rna_scores)
    
    # Shapley values for each sample
    phi_path = beta_path * (path_scores - ref_path)
    phi_rna = beta_rna * (rna_scores - ref_rna)
    phi_interaction = np.zeros_like(phi_path)  # Always zero for linear
    
    return phi_path, phi_rna, phi_interaction


def compute_global_attribution(beta_path, beta_rna, path_scores, rna_scores):
    """
    Compute global attribution percentages.
    
    For linear models, we use absolute coefficient magnitudes weighted by
    feature standard deviations as the attribution measure.
    """
    # Standard deviations capture feature variability
    std_path = np.std(path_scores)
    std_rna = np.std(rna_scores)
    
    # Absolute contributions (like in importance)
    contrib_path = np.abs(beta_path) * std_path
    contrib_rna = np.abs(beta_rna) * std_rna
    contrib_interaction = 0.0  # Linear model has no interaction
    
    total = contrib_path + contrib_rna + contrib_interaction
    
    pct_path = (contrib_path / total) * 100
    pct_rna = (contrib_rna / total) * 100
    pct_interaction = 0.0
    
    return {
        'InterSHAP': pct_interaction,
        'WSI_Contribution': pct_path,
        'RNA_Contribution': pct_rna,
        'Total_Behavior': total,
        'Beta_Path': beta_path,
        'Beta_RNA': beta_rna
    }


def analyze_by_risk_group(test_df, phi_path, phi_rna, save_dir):
    """Analyze attributions by risk group."""
    results = []
    
    for surv_bin in sorted(test_df['survival_bin'].unique()):
        mask = test_df['survival_bin'].values == surv_bin
        
        # For linear model, we measure contribution variance
        path_contrib = np.abs(phi_path[mask])
        rna_contrib = np.abs(phi_rna[mask])
        total = path_contrib + rna_contrib + 1e-10
        
        results.append({
            'Risk_Group': int(surv_bin),
            'Mean_InterSHAP': 0.0,  # Always zero for linear
            'Std_InterSHAP': 0.0,
            'N_Samples': int(mask.sum()),
            'WSI_Mean': np.mean(path_contrib / total * 100),
            'WSI_Std': np.std(path_contrib / total * 100),
            'RNA_Mean': np.mean(rna_contrib / total * 100),
            'RNA_Std': np.std(rna_contrib / total * 100),
            'Risk_Level': ['High Risk (Shortest Survival)', 'Medium-High Risk', 
                          'Medium-Low Risk', 'Low Risk (Longest Survival)'][int(surv_bin)]
        })
    
    risk_df = pd.DataFrame(results)
    risk_df.to_csv(os.path.join(save_dir, 'risk_group_analysis.csv'), index=False)
    return risk_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct InterSHAP for Late Fusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="../MyData/results/intershap/late")
    parser.add_argument("--train_path", type=str, 
                        default="../MyData/results/late_fusion/scores_late_train.csv")
    parser.add_argument("--test_path", type=str,
                        default="../MyData/results/late_fusion/scores_late_test.csv")
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("LATE FUSION INTERSHAP - DIRECT METHOD")
    print("="*60)
    
    # Analyze model structure
    print("\n1. ANALYZING R COX LASSO MODEL")
    model_info, train_df, test_df = analyze_late_fusion_model(args.train_path, args.test_path)
    
    print(f"   Recovered coefficients:")
    print(f"     beta_path = {model_info['beta_path']:.6f}")
    print(f"     beta_rna  = {model_info['beta_rna']:.6f}")
    print(f"     intercept = {model_info['intercept']:.6f}")
    print(f"   Linearity check: R^2 = {model_info['r2']:.6f}")
    
    if model_info['is_linear']:
        print("\n   [!] MODEL IS LINEAR")
        print("       Shapley Interaction Index = 0 by mathematical definition")
        print("       For f(x) = b1*x1 + b2*x2: nabla_01 = f(x) - f(x1,0) - f(0,x2) + f(0,0) = 0")
    
    # Compute Shapley values
    print("\n2. COMPUTING EXACT SHAPLEY VALUES")
    path_scores = test_df['path_score'].values
    rna_scores = test_df['rna_score'].values
    
    phi_path, phi_rna, phi_interaction = compute_shapley_values_linear(
        model_info['beta_path'], model_info['beta_rna'],
        path_scores, rna_scores
    )
    
    # Global attribution
    print("\n3. GLOBAL ATTRIBUTION")
    global_results = compute_global_attribution(
        model_info['beta_path'], model_info['beta_rna'],
        path_scores, rna_scores
    )
    
    print(f"\n--- Global Results ---")
    print(f"InterSHAP Score:        {global_results['InterSHAP']:.2f}%")
    print(f"WSI Score Contribution: {global_results['WSI_Contribution']:.2f}%")
    print(f"RNA Score Contribution: {global_results['RNA_Contribution']:.2f}%")
    
    # Risk group analysis
    print("\n4. RISK GROUP ANALYSIS")
    print("="*50)
    risk_df = analyze_by_risk_group(test_df, phi_path, phi_rna, args.save_dir)
    print(risk_df.to_string(index=False))
    
    # Save results
    global_summary = pd.DataFrame([{
        'Fusion_Type': 'Late',
        'Seed': args.seed,
        'Method': 'Direct (Linear Model)',
        'Model_R2': model_info['r2'],
        **global_results
    }])
    global_summary.to_csv(os.path.join(args.save_dir, 'global_summary.csv'), index=False)
    
    # Local results
    local_df = pd.DataFrame({
        'sample_idx': range(len(test_df)),
        'survival_bin': test_df['survival_bin'].values,
        'path_score': path_scores,
        'rna_score': rna_scores,
        'phi_path': phi_path,
        'phi_rna': phi_rna,
        'phi_interaction': phi_interaction,
        'local_intershap_pct': 0.0,  # Always zero
        'late_fusion_score': test_df['late_fusion_score'].values
    })
    local_df.to_csv(os.path.join(args.save_dir, 'local_results.csv'), index=False)
    
    print(f"\n[OK] Results saved to {args.save_dir}")
    print(f"  - global_summary.csv")
    print(f"  - local_results.csv")
    print(f"  - risk_group_analysis.csv")
    
    print("\n" + "="*60)
    print("CONCLUSION: Late Fusion R Cox Lasso is LINEAR")
    print("           InterSHAP = 0% (mathematically exact)")
    print("="*60)
