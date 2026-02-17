#!/bin/bash

# =======================================================
# Thesis Multimodal InterSHAP System
# =======================================================
# Usage: bash run_system.sh [--seeds "42 123 2024"] [--mask_type mean] [--skip_synthetic]
#
# This script implements the full InterSHAP analysis pipeline:
# 1. Synthetic Verification (HD-XOR) - Validates implementation
# 2. Multi-seed Robustness Analysis - All three fusion strategies
# 3. Results Aggregation - Cross-seed statistics and risk analysis
#
# Reference: Wenderoth et al. "Measuring Cross-Modal Interactions 
#            in Multimodal Models" AAAI-25
# =======================================================
set -e

# Parse arguments
SEEDS=(42 123 2024)
MASK_TYPE="mean"
SKIP_SYNTHETIC=false
SKIP_JOINT_EMBED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) SEEDS=($2); shift 2 ;;
        --mask_type) MASK_TYPE="$2"; shift 2 ;;
        --skip_synthetic) SKIP_SYNTHETIC=true; shift ;;
        --skip_embed) SKIP_JOINT_EMBED=true; shift ;;
        --help) 
            echo "Usage: bash run_system.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --seeds \"42 123 2024\"  Seeds for robustness analysis"
            echo "  --mask_type mean       Masking strategy (mean/zero/noise)"
            echo "  --skip_synthetic       Skip synthetic verification"
            echo "  --skip_embed           Use cached joint embeddings"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Directories
BASE_RESULTS="../MyData/results/intershap"
mkdir -p "$BASE_RESULTS"

echo "======================================================="
echo " InterSHAP Analysis for Multimodal Survival Prediction"
echo "======================================================="
echo ""
echo " Reference: Wenderoth et al. AAAI-25"
echo " Modalities: Histopathology (WSI) + Gene Expression (RNA)"
echo " Task: 4-class Survival Prediction"
echo ""
echo " Configuration:"
echo "   Seeds:     ${SEEDS[*]}"
echo "   Mask Type: $MASK_TYPE"
echo "   Results:   $BASE_RESULTS"
echo ""

# =======================================================
# STEP 1: SYNTHETIC VERIFICATION (HD-XOR)
# =======================================================
# Before running on real clinical data, we MUST verify that
# InterSHAP correctly detects known interactions in synthetic
# data with ground truth.
#
# Expected Results:
#   Uniqueness: InterSHAP ≈ 0%   (all info in one modality)
#   Synergy:    InterSHAP ≈ 100% (XOR requires both)
#   Redundancy: InterSHAP ≈ 30-50%

if [ "$SKIP_SYNTHETIC" = false ]; then
    echo ""
    echo "======================================================="
    echo " STEP 1: SYNTHETIC VERIFICATION (HD-XOR)"
    echo "======================================================="
    echo ""
    echo "Testing InterSHAP on synthetic data with KNOWN ground truth..."
    echo "This validates the implementation before clinical application."
    echo ""
    
    python synthetic_verification.py \
        --seed 42 \
        --n_samples 5000 \
        --n_features 100 \
        --save_dir "$BASE_RESULTS/synthetic" \
        --settings uniqueness synergy redundancy
    
    SYNTH_EXIT=$?
    
    if [ $SYNTH_EXIT -ne 0 ]; then
        echo ""
        echo "======================================================="
        echo " ✗ SYNTHETIC VERIFICATION FAILED!"
        echo "======================================================="
        echo " InterSHAP did not detect expected interactions."
        echo " Please check the implementation before proceeding."
        echo ""
        echo " To skip this check (not recommended), use:"
        echo "   bash run_system.sh --skip_synthetic"
        echo "======================================================="
        exit 1
    fi
    
    echo ""
    echo "✓ Synthetic verification passed! Proceeding to clinical data..."
    echo ""
else
    echo ""
    echo "⚠ Skipping synthetic verification (--skip_synthetic)"
    echo ""
fi

# =======================================================
# STEP 2: MULTI-SEED ROBUSTNESS ANALYSIS
# =======================================================
# Run InterSHAP analysis on all three fusion strategies
# with multiple seeds to ensure stability.

echo ""
echo "======================================================="
echo " STEP 2: MULTI-SEED ROBUSTNESS ANALYSIS"
echo "======================================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "======================================================="
    echo " SEED: $seed"
    echo "======================================================="
    
    # A. EARLY FUSION
    # -------------------------------------------------------
    # Concatenated 2048-dim WSI + 2048-dim RNA features
    # Input: 4096-dim vector → MLP → 4 classes
    echo ""
    echo "--- [1/3] Early Fusion ---"
    echo "Architecture: Concat(WSI[2048], RNA[2048]) → MLP → 4 classes"
    
    python ../3_EarlyFusion/apply_intershap.py \
        --seed $seed \
        --save_dir "$BASE_RESULTS/early_seed_$seed" \
        --checkpoint "../MyData/results/early_fusion/checkpoints/models/earlyfusion_model_survcox_fold0/model_dict_best.pt" \
        --data_path "../MyData/splits/early_test.csv" \
        --mask_type "$MASK_TYPE"
        
    # B. LATE FUSION
    # -------------------------------------------------------
    # Combines scalar risk scores via proxy model
    # Original: R Cox Lasso → Must use PyTorch proxy
    echo ""
    echo "--- [2/3] Late Fusion ---"
    echo "Architecture: Proxy MLP mimicking R Cox Lasso"
    echo "Note: Fidelity check ensures proxy accuracy"
    
    python ../4_LateFusion/apply_intershap.py \
        --seed $seed \
        --save_dir "$BASE_RESULTS/late_seed_$seed" \
        --train_path "../MyData/results/late_fusion/combined_scores_train.csv" \
        --test_path "../MyData/results/late_fusion/combined_scores_test.csv" \
        --mask_type "$MASK_TYPE"
        
    # C. JOINT FUSION
    # -------------------------------------------------------
    # End-to-end: ResNet50 + MLP with fusion head
    # Strategy: Pre-compute embeddings, analyze fusion head only
    echo ""
    echo "--- [3/3] Joint Fusion ---"
    echo "Architecture: ResNet50(WSI) + MLP(RNA) → Fusion Head → 4 classes"
    echo "Strategy: Pre-computed embeddings + Head Analysis"
    
    JOINT_ARGS="--seed $seed --save_dir $BASE_RESULTS/joint_seed_$seed"
    JOINT_ARGS="$JOINT_ARGS --checkpoint ../MyData/results/joint_fusion/checkpoints/models/jointfusion_model_survcox_fold0/model_dict_best.pt"
    JOINT_ARGS="$JOINT_ARGS --data_path ../MyData/splits/joint_test.csv"
    JOINT_ARGS="$JOINT_ARGS --mask_type $MASK_TYPE"
    
    # Use cached embeddings after first run to save compute
    if [ "$SKIP_JOINT_EMBED" = true ] || [ -f "$BASE_RESULTS/joint_seed_$seed/embeddings_cache.npz" ]; then
        JOINT_ARGS="$JOINT_ARGS --use_cached"
    fi
    
    python ../5_JointFusion/apply_intershap.py $JOINT_ARGS
done

# =======================================================
# STEP 3: AGGREGATE RESULTS
# =======================================================
echo ""
echo "======================================================="
echo " STEP 3: AGGREGATE RESULTS"
echo "======================================================="

python -c "
import pandas as pd
import glob
import numpy as np
import os

BASE_RESULTS = '$BASE_RESULTS'

# Gather all global summaries
files = glob.glob(os.path.join(BASE_RESULTS, '*/global_summary.csv'))
if not files:
    print('No results found!')
    exit(1)

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Calculate statistics per fusion type
print()
print('=' * 60)
print(' SUMMARY STATISTICS (Mean ± Std across seeds)')
print('=' * 60)
print()

summary_rows = []
for fusion in ['Early', 'Late', 'Joint']:
    fdf = df[df['Fusion_Type'] == fusion]
    if len(fdf) == 0:
        continue
    
    row = {
        'Fusion': fusion,
        'n_seeds': len(fdf),
        'InterSHAP': f\"{fdf['InterSHAP'].mean():.1f}% ± {fdf['InterSHAP'].std():.1f}%\",
        'WSI': f\"{fdf['WSI_Contribution'].mean():.1f}% ± {fdf['WSI_Contribution'].std():.1f}%\",
        'RNA': f\"{fdf['RNA_Contribution'].mean():.1f}% ± {fdf['RNA_Contribution'].std():.1f}%\"
    }
    summary_rows.append(row)
    
    print(f'{fusion} Fusion (n={len(fdf)} seeds):')
    print(f'  InterSHAP (Synergy):  {fdf[\"InterSHAP\"].mean():.2f}% ± {fdf[\"InterSHAP\"].std():.2f}%')
    print(f'  WSI Contribution:     {fdf[\"WSI_Contribution\"].mean():.2f}% ± {fdf[\"WSI_Contribution\"].std():.2f}%')
    print(f'  RNA Contribution:     {fdf[\"RNA_Contribution\"].mean():.2f}% ± {fdf[\"RNA_Contribution\"].std():.2f}%')
    print()

# Save summary table
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(BASE_RESULTS, 'summary_table.csv'), index=False)

# Save detailed aggregated results
agg = df.groupby('Fusion_Type').agg({
    'InterSHAP': ['mean', 'std', 'min', 'max'],
    'WSI_Contribution': ['mean', 'std'],
    'RNA_Contribution': ['mean', 'std'],
    'Total_Behavior': ['mean', 'std']
}).round(4)

agg.to_csv(os.path.join(BASE_RESULTS, 'aggregated_detailed.csv'))

# Risk Group Analysis
print('=' * 60)
print(' RISK GROUP ANALYSIS (Clinical Stratification)')
print('=' * 60)
print()
print('Question: Does synergy differ between risk groups?')
print('(Bin 0 = High Risk, Bin 3 = Low Risk)')
print()

risk_files = glob.glob(os.path.join(BASE_RESULTS, '*/risk_group_analysis.csv'))
if risk_files:
    risk_dfs = []
    for f in risk_files:
        rdf = pd.read_csv(f)
        # Extract fusion type from path
        if 'early' in f.lower():
            rdf['Fusion'] = 'Early'
        elif 'late' in f.lower():
            rdf['Fusion'] = 'Late'
        elif 'joint' in f.lower():
            rdf['Fusion'] = 'Joint'
        risk_dfs.append(rdf)
    
    risk_df = pd.concat(risk_dfs, ignore_index=True)
    
    # Aggregate by fusion and risk group
    risk_agg = risk_df.groupby(['Fusion', 'Risk_Group']).agg({
        'Mean_InterSHAP': ['mean', 'std'],
        'N_Samples': 'sum'
    }).round(2)
    
    print(risk_agg.to_string())
    print()
    
    risk_agg.to_csv(os.path.join(BASE_RESULTS, 'risk_group_aggregated.csv'))
    
    # Clinical interpretation
    print()
    print('Clinical Interpretation:')
    for fusion in ['Early', 'Late', 'Joint']:
        fusion_risk = risk_df[risk_df['Fusion'] == fusion]
        if len(fusion_risk) == 0:
            continue
        
        high_risk = fusion_risk[fusion_risk['Risk_Group'] == 0]['Mean_InterSHAP'].mean()
        low_risk = fusion_risk[fusion_risk['Risk_Group'] == 3]['Mean_InterSHAP'].mean()
        
        if not np.isnan(high_risk) and not np.isnan(low_risk):
            diff = high_risk - low_risk
            if abs(diff) > 5:
                direction = 'MORE' if diff > 0 else 'LESS'
                print(f'  {fusion}: High-risk patients show {direction} synergy ({high_risk:.1f}% vs {low_risk:.1f}%)')
            else:
                print(f'  {fusion}: Synergy similar across risk groups ({high_risk:.1f}% vs {low_risk:.1f}%)')

print()
print('✓ Results saved to: $BASE_RESULTS')
"

echo ""
echo "======================================================="
echo " ANALYSIS COMPLETE"
echo "======================================================="
echo ""
echo "Results saved to: $BASE_RESULTS"
echo ""
echo "Output Files:"
echo "  - synthetic/synthetic_results.csv     (verification results)"
echo "  - summary_table.csv                   (main results table)"
echo "  - aggregated_detailed.csv             (full statistics)"
echo "  - risk_group_aggregated.csv           (clinical stratification)"
echo "  - */global_summary.csv                (per-run global results)"
echo "  - */local_results.csv                 (per-sample InterSHAP)"
echo "  - */risk_group_analysis.csv           (per-run risk analysis)"
echo ""
echo "Interpretation Guide:"
echo "  InterSHAP ≈ 0%:   No cross-modal interaction (single modality sufficient)"
echo "  InterSHAP ≈ 50%:  Moderate interaction (both contribute, some synergy)"
echo "  InterSHAP ≈ 100%: Pure synergy (prediction requires BOTH modalities)"
echo ""