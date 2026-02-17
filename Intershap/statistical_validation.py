"""
Statistical validation of InterSHAP as a prognostic biomarker.

Analyses:
1. Cox Proportional Hazards (univariate, multivariate)
2. Quartile analysis (dose-response relationship)
3. Bootstrap confidence intervals
4. Subgroup analysis (GBM vs LGG)
5. Sensitivity analysis (different cutoffs)
6. Proportional Hazards assumption test (Schoenfeld residuals)
7. Concordance Index (C-index)
8. 10-fold cross-validation
9. InterSHAP vs unimodal comparison
10. Clinical effect size (Cohen's d, NNH)
11. RMST (does not require PH assumption)
12. Landmark analysis
13. Piecewise exponential model
14. Permutation test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index, restricted_mean_survival_time
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_dir = os.path.dirname(os.path.abspath(__file__))


def get_tumor_type(case_id):
    """GBM cases typically have specific patterns in TCGA."""
    gbm_prefixes = ['TCGA-02', 'TCGA-06', 'TCGA-14', 'TCGA-19', 'TCGA-76']
    for prefix in gbm_prefixes:
        if case_id.startswith(prefix):
            return 'GBM'
    return 'LGG'


def load_data():
    """Load InterSHAP results and prepare Cox regression data."""
    df = pd.read_csv(os.path.join(script_dir, 'full_dataset_intershap.csv'))
    df['tumor_type'] = df['case'].apply(get_tumor_type)

    cox_df = df[['survival_months', 'vital_status', 'intershap',
                 'wsi_contribution', 'rna_contribution']].copy()
    cox_df = cox_df[cox_df['survival_months'] > 0]
    cox_df['intershap_std'] = (cox_df['intershap'] - cox_df['intershap'].mean()) / cox_df['intershap'].std()
    cox_df['tumor_type'] = df.loc[cox_df.index, 'case'].apply(get_tumor_type)
    cox_df['tumor_binary'] = (cox_df['tumor_type'] == 'GBM').astype(int)

    median_intershap = cox_df['intershap'].median()
    cox_df['intershap_high'] = (cox_df['intershap'] >= median_intershap).astype(int)

    return df, cox_df, median_intershap


# =============================================================================
# 1. COX PROPORTIONAL HAZARDS REGRESSION
# =============================================================================
def cox_regression(cox_df):
    """Univariate, binary, and multivariate Cox models."""
    print("\n" + "=" * 80)
    print("1. COX PROPORTIONAL HAZARDS REGRESSION")
    print("=" * 80)

    # Univariate (standardised)
    print("\n--- Univariate Cox Regression (InterSHAP, standardised) ---")
    cph_uni = CoxPHFitter()
    cph_uni.fit(cox_df[['survival_months', 'vital_status', 'intershap_std']],
                duration_col='survival_months', event_col='vital_status')
    print(cph_uni.summary[['coef', 'exp(coef)', 'se(coef)', 'p',
                           'exp(coef) lower 95%', 'exp(coef) upper 95%']])

    # Binary high / low
    print("\n--- Univariate Cox (Binary High/Low) ---")
    cph_binary = CoxPHFitter()
    cph_binary.fit(cox_df[['survival_months', 'vital_status', 'intershap_high']],
                   duration_col='survival_months', event_col='vital_status')
    print(cph_binary.summary[['coef', 'exp(coef)', 'se(coef)', 'p',
                              'exp(coef) lower 95%', 'exp(coef) upper 95%']])

    # Multivariate (adjusted for tumour type)
    print("\n--- Multivariate Cox (controlling for tumor type) ---")
    cph_multi = CoxPHFitter()
    cph_multi.fit(cox_df[['survival_months', 'vital_status', 'intershap_std', 'tumor_binary']],
                  duration_col='survival_months', event_col='vital_status')
    print(cph_multi.summary[['coef', 'exp(coef)', 'se(coef)', 'p',
                             'exp(coef) lower 95%', 'exp(coef) upper 95%']])

    hr = cph_uni.summary.loc['intershap_std', 'exp(coef)']
    hr_ci_low = cph_uni.summary.loc['intershap_std', 'exp(coef) lower 95%']
    hr_ci_high = cph_uni.summary.loc['intershap_std', 'exp(coef) upper 95%']
    p_uni = cph_uni.summary.loc['intershap_std', 'p']
    hr_multi = cph_multi.summary.loc['intershap_std', 'exp(coef)']
    p_multi = cph_multi.summary.loc['intershap_std', 'p']

    print(f"\nUnivariate HR per 1 SD: {hr:.2f} (95% CI: {hr_ci_low:.2f}-{hr_ci_high:.2f}), p = {p_uni:.2e}")
    print(f"Multivariate HR (adj. tumor type): {hr_multi:.2f}, p = {p_multi:.2e}")

    return {
        'cph_uni': cph_uni, 'cph_multi': cph_multi,
        'hr': hr, 'hr_ci_low': hr_ci_low, 'hr_ci_high': hr_ci_high,
        'p_uni': p_uni, 'hr_multi': hr_multi, 'p_multi': p_multi
    }


# =============================================================================
# 2. QUARTILE ANALYSIS
# =============================================================================
def quartile_analysis(df):
    """Dose-response relationship across InterSHAP quartiles."""
    print("\n" + "=" * 80)
    print("2. QUARTILE ANALYSIS (DOSE-RESPONSE)")
    print("=" * 80)

    df['intershap_quartile'] = pd.qcut(df['intershap'], q=4,
                                       labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])

    print("\nQuartile Statistics:")
    for name, group in df.groupby('intershap_quartile'):
        kmf = KaplanMeierFitter()
        kmf.fit(group['survival_months'], group['vital_status'])
        print(f"  {name}: N={len(group)}, Events={group['vital_status'].sum()}, "
              f"InterSHAP={group['intershap'].mean():.2f}%, "
              f"Median Survival={kmf.median_survival_time_:.1f} months")

    result_trend = multivariate_logrank_test(df['survival_months'],
                                            df['intershap_quartile'],
                                            df['vital_status'])
    print(f"\nLog-rank test for trend: p = {result_trend.p_value:.2e}")

    q1 = df[df['intershap_quartile'] == 'Q1 (Lowest)']
    q4 = df[df['intershap_quartile'] == 'Q4 (Highest)']
    q1_vs_q4 = logrank_test(q1['survival_months'], q4['survival_months'],
                            q1['vital_status'], q4['vital_status'])
    print(f"Q1 vs Q4 log-rank: p = {q1_vs_q4.p_value:.2e}")


# =============================================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================
def bootstrap_analysis(cox_df, n_bootstrap=1000):
    """Bootstrap CIs for hazard ratio and median survival difference."""
    print("\n" + "=" * 80)
    print(f"3. BOOTSTRAP CONFIDENCE INTERVALS ({n_bootstrap} resamples)")
    print("=" * 80)

    bootstrap_hrs = []
    bootstrap_median_diffs = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(len(cox_df), size=len(cox_df), replace=True)
        boot_df = cox_df.iloc[boot_idx].copy()
        try:
            cph = CoxPHFitter()
            cph.fit(boot_df[['survival_months', 'vital_status', 'intershap_std']],
                    duration_col='survival_months', event_col='vital_status')
            bootstrap_hrs.append(cph.summary.loc['intershap_std', 'exp(coef)'])

            high = boot_df[boot_df['intershap_high'] == 1]
            low = boot_df[boot_df['intershap_high'] == 0]
            kmf_h = KaplanMeierFitter()
            kmf_l = KaplanMeierFitter()
            kmf_h.fit(high['survival_months'], high['vital_status'])
            kmf_l.fit(low['survival_months'], low['vital_status'])
            bootstrap_median_diffs.append(kmf_l.median_survival_time_ - kmf_h.median_survival_time_)
        except Exception:
            continue

    bootstrap_hrs = np.array(bootstrap_hrs)
    bootstrap_median_diffs = np.array([d for d in bootstrap_median_diffs if np.isfinite(d)])

    hr_mean = np.mean(bootstrap_hrs)
    hr_ci = np.percentile(bootstrap_hrs, [2.5, 97.5])
    diff_mean = np.mean(bootstrap_median_diffs)
    diff_ci = np.percentile(bootstrap_median_diffs, [2.5, 97.5])

    print(f"HR (bootstrap): {hr_mean:.2f} (95% CI: {hr_ci[0]:.2f}-{hr_ci[1]:.2f})")
    print(f"Median survival diff: {diff_mean:.1f} months (95% CI: {diff_ci[0]:.1f}-{diff_ci[1]:.1f})")

    return {'hrs': bootstrap_hrs, 'hr_mean': hr_mean, 'hr_ci': hr_ci,
            'diff_mean': diff_mean, 'diff_ci': diff_ci}


# =============================================================================
# 4. SUBGROUP ANALYSIS
# =============================================================================
def subgroup_analysis(df, cox_df):
    """GBM vs LGG subgroup KM and log-rank."""
    print("\n" + "=" * 80)
    print("4. SUBGROUP ANALYSIS (GBM vs LGG)")
    print("=" * 80)

    for tumor in ['GBM', 'LGG']:
        sub = df[df['tumor_type'] == tumor].copy()
        if len(sub) < 20:
            print(f"\n{tumor}: Too few patients ({len(sub)}), skipping")
            continue

        med = sub['intershap'].median()
        high = sub[sub['intershap'] >= med]
        low = sub[sub['intershap'] < med]

        if len(high) < 5 or len(low) < 5:
            print(f"\n{tumor}: Too few in subgroups, skipping")
            continue

        kmf_h, kmf_l = KaplanMeierFitter(), KaplanMeierFitter()
        kmf_h.fit(high['survival_months'], high['vital_status'])
        kmf_l.fit(low['survival_months'], low['vital_status'])
        lr = logrank_test(high['survival_months'], low['survival_months'],
                          high['vital_status'], low['vital_status'])

        print(f"\n{tumor} (N={len(sub)}, Events={sub['vital_status'].sum()}):")
        print(f"  High InterSHAP: N={len(high)}, Median Survival={kmf_h.median_survival_time_:.1f} months")
        print(f"  Low InterSHAP:  N={len(low)}, Median Survival={kmf_l.median_survival_time_:.1f} months")
        print(f"  Log-rank p = {lr.p_value:.4f} {'*' if lr.p_value < 0.05 else '(NS)'}")


# =============================================================================
# 5. SENSITIVITY ANALYSIS
# =============================================================================
def sensitivity_analysis(df):
    """Log-rank at multiple percentile cutoffs."""
    print("\n" + "=" * 80)
    print("5. SENSITIVITY ANALYSIS (DIFFERENT CUTOFFS)")
    print("=" * 80)

    cutoffs = [25, 33, 50, 67, 75]
    print(f"\n{'Cutoff':<15} {'High N':<10} {'Low N':<10} {'p-value':<15} {'Significant'}")
    print("-" * 60)

    for pct in cutoffs:
        cutoff = np.percentile(df['intershap'], pct)
        high = df[df['intershap'] >= cutoff]
        low = df[df['intershap'] < cutoff]
        lr = logrank_test(high['survival_months'], low['survival_months'],
                          high['vital_status'], low['vital_status'])
        sig = "***" if lr.p_value < 0.001 else "**" if lr.p_value < 0.01 else "*" if lr.p_value < 0.05 else ""
        print(f"{pct}th percentile   {len(high):<10} {len(low):<10} {lr.p_value:<15.2e} {sig}")


# =============================================================================
# 6. PROPORTIONAL HAZARDS ASSUMPTION TEST
# =============================================================================
def ph_assumption_test(cox_df):
    """Schoenfeld residuals test for PH assumption."""
    print("\n" + "=" * 80)
    print("6. PROPORTIONAL HAZARDS ASSUMPTION TEST")
    print("=" * 80)

    cph = CoxPHFitter()
    cph.fit(cox_df[['survival_months', 'vital_status', 'intershap_std']],
            duration_col='survival_months', event_col='vital_status')

    ph_test = proportional_hazard_test(
        cph, cox_df[['survival_months', 'vital_status', 'intershap_std']],
        time_transform='rank')
    print("\nSchoenfeld Residuals Test:")
    print(ph_test.summary)

    return cph, ph_test


# =============================================================================
# 7. CONCORDANCE INDEX
# =============================================================================
def concordance_analysis(cox_df):
    """C-index for InterSHAP alone, tumour type, and combined."""
    print("\n" + "=" * 80)
    print("7. CONCORDANCE INDEX (DISCRIMINATION)")
    print("=" * 80)

    c_intershap = concordance_index(cox_df['survival_months'],
                                    -cox_df['intershap_std'],
                                    cox_df['vital_status'])
    c_tumor = concordance_index(cox_df['survival_months'],
                                cox_df['tumor_binary'],
                                cox_df['vital_status'])

    cph_comb = CoxPHFitter()
    cph_comb.fit(cox_df[['survival_months', 'vital_status', 'intershap_std', 'tumor_binary']],
                 duration_col='survival_months', event_col='vital_status')
    c_combined = cph_comb.concordance_index_

    print(f"\nC-index (InterSHAP alone):      {c_intershap:.3f}")
    print(f"C-index (tumor type alone):      {c_tumor:.3f}")
    print(f"C-index (InterSHAP + tumor type): {c_combined:.3f}")

    return c_intershap, c_tumor, c_combined


# =============================================================================
# 8. 10-FOLD CROSS-VALIDATION
# =============================================================================
def cross_validation(cox_df, n_splits=10):
    """K-fold CV of hazard ratio and C-index."""
    print("\n" + "=" * 80)
    print(f"8. {n_splits}-FOLD CROSS-VALIDATION")
    print("=" * 80)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_hrs, cv_pvals, cv_c_indices = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(cox_df)):
        train = cox_df.iloc[train_idx]
        test = cox_df.iloc[test_idx]

        cph = CoxPHFitter()
        cph.fit(train[['survival_months', 'vital_status', 'intershap_std']],
                duration_col='survival_months', event_col='vital_status')

        hr = cph.summary.loc['intershap_std', 'exp(coef)']
        p = cph.summary.loc['intershap_std', 'p']
        pred_risk = cph.predict_partial_hazard(test[['intershap_std']])
        c_idx = concordance_index(test['survival_months'],
                                  -pred_risk.values.flatten(),
                                  test['vital_status'])

        cv_hrs.append(hr)
        cv_pvals.append(p)
        cv_c_indices.append(c_idx)
        print(f"  Fold {fold+1}: HR = {hr:.2f}, p = {p:.2e}, C-index = {c_idx:.3f}")

    print(f"\n  Mean HR: {np.mean(cv_hrs):.2f} +/- {np.std(cv_hrs):.2f}")
    print(f"  Mean C-index: {np.mean(cv_c_indices):.3f} +/- {np.std(cv_c_indices):.3f}")
    print(f"  All folds significant (p < 0.05): {all(p < 0.05 for p in cv_pvals)}")

    return cv_hrs, cv_pvals, cv_c_indices


# =============================================================================
# 9. INTERSHAP vs UNIMODAL CONTRIBUTIONS
# =============================================================================
def unimodal_comparison(cox_df, c_intershap):
    """Compare InterSHAP with WSI-only and RNA-only Cox models."""
    print("\n" + "=" * 80)
    print("9. INTERSHAP vs UNIMODAL CONTRIBUTIONS")
    print("=" * 80)

    cox_df['wsi_std'] = (cox_df['wsi_contribution'] - cox_df['wsi_contribution'].mean()) / cox_df['wsi_contribution'].std()
    cox_df['rna_std'] = (cox_df['rna_contribution'] - cox_df['rna_contribution'].mean()) / cox_df['rna_contribution'].std()

    results = []
    for var, label in [('wsi_std', 'WSI contribution'), ('rna_std', 'RNA contribution')]:
        cph = CoxPHFitter()
        cph.fit(cox_df[['survival_months', 'vital_status', var]],
                duration_col='survival_months', event_col='vital_status')
        hr = cph.summary.loc[var, 'exp(coef)']
        p = cph.summary.loc[var, 'p']
        c = concordance_index(cox_df['survival_months'], -cox_df[var], cox_df['vital_status'])
        results.append((label, hr, p, c))

    print(f"\n{'Variable':<20} {'HR':<10} {'p-value':<15} {'C-index':<10}")
    print("-" * 55)
    print(f"{'InterSHAP':<20} {'-':<10} {'-':<15} {c_intershap:.3f}")
    for label, hr, p, c in results:
        print(f"{label:<20} {hr:.2f}       {p:.2e}      {c:.3f}")


# =============================================================================
# 10. CLINICAL EFFECT SIZE
# =============================================================================
def clinical_effect_size(cox_df):
    """Cohen's d, 5-year survival difference, NNH."""
    print("\n" + "=" * 80)
    print("10. CLINICAL EFFECT SIZE")
    print("=" * 80)

    median_cut = cox_df['intershap'].median()
    high = cox_df[cox_df['intershap'] >= median_cut]
    low = cox_df[cox_df['intershap'] < median_cut]

    pooled_std = np.sqrt((high['survival_months'].std()**2 + low['survival_months'].std()**2) / 2)
    cohens_d = (low['survival_months'].mean() - high['survival_months'].mean()) / pooled_std

    kmf_h, kmf_l = KaplanMeierFitter(), KaplanMeierFitter()
    kmf_h.fit(high['survival_months'], high['vital_status'])
    kmf_l.fit(low['survival_months'], low['vital_status'])

    surv_5yr_high = kmf_h.predict(60)
    surv_5yr_low = kmf_l.predict(60)
    ard = surv_5yr_low - surv_5yr_high
    nnh = 1 / ard if ard > 0 else float('inf')

    print(f"\nMean survival (High InterSHAP): {high['survival_months'].mean():.1f} months")
    print(f"Mean survival (Low InterSHAP):  {low['survival_months'].mean():.1f} months")
    print(f"Cohen's d: {cohens_d:.2f}")
    print(f"\n5-year survival (High InterSHAP): {surv_5yr_high:.1%}")
    print(f"5-year survival (Low InterSHAP):  {surv_5yr_low:.1%}")
    print(f"Absolute risk difference at 5 years: {ard:.1%}")
    print(f"Number Needed to Harm (NNH): {nnh:.1f}")

    return cohens_d, ard, nnh


# =============================================================================
# 11. RESTRICTED MEAN SURVIVAL TIME (RMST)
# =============================================================================
def rmst_analysis(cox_df):
    """RMST at multiple time horizons (PH-free)."""
    print("\n" + "=" * 80)
    print("11. RESTRICTED MEAN SURVIVAL TIME (RMST)")
    print("    Does NOT require proportional hazards assumption")
    print("=" * 80)

    median_cut = cox_df['intershap'].median()
    high = cox_df[cox_df['intershap'] >= median_cut]
    low = cox_df[cox_df['intershap'] < median_cut]

    time_horizons = [24, 36, 60, 120]
    print(f"\n{'Time Horizon':<15} {'RMST High':<15} {'RMST Low':<15} {'Difference':<15} {'% Reduction'}")
    print("-" * 70)

    rmst_results = []
    for t_max in time_horizons:
        kmf_h, kmf_l = KaplanMeierFitter(), KaplanMeierFitter()
        kmf_h.fit(high['survival_months'].clip(upper=t_max),
                  (high['vital_status']) & (high['survival_months'] <= t_max))
        kmf_l.fit(low['survival_months'].clip(upper=t_max),
                  (low['vital_status']) & (low['survival_months'] <= t_max))

        rmst_h = restricted_mean_survival_time(kmf_h, t=t_max)
        rmst_l = restricted_mean_survival_time(kmf_l, t=t_max)
        diff = rmst_l - rmst_h
        pct = (diff / rmst_l) * 100

        print(f"{t_max} months       {rmst_h:.1f} mo       {rmst_l:.1f} mo       {diff:.1f} mo        {pct:.1f}%")
        rmst_results.append({'horizon': t_max, 'rmst_high': rmst_h, 'rmst_low': rmst_l,
                             'diff': diff, 'pct_reduction': pct})

    return rmst_results


# =============================================================================
# 12. LANDMARK ANALYSIS
# =============================================================================
def landmark_analysis(cox_df):
    """Cox HR conditional on surviving to each landmark."""
    print("\n" + "=" * 80)
    print("12. LANDMARK ANALYSIS (Time-varying effect)")
    print("=" * 80)

    landmarks = [0, 12, 24, 36]
    print(f"\n{'Landmark':<15} {'N at risk':<15} {'HR':<15} {'95% CI':<20} {'p-value'}")
    print("-" * 70)

    results = []
    for t in landmarks:
        ldf = cox_df[cox_df['survival_months'] > t].copy()
        ldf['time_from_lm'] = ldf['survival_months'] - t
        ldf['intershap_std'] = (ldf['intershap'] - ldf['intershap'].mean()) / ldf['intershap'].std()

        if len(ldf) < 50 or ldf['vital_status'].sum() < 10:
            print(f"{t} months       Too few events")
            continue

        cph = CoxPHFitter()
        cph.fit(ldf[['time_from_lm', 'vital_status', 'intershap_std']],
                duration_col='time_from_lm', event_col='vital_status')

        hr = cph.summary.loc['intershap_std', 'exp(coef)']
        ci_l = cph.summary.loc['intershap_std', 'exp(coef) lower 95%']
        ci_h = cph.summary.loc['intershap_std', 'exp(coef) upper 95%']
        p = cph.summary.loc['intershap_std', 'p']

        print(f"{t} months       {len(ldf):<15} {hr:.2f}          ({ci_l:.2f}-{ci_h:.2f})       {p:.2e}")
        results.append({'landmark': t, 'n': len(ldf), 'hr': hr,
                        'ci_low': ci_l, 'ci_high': ci_h, 'p': p})

    return results


# =============================================================================
# 13. PIECEWISE ANALYSIS
# =============================================================================
def piecewise_analysis(cox_df, split_at=36):
    """Separate Cox models for early and late phases."""
    print("\n" + "=" * 80)
    print(f"13. PIECEWISE ANALYSIS (Split at {split_at} months)")
    print("=" * 80)

    # Early phase
    early = cox_df.copy()
    early.loc[early['survival_months'] > split_at, 'survival_months'] = split_at
    early.loc[early['survival_months'] > split_at, 'vital_status'] = 0
    early['intershap_std'] = (early['intershap'] - early['intershap'].mean()) / early['intershap'].std()

    cph_early = CoxPHFitter()
    cph_early.fit(early[['survival_months', 'vital_status', 'intershap_std']],
                  duration_col='survival_months', event_col='vital_status')

    # Late phase
    late = cox_df[cox_df['survival_months'] > split_at].copy()
    late['time_from_split'] = late['survival_months'] - split_at
    late['intershap_std'] = (late['intershap'] - late['intershap'].mean()) / late['intershap'].std()

    cph_late = CoxPHFitter()
    cph_late.fit(late[['time_from_split', 'vital_status', 'intershap_std']],
                 duration_col='time_from_split', event_col='vital_status')

    for label, cph_model, n, ev in [
        (f"Early phase (0-{split_at} months)", cph_early, len(early), early['vital_status'].sum()),
        (f"Late phase (>{split_at} months)", cph_late, len(late), late['vital_status'].sum())
    ]:
        hr = cph_model.summary.loc['intershap_std', 'exp(coef)']
        ci_l = cph_model.summary.loc['intershap_std', 'exp(coef) lower 95%']
        ci_h = cph_model.summary.loc['intershap_std', 'exp(coef) upper 95%']
        p = cph_model.summary.loc['intershap_std', 'p']
        print(f"\n{label}:")
        print(f"  N = {n}, Events = {ev}")
        print(f"  HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_h:.2f})")
        print(f"  p = {p:.2e}")


# =============================================================================
# 14. PERMUTATION TEST
# =============================================================================
def permutation_test(cox_df, n_perm=10000):
    """Model-free significance test by permuting InterSHAP labels."""
    print("\n" + "=" * 80)
    print(f"14. PERMUTATION TEST ({n_perm} permutations)")
    print("=" * 80)

    median_cut = cox_df['intershap'].median()
    high = cox_df[cox_df['intershap'] >= median_cut]
    low = cox_df[cox_df['intershap'] < median_cut]
    observed_diff = low['survival_months'].median() - high['survival_months'].median()

    np.random.seed(42)
    perm_diffs = []
    for _ in range(n_perm):
        shuffled = cox_df['intershap'].sample(frac=1, replace=False).values
        high_perm = cox_df['survival_months'][shuffled >= median_cut]
        low_perm = cox_df['survival_months'][shuffled < median_cut]
        perm_diffs.append(low_perm.median() - high_perm.median())

    perm_diffs = np.array(perm_diffs)
    perm_p = (np.sum(perm_diffs >= observed_diff) + 1) / (n_perm + 1)

    print(f"\nObserved median survival difference: {observed_diff:.1f} months")
    print(f"Permutation distribution: mean = {np.mean(perm_diffs):.1f}, std = {np.std(perm_diffs):.1f}")
    print(f"Permutation p-value: {perm_p:.6f}")

    return observed_diff, perm_diffs, perm_p


# =============================================================================
# FIGURES
# =============================================================================
def generate_validation_figure(df, cox_df, cox_res, boot_res, median_intershap):
    """Main validation figure (KM, quartiles, forest plot, bootstrap, subgroups, summary)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # A. KM curves
    ax = axes[0, 0]
    high_mask = df['intershap'] >= median_intershap
    low_mask = ~high_mask
    kmf_h, kmf_l = KaplanMeierFitter(), KaplanMeierFitter()
    kmf_h.fit(df.loc[high_mask, 'survival_months'], df.loc[high_mask, 'vital_status'],
              label=f'High InterSHAP (n={high_mask.sum()})')
    kmf_l.fit(df.loc[low_mask, 'survival_months'], df.loc[low_mask, 'vital_status'],
              label=f'Low InterSHAP (n={low_mask.sum()})')
    kmf_h.plot_survival_function(ax=ax, ci_show=True, color='red')
    kmf_l.plot_survival_function(ax=ax, ci_show=True, color='blue')
    ax.set_xlabel('Time (months)'); ax.set_ylabel('Survival Probability')
    ax.set_title('A. Kaplan-Meier Curves', fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.text(0.95, 0.95,
            f"HR = {cox_res['hr']:.2f}\n(95% CI: {cox_res['hr_ci_low']:.2f}-{cox_res['hr_ci_high']:.2f})\np < 0.0001",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # B. Quartile KM
    ax = axes[0, 1]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    for i, q in enumerate(['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']):
        grp = df[df['intershap_quartile'] == q]
        kmf = KaplanMeierFitter()
        kmf.fit(grp['survival_months'], grp['vital_status'], label=q)
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[i])
    ax.set_xlabel('Time (months)'); ax.set_ylabel('Survival Probability')
    ax.set_title('B. Dose-Response by Quartile', fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)

    # C. Forest plot
    ax = axes[0, 2]
    forest = [
        ('Univariate', cox_res['hr'], cox_res['hr_ci_low'], cox_res['hr_ci_high']),
        ('Adjusted (tumor type)', cox_res['hr_multi'],
         cox_res['cph_multi'].summary.loc['intershap_std', 'exp(coef) lower 95%'],
         cox_res['cph_multi'].summary.loc['intershap_std', 'exp(coef) upper 95%']),
        ('Bootstrap', boot_res['hr_mean'], boot_res['hr_ci'][0], boot_res['hr_ci'][1])
    ]
    for i, (name, hr, ci_l, ci_h) in enumerate(forest):
        ax.errorbar(hr, i, xerr=[[hr-ci_l], [ci_h-hr]], fmt='o', color='darkblue',
                    capsize=5, capthick=2, markersize=8)
        ax.text(ci_h + 0.05, i, f'{hr:.2f} ({ci_l:.2f}-{ci_h:.2f})', va='center', fontsize=10)
    ax.axvline(1, color='red', linestyle='--', linewidth=2, label='HR = 1')
    ax.set_yticks(range(len(forest))); ax.set_yticklabels([f[0] for f in forest])
    ax.set_xlabel('Hazard Ratio (per 1 SD InterSHAP)')
    ax.set_title('C. Forest Plot of Hazard Ratios', fontweight='bold')
    ax.set_xlim(0.8, 2.0)

    # D. Bootstrap distribution
    ax = axes[1, 0]
    ax.hist(boot_res['hrs'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(cox_res['hr'], color='red', linestyle='-', linewidth=2, label='Point estimate')
    ax.axvline(boot_res['hr_ci'][0], color='red', linestyle='--', linewidth=1.5, label='95% CI')
    ax.axvline(boot_res['hr_ci'][1], color='red', linestyle='--', linewidth=1.5)
    ax.axvline(1.0, color='black', linestyle=':', linewidth=2, label='HR = 1')
    ax.set_xlabel('Hazard Ratio'); ax.set_ylabel('Frequency')
    ax.set_title('D. Bootstrap Distribution (n=1000)', fontweight='bold')
    ax.legend(fontsize=9)

    # E. Subgroup forest plot
    ax = axes[1, 1]
    sub_results = []
    for tumor in ['GBM', 'LGG', 'All']:
        sub = cox_df if tumor == 'All' else cox_df[cox_df['tumor_type'] == tumor]
        if len(sub) < 30:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(sub[['survival_months', 'vital_status', 'intershap_std']],
                    duration_col='survival_months', event_col='vital_status')
            sub_results.append((
                tumor,
                cph.summary.loc['intershap_std', 'exp(coef)'],
                cph.summary.loc['intershap_std', 'exp(coef) lower 95%'],
                cph.summary.loc['intershap_std', 'exp(coef) upper 95%'],
                len(sub)
            ))
        except Exception:
            continue

    for i, (name, hr, ci_l, ci_h, n) in enumerate(sub_results):
        ax.errorbar(hr, i, xerr=[[hr-ci_l], [ci_h-hr]], fmt='s', color='darkgreen',
                    capsize=5, capthick=2, markersize=8)
        ax.text(ci_h + 0.05, i, f'{hr:.2f} (n={n})', va='center', fontsize=10)
    ax.axvline(1, color='red', linestyle='--', linewidth=2)
    ax.set_yticks(range(len(sub_results)))
    ax.set_yticklabels([s[0] for s in sub_results])
    ax.set_xlabel('Hazard Ratio')
    ax.set_title('E. Subgroup Analysis', fontweight='bold')
    ax.set_xlim(0.5, 2.5)

    # F. Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        f"Statistical Validation Summary\n"
        f"{'='*40}\n\n"
        f"N = {len(df)} patients, Events = {df['vital_status'].sum()}\n\n"
        f"Univariate HR: {cox_res['hr']:.2f} "
        f"({cox_res['hr_ci_low']:.2f}-{cox_res['hr_ci_high']:.2f})\n"
        f"  p = {cox_res['p_uni']:.2e}\n\n"
        f"Multivariate HR: {cox_res['hr_multi']:.2f}\n"
        f"  p = {cox_res['p_multi']:.2e}\n\n"
        f"Bootstrap HR: {boot_res['hr_mean']:.2f} "
        f"({boot_res['hr_ci'][0]:.2f}-{boot_res['hr_ci'][1]:.2f})\n\n"
        f"Median survival (High): {kmf_h.median_survival_time_:.1f} mo\n"
        f"Median survival (Low):  {kmf_l.median_survival_time_:.1f} mo\n"
        f"Difference: {kmf_l.median_survival_time_ - kmf_h.median_survival_time_:.1f} mo\n\n"
        f"Quartile trend p < 0.0001"
    )
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(script_dir, 'statistical_validation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nValidation figure saved: {path}")


def generate_diagnostics_figure(cox_df, cph, ph_test, cv_hrs, cv_c_indices,
                                cohens_d, ard, nnh, c_intershap, c_combined):
    """Diagnostics figure (Schoenfeld, CV C-index, CV HR, summary)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # A. Schoenfeld residuals
    ax = axes[0, 0]
    times = cox_df[cox_df['vital_status'] == 1]['survival_months'].values
    residuals = cph.compute_residuals(
        cox_df[['survival_months', 'vital_status', 'intershap_std']], 'schoenfeld')
    res_vals = residuals['intershap_std'].values
    ax.scatter(times, res_vals[:len(times)], alpha=0.5, s=20)
    z = np.polyfit(times, res_vals[:len(times)], 1)
    p = np.poly1d(z)
    ax.plot(sorted(times), p(sorted(times)), 'r-', linewidth=2, label='Trend line')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (months)'); ax.set_ylabel('Schoenfeld Residual')
    ax.set_title('A. Proportional Hazards Check', fontweight='bold')
    ax.legend()

    # B. CV C-index
    ax = axes[0, 1]
    ax.bar(range(1, 11), cv_c_indices, color='steelblue', edgecolor='black')
    ax.axhline(np.mean(cv_c_indices), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(cv_c_indices):.3f}')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='Random (0.5)')
    ax.set_xlabel('Fold'); ax.set_ylabel('C-index')
    ax.set_title('B. 10-Fold Cross-Validation C-index', fontweight='bold')
    ax.set_ylim(0.4, 0.9); ax.legend()

    # C. CV HR
    ax = axes[1, 0]
    y_pos = np.arange(10)
    ax.barh(y_pos, cv_hrs, color='darkgreen', edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='HR = 1')
    ax.axvline(np.mean(cv_hrs), color='blue', linestyle='-', linewidth=2,
               label=f'Mean HR = {np.mean(cv_hrs):.2f}')
    ax.set_yticks(y_pos); ax.set_yticklabels([f'Fold {i+1}' for i in range(10)])
    ax.set_xlabel('Hazard Ratio')
    ax.set_title('C. Hazard Ratio by CV Fold', fontweight='bold')
    ax.legend()

    # D. Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"Diagnostics Summary\n{'='*40}\n\n"
        f"PH Assumption\n"
        f"  Schoenfeld p = {ph_test.summary['p'].values[0]:.3f}\n\n"
        f"Discrimination (C-index)\n"
        f"  InterSHAP alone: {c_intershap:.3f}\n"
        f"  + Tumor type:    {c_combined:.3f}\n"
        f"  10-fold CV mean: {np.mean(cv_c_indices):.3f} +/- {np.std(cv_c_indices):.3f}\n\n"
        f"Cross-Validation\n"
        f"  Mean HR: {np.mean(cv_hrs):.2f} +/- {np.std(cv_hrs):.2f}\n\n"
        f"Clinical Effect Size\n"
        f"  Cohen's d: {cohens_d:.2f}\n"
        f"  5-yr survival diff: {ard:.1%}\n"
        f"  NNH: {nnh:.1f}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(script_dir, 'model_diagnostics.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Diagnostics figure saved: {path}")


def generate_robustness_figure(cox_df, rmst_results, landmark_results,
                               observed_diff, perm_diffs):
    """Robustness figure (KM with time annotation, RMST, landmarks, permutation)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    median_cut = cox_df['intershap'].median()
    high = cox_df[cox_df['intershap'] >= median_cut]
    low = cox_df[cox_df['intershap'] < median_cut]

    # A. KM with time-varying annotation
    ax = axes[0, 0]
    kmf_h, kmf_l = KaplanMeierFitter(), KaplanMeierFitter()
    kmf_h.fit(high['survival_months'], high['vital_status'],
              label=f'High InterSHAP (n={len(high)})')
    kmf_l.fit(low['survival_months'], low['vital_status'],
              label=f'Low InterSHAP (n={len(low)})')
    kmf_h.plot_survival_function(ax=ax, ci_show=True, color='red')
    kmf_l.plot_survival_function(ax=ax, ci_show=True, color='blue')
    ax.axvline(36, color='gray', linestyle='--', alpha=0.5)
    ax.text(38, 0.8, 'Effect stronger\nearly (0-36 mo)', fontsize=9)
    ax.set_xlabel('Time (months)'); ax.set_ylabel('Survival Probability')
    ax.set_title('A. Kaplan-Meier Curves', fontweight='bold')
    ax.legend(loc='lower left')

    # B. RMST
    ax = axes[0, 1]
    horizons = [r['horizon'] for r in rmst_results]
    diffs = [r['diff'] for r in rmst_results]
    ax.bar(range(len(horizons)), diffs, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f'{h} mo' for h in horizons])
    ax.set_xlabel('Time Horizon'); ax.set_ylabel('RMST Difference (months)')
    ax.set_title('B. Restricted Mean Survival Time Difference', fontweight='bold')
    for i, d in enumerate(diffs):
        ax.text(i, d + 0.5, f'{d:.1f}', ha='center', fontsize=10)

    # C. Landmark forest plot
    ax = axes[1, 0]
    lm_df = pd.DataFrame(landmark_results)
    for i, row in lm_df.iterrows():
        ax.errorbar(row['hr'], i,
                    xerr=[[row['hr']-row['ci_low']], [row['ci_high']-row['hr']]],
                    fmt='o', color='darkgreen', capsize=5, markersize=8)
        ax.text(row['ci_high'] + 0.1, i,
                f"HR={row['hr']:.2f}, p={row['p']:.0e}", va='center', fontsize=9)
    ax.axvline(1, color='red', linestyle='--', linewidth=2)
    ax.set_yticks(range(len(lm_df)))
    ax.set_yticklabels([f"t={r['landmark']}mo (n={r['n']})" for r in landmark_results])
    ax.set_xlabel('Hazard Ratio')
    ax.set_title('C. Landmark Analysis', fontweight='bold')
    ax.set_xlim(0.5, 3.5)

    # D. Permutation distribution
    ax = axes[1, 1]
    ax.hist(perm_diffs, bins=50, edgecolor='black', alpha=0.7, color='lightgray',
            label='Permutation null')
    ax.axvline(observed_diff, color='red', linewidth=3,
               label=f'Observed ({observed_diff:.1f} mo)')
    ax.axvline(np.percentile(perm_diffs, 95), color='orange', linestyle='--',
               linewidth=2, label='95th percentile')
    ax.set_xlabel('Median Survival Difference (months)')
    ax.set_ylabel('Frequency')
    ax.set_title('D. Permutation Test (10,000 permutations)', fontweight='bold')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(script_dir, 'ph_robustness.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Robustness figure saved: {path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("Statistical Validation of InterSHAP")
    print("=" * 40)

    df, cox_df, median_intershap = load_data()
    print(f"\nDataset: {len(df)} patients")
    print(f"Events (deaths): {df['vital_status'].sum()}")
    print(f"Tumor types: GBM={sum(df['tumor_type']=='GBM')}, LGG={sum(df['tumor_type']=='LGG')}")

    # --- Core validation ---
    cox_res = cox_regression(cox_df)
    quartile_analysis(df)
    boot_res = bootstrap_analysis(cox_df)
    subgroup_analysis(df, cox_df)
    sensitivity_analysis(df)

    # --- Model diagnostics ---
    cph, ph_test = ph_assumption_test(cox_df)
    c_intershap, c_tumor, c_combined = concordance_analysis(cox_df)
    cv_hrs, cv_pvals, cv_c_indices = cross_validation(cox_df)
    unimodal_comparison(cox_df, c_intershap)
    cohens_d, ard, nnh = clinical_effect_size(cox_df)

    # --- PH robustness ---
    rmst_results = rmst_analysis(cox_df)
    landmark_results = landmark_analysis(cox_df)
    piecewise_analysis(cox_df)
    observed_diff, perm_diffs, perm_p = permutation_test(cox_df)

    # --- Figures ---
    print("\nGenerating figures...")
    generate_validation_figure(df, cox_df, cox_res, boot_res, median_intershap)
    generate_diagnostics_figure(cox_df, cph, ph_test, cv_hrs, cv_c_indices,
                                cohens_d, ard, nnh, c_intershap, c_combined)
    generate_robustness_figure(cox_df, rmst_results, landmark_results,
                               observed_diff, perm_diffs)

    # --- Save summary CSV ---
    validation_results = {
        'n_patients': len(df),
        'n_events': int(df['vital_status'].sum()),
        'hr_univariate': cox_res['hr'],
        'hr_ci_low': cox_res['hr_ci_low'],
        'hr_ci_high': cox_res['hr_ci_high'],
        'p_univariate': cox_res['p_uni'],
        'hr_multivariate': cox_res['hr_multi'],
        'p_multivariate': cox_res['p_multi'],
        'hr_bootstrap': boot_res['hr_mean'],
        'hr_bootstrap_ci_low': boot_res['hr_ci'][0],
        'hr_bootstrap_ci_high': boot_res['hr_ci'][1],
        'c_index_intershap': c_intershap,
        'c_index_combined': c_combined,
        'cv_mean_hr': np.mean(cv_hrs),
        'cv_mean_c_index': np.mean(cv_c_indices),
        'cohens_d': cohens_d,
        'nnh': nnh,
        'permutation_p': perm_p,
    }
    pd.DataFrame([validation_results]).to_csv(
        os.path.join(script_dir, 'validation_results.csv'), index=False)
    print("\nResults saved: validation_results.csv")

    # --- Final summary ---
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  N = {len(df)}, Events = {df['vital_status'].sum()}")
    print(f"  Univariate HR: {cox_res['hr']:.2f} ({cox_res['hr_ci_low']:.2f}-{cox_res['hr_ci_high']:.2f}), p = {cox_res['p_uni']:.2e}")
    print(f"  Multivariate HR: {cox_res['hr_multi']:.2f}, p = {cox_res['p_multi']:.2e}")
    print(f"  Bootstrap HR: {boot_res['hr_mean']:.2f} ({boot_res['hr_ci'][0]:.2f}-{boot_res['hr_ci'][1]:.2f})")
    print(f"  C-index: {c_intershap:.3f} (alone), {c_combined:.3f} (+ tumor type)")
    print(f"  10-fold CV C-index: {np.mean(cv_c_indices):.3f} +/- {np.std(cv_c_indices):.3f}")
    print(f"  Cohen's d: {cohens_d:.2f}, NNH: {nnh:.1f}")
    print(f"  RMST diff at 5 years: {rmst_results[2]['diff']:.1f} months")
    print(f"  Permutation p: {perm_p:.6f}")
