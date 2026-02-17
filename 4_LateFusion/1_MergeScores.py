########################################################
### MERGE HISTOPATH AND GENE EXPRESSION UNIMODAL SCORES
#########################################################
### This script merges the survival scores of the histopathology and gene expression unimodal models
### --> this is the input of the Late Fusion model
### - input: 
###          - FFPE model scores (from 3_HistoPath_savescore.py)
###          - RNA model scores (from 2_GeneExpress_savescore.py)
### - output:  Concatenated FFPE and RNA model scores for Late Fusion model
###############################################################################
###############################################################################
### Example command
### $ python 1_MergeScores.py
###################################################
###################################################

### Set Environment
####################
import pandas as pd
import numpy as np

# Load patient info
patient_info_path = "MyData/patientinfo.csv"
try:
    patient_df = pd.read_csv(patient_info_path)
except FileNotFoundError:
    print(f"Warning: Patient info file not found at {patient_info_path}")
    patient_df = pd.DataFrame()

### Merge Scores
#################

# Note: Using 80/20 split per paper methodology (no separate val set)
splits = ['train', 'test']

for split in splits:
    print(f"Processing {split} split...")
    
    ## Pathology
    # Path based on config_ffpe_savescore_survcox.json output
    ffpe_path = f"MyData/results/ffpe/checkpoints/outputs/ffpe_model_survcox_fold4/model_dict_best.pt_pathology_{split}_df.csv"
    try:
        path_df = pd.read_csv(ffpe_path, header=0)
        path_df.rename({'score':'path_score', 'id':'case'}, inplace=True, axis=1)
    except FileNotFoundError:
        print(f"Warning: FFPE score file not found at {ffpe_path}")
        continue
        
    ## RNA
    # Path based on config_rna_savescore_survcox.json output
    rna_path = f"MyData/results/rna/checkpoints/outputs/rna_model_survcox_fold4/model_dict_best.pt_rna_{split}_df.csv"
    try:
        rna_df = pd.read_csv(rna_path, header=0)
        rna_df.rename({'score':'rna_score', 'id':'case'}, inplace=True, axis=1)
        # Truncate RNA case IDs to match Pathology case IDs (first 12 chars)
        rna_df['case'] = rna_df['case'].apply(lambda x: x[:12])
        
        # Handle duplicate RNA samples per patient (multiple tumor aliquots)
        # Average the scores for patients with multiple RNA samples
        duplicates_before = len(rna_df)
        rna_df = rna_df.groupby('case', as_index=False).agg({'rna_score': 'mean'})
        if duplicates_before > len(rna_df):
            print(f"  Deduplicated RNA: {duplicates_before} -> {len(rna_df)} (averaged {duplicates_before - len(rna_df)} duplicate samples)")
    except FileNotFoundError:
        print(f"Warning: RNA score file not found at {rna_path}")
        continue

    ### Merge
    # Ensure we only keep common cases
    final_df = path_df.merge(rna_df[['case','rna_score']], how="inner", on="case")

    # Add survival_bin
    if not patient_df.empty and 'survival_bin' in patient_df.columns:
        final_df = final_df.merge(patient_df[['case', 'survival_bin']], how="left", on="case")

    # Reorder columns
    desired_cols = ['case', 'survival_months', 'vital_status', 'survival_bin', 'rna_score', 'path_score']
    final_cols = [c for c in desired_cols if c in final_df.columns]
    final_df = final_df[final_cols]

    print(f"Merged {len(final_df)} cases for {split} split.")
    
    output_path = f"MyData/results/late_fusion/combined_scores_{split}.csv"
    if 'Unnamed: 0' in final_df.columns:
        final_df.drop(['Unnamed: 0'], axis=1).to_csv(output_path, index=False)
    else:
        final_df.to_csv(output_path, index=False)
        
    print(f"Saved merged scores to {output_path}")