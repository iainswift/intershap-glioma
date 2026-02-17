#########################################
### CONCATENATE RNA AND FFPE FEATURES
#########################################
### This script concatenates the FFPE and RNA features which is used as input Early Fusion model 
### - input: 
###          - FFPE features (split into train/val/test in specific output dirs)
###          - RNA features (split into train/val/test in specific output dirs)
###          - Patientinfo (columns: case, survival_months, vital_status (1 or 0), survival bin)
### - output:  Concatenated FFPE and RNA features for Early Fusion model
###############################################################################
###############################################################################
### Example command
### $ python 1_Concat2Features.py
###################################################
###################################################

### Set Environment
####################
import pandas as pd
import numpy as np
import os

# Define output paths based on your config files
rna_dir = "MyData/results/rna/checkpoints/outputs/rna_model_survcox_fold4/"
ffpe_dir = "MyData/results/ffpe/checkpoints/outputs/ffpe_model_survcox_fold4/"

### Helper function to load and combine splits
def load_and_combine_splits(base_dir, modality_prefix, file_type):
    """
    Loads train and test files and concatenates them.
    Note: Using 80/20 split per paper methodology (no separate val set)
    file_type: 'cases' (has header) or 'features' (no header)
    """
    splits = ['train', 'test']
    df_list = []
    
    print(f"Loading {modality_prefix} {file_type}...")
    
    for split in splits:
        # Construct filename: e.g., rna_cases_train.csv or pathology_features_val.csv
        filename = f"{modality_prefix}_{file_type}_{split}.csv"
        file_path = os.path.join(base_dir, filename)
        
        if os.path.exists(file_path):
            if file_type == 'cases':
                # Cases files usually have a header and an index column
                df = pd.read_csv(file_path, header=0)
            else:
                # Features files usually have no header
                df = pd.read_csv(file_path, header=None)
            df_list.append(df)
            print(f"  - Loaded {split}: {df.shape}")
        else:
            print(f"  ! Warning: File not found: {file_path}")
            
    if not df_list:
        raise FileNotFoundError(f"No {file_type} files found in {base_dir}")
        
    return pd.concat(df_list, axis=0, ignore_index=True)

### Concat Features
####################

### RNA
# Load all splits for RNA cases and features
rna_cases = load_and_combine_splits(rna_dir, "rna", "cases")
rna_features = load_and_combine_splits(rna_dir, "rna", "features")
print(f"Total RNA Data: {rna_cases.shape}, {rna_features.shape}")

## Pathology
# Load all splits for Pathology cases and features
pathology_cases = load_and_combine_splits(ffpe_dir, "pathology", "cases")
pathology_features = load_and_combine_splits(ffpe_dir, "pathology", "features")
print(f"Total Pathology Data: {pathology_cases.shape}, {pathology_features.shape}")

## Patient Info
# needed: 
# - patient id (= case_)
# - survival_months
# - vital_status (1 or 0) (if task = survival prediction)
# - survival bin (0 - 4) (if task = survival bin)
print("\nLoading Patient Info...")
try:
    patientinfo = pd.read_csv("MyData/patientinfo.csv", header=0)
    # Ensure standard column names
    if 'case' not in patientinfo.columns and 'case_id' in patientinfo.columns:
        patientinfo.rename(columns={'case_id': 'case'}, inplace=True)
        
    # Check for required columns
    required_cols = ['case', 'survival_months', 'vital_status']
    if 'survival_bin' in patientinfo.columns:
        required_cols.append('survival_bin')
    
    patientinfo = patientinfo[required_cols]
    # Drop duplicates in patient info
    patientinfo = patientinfo.drop_duplicates(subset=['case'])
    print(f"Patient Info loaded. Shape: {patientinfo.shape}")
except FileNotFoundError:
    print("Error: 'patientinfo.csv' not found. Please ensure it is in the current directory.")
    exit(1)

## Sanity check
# Extract the Case IDs (Assuming column '0' contains the ID based on standard output)
rna_cases_id = list(rna_cases['0'])
pathology_cases_id = list(pathology_cases['0'])

# Truncate RNA case IDs to match Pathology format (remove -01 suffix)
# RNA: TCGA-HT-7480-01 -> TCGA-HT-7480
# FFPE: TCGA-HT-7875
rna_cases_id_truncated = [x[:12] for x in rna_cases_id]

common_ids = set(rna_cases_id_truncated).intersection(set(pathology_cases_id))
print(f"\nSanity Check: {len(common_ids)} common cases found between modalities.")

## Merge
# Prepare RNA DataFrame
rna_df = rna_features.copy()
rna_df['case'] = rna_cases_id_truncated
# Drop duplicates in RNA features (average if multiple samples per patient)
rna_df = rna_df.groupby('case').mean().reset_index()

# Prepare Pathology DataFrame
pathology_df = pathology_features.copy()
pathology_df['case'] = pathology_cases_id
# Drop duplicates in Pathology features
pathology_df = pathology_df.drop_duplicates(subset=['case'])

# Merge RNA and Pathology on Case ID
# This aligns the rows ensuring we only keep patients with both data types
merged_df = rna_df.merge(pathology_df, how="inner", on='case')
print(f"Merged Modalities Shape: {merged_df.shape}")

# Merge with Patient Info
final_df = patientinfo.merge(merged_df, how="inner", on='case')
print(f"Final Merged Dataset Shape: {final_df.shape}")

## Output
# Rename columns to standardized format: metadata + feature_0, feature_1, ...
# Detect metadata columns (everything from patientinfo)
metadata_cols = list(patientinfo.columns) 
feature_cols = ['feature_'+str(i) for i in range(final_df.shape[1] - len(metadata_cols))]

final_df.columns = metadata_cols + feature_cols
final_df.to_csv('MyData/results/early_fusion/features.csv', index=False)
print("\nSuccess! Saved to 'MyData/results/early_fusion/features.csv'")