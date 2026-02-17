import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your RAW RNA data (e.g., 'rna.csv' or the file used for unimodal RNA training)
# It should have a 'case' column and gene columns.
raw_rna_path = "MyData/splits/rna_processed.csv"  # <--- UPDATE THIS PATH
# 2. Paths to the WSI Metadata you uploaded
splits_dir = "MyData/splits"
wsi_files = {
    'train': os.path.join(splits_dir, "train_wsi.csv"),
    'val':   os.path.join(splits_dir, "val_wsi.csv"),
    'test':  os.path.join(splits_dir, "test_wsi.csv")
}

# 3. Output Directory
output_dir = "MyData/splits"

# ==========================================
# SCRIPT
# ==========================================
def fix_joint_data():
    if not os.path.exists(raw_rna_path):
        print(f"[ERROR] Raw RNA file not found at: {raw_rna_path}")
        print("Please edit 'raw_rna_path' in this script to point to your original RNA dataset.")
        return

    print(f"Loading Raw RNA from {raw_rna_path}...")
    rna_df = pd.read_csv(raw_rna_path)
    
    # Ensure ID column is 'case'
    if 'case' not in rna_df.columns:
        # Common variations
        for col in ['case_id', 'id', '0']:
            if col in rna_df.columns:
                rna_df.rename(columns={col: 'case'}, inplace=True)
                break
    
    # Keep only 'case' and RNA columns, drop clinical metadata to avoid duplicates
    rna_cols_to_keep = ['case'] + [c for c in rna_df.columns if c.startswith('rna_')]
    rna_df = rna_df[rna_cols_to_keep]
    
    print(f"  RNA Shape: {rna_df.shape} (Should be ~12k columns)")
    
    # Fix Case ID Format: Truncate to first 12 characters (e.g., TCGA-02-0003-01 -> TCGA-02-0003)
    print("  Truncating RNA case IDs to match WSI format...")
    rna_df['case'] = rna_df['case'].astype(str).str[:12]
    print(f"  Example RNA case ID after truncation: {rna_df['case'].iloc[0]}")
    
    # Handle RNA duplicates (multiple tumor samples per patient): keep only the first
    original_rna_count = len(rna_df)
    rna_df = rna_df.drop_duplicates(subset='case', keep='first')
    if original_rna_count > len(rna_df):
        print(f"  Removed {original_rna_count - len(rna_df)} duplicate RNA samples (multiple samples per patient)")

    # Rename gene columns to rna_0, rna_1... if they aren't already
    # This matches the dataset loader expectation
    feature_cols = [c for c in rna_df.columns if c != 'case']
    if not feature_cols[0].startswith('rna_'):
        print("  Renaming gene columns to rna_0, rna_1...")
        rename_map = {col: f"rna_{i}" for i, col in enumerate(feature_cols)}
        rna_df.rename(columns=rename_map, inplace=True)

    # Track all cases to detect cross-split contamination
    all_split_cases = {}

    for split, path in wsi_files.items():
        print(f"\nProcessing {split}...")
        if not os.path.exists(path):
            print(f"  [Error] Metadata file not found: {path}")
            continue

        # Load WSI Metadata (case, wsi_file_name, survival...)
        wsi_df = pd.read_csv(path)
        
        # Normalize columns
        if 'slide_id' in wsi_df.columns: wsi_df.rename(columns={'slide_id': 'wsi_file_name'}, inplace=True)
        if 'case_id' in wsi_df.columns: wsi_df.rename(columns={'case_id': 'case'}, inplace=True)

        # Handle multiple slides per patient: keep only the first slide
        # This prevents duplicate rows after merging with RNA data (which has 1 row per patient)
        original_count = len(wsi_df)
        wsi_df = wsi_df.drop_duplicates(subset='case', keep='first')
        if original_count > len(wsi_df):
            print(f"  Removed {original_count - len(wsi_df)} duplicate slides (multiple slides per patient)")

        # Merge Inner: Keep only patients with BOTH WSI and RNA
        joint_df = wsi_df.merge(rna_df, on='case', how='inner')

        # Remove cases that appeared in previous splits (train > val > test priority)
        if split != 'train':
            before = len(joint_df)
            for prev_split in ['train'] if split != 'train' else []:
                if prev_split in all_split_cases:
                    joint_df = joint_df[~joint_df['case'].isin(all_split_cases[prev_split])]
            if len(joint_df) < before:
                print(f"  Removed {before - len(joint_df)} cases that appeared in previous splits")
        
        # Store cases for this split
        all_split_cases[split] = set(joint_df['case'])

        # Critical: Ensure 'wsi_file_name' exists (required by datasets.py)
        if 'wsi_file_name' not in joint_df.columns:
            joint_df['wsi_file_name'] = joint_df['case']

        # Reorder columns to match joint_example.csv format
        # Format: case, survival_months, vital_status, survival_bin, wsi_file_name, rna_0, rna_1, ...
        meta_cols = ['case', 'survival_months', 'vital_status', 'survival_bin', 'wsi_file_name']
        rna_cols = [c for c in joint_df.columns if c.startswith('rna_')]
        joint_df = joint_df[meta_cols + rna_cols]

        # Save
        out_path = os.path.join(output_dir, f"joint_{split}.csv")
        joint_df.to_csv(out_path, index=False)
        print(f"  -> Saved Corrected File: {out_path} (Cols: {joint_df.shape[1]})")

if __name__ == "__main__":
    fix_joint_data()