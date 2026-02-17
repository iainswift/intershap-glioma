
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
import os

def calculate_c_index(file_path, split_name):
    if not os.path.exists(file_path):
        print(f"{split_name}: File not found at {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # Check for NaNs
    if df.isnull().values.any():
        print(f"{split_name}: Warning - NaNs found, dropping...")
        df = df.dropna()

    scores = df['score'].values
    survival_months = df['survival_months'].values
    vital_status = df['vital_status'].values

    # Calculate C-index
    # We use -scores because higher risk score implies shorter survival time
    try:
        c_index = concordance_index(survival_months, -scores, vital_status)
        print(f"{split_name} C-Index: {c_index:.4f}")
    except Exception as e:
        print(f"{split_name}: Error calculating C-Index - {e}")

base_path = r"c:\Users\iains\Documents\Thesis-MultiModal-Survival\MyData\results\ffpe\checkpoints\outputs\ffpe_model_survcox"

print("--- Analysis of Best Model Results ---")
calculate_c_index(os.path.join(base_path, "train_output_best.csv"), "Train")
calculate_c_index(os.path.join(base_path, "val_output_best.csv"), "Val")
calculate_c_index(os.path.join(base_path, "test_output_best.csv"), "Test")

print("\n--- Analysis of Last Model Results ---")
calculate_c_index(os.path.join(base_path, "train_output_last.csv"), "Train")
calculate_c_index(os.path.join(base_path, "val_output_last.csv"), "Val")
calculate_c_index(os.path.join(base_path, "test_output_last.csv"), "Test")
