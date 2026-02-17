import pandas as pd
import os

# Paths
# Note: Using 80/20 split per paper methodology (no separate val set)
features_path = "MyData/results/early_fusion/features.csv"
splits_dir = "MyData/splits"
train_wsi_path = os.path.join(splits_dir, "train_wsi.csv")
test_wsi_path = os.path.join(splits_dir, "test_wsi.csv")

output_train_path = os.path.join(splits_dir, "early_train.csv")
output_test_path = os.path.join(splits_dir, "early_test.csv")

print("Loading features...")
features_df = pd.read_csv(features_path)
print(f"Features loaded: {features_df.shape}")

def create_split(split_name, wsi_path, output_path):
    print(f"Processing {split_name} split...")
    if not os.path.exists(wsi_path):
        print(f"Error: {wsi_path} not found.")
        return

    wsi_df = pd.read_csv(wsi_path)
    # Get list of cases
    cases = wsi_df['case'].unique()
    
    # Filter features
    split_df = features_df[features_df['case'].isin(cases)]
    
    print(f"  - {split_name} cases: {len(cases)}")
    print(f"  - {split_name} features found: {split_df.shape}")
    
    split_df.to_csv(output_path, index=False)
    print(f"  - Saved to {output_path}")

create_split("Train", train_wsi_path, output_train_path)
create_split("Test", test_wsi_path, output_test_path)

print("Done.")
