import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# Configuration
RNA_DATA_DIR = 'rna pre processing/rna_data'
MANIFEST_FILE = 'rna pre processing/tcga_gbmlgg_manifest.csv'
CLINICAL_FILE = 'rna pre processing/survival_GBMLGG_survival.txt'
GRADE_FILE = 'rna pre processing/TCGA.GBMLGG.sampleMap_GBMLGG_clinicalMatrix'
GENES_FILE = '2_GeneExpression/genes.txt'
OUTPUT_FILE = 'rna_processed.csv'

def load_target_genes(genes_path):
    """
    Load the list of target genes from genes.txt.
    Expected format: CSV with an index column and a gene name column.
    """
    print(f"Loading target genes from {genes_path}...")
    # Read CSV. The file has a header like ",0"
    df = pd.read_csv(genes_path)
    # The gene names are in the second column (index 1)
    # The first column is likely an index.
    # We assume the column with gene names is named '0' based on the file content ",0"
    if '0' in df.columns:
        genes = df['0'].values
    else:
        # Fallback: assume it's the second column
        genes = df.iloc[:, 1].values
        
    print(f"Loaded {len(genes)} target genes.")
    return genes

def load_manifest(manifest_path):
    """
    Load manifest to map filenames to TCGA barcodes.
    Returns a dictionary: filename -> sample_id (truncated to 15 chars)
    """
    print(f"Loading manifest from {manifest_path}...")
    manifest = pd.read_csv(manifest_path)
    mapping = {}
    for _, row in manifest.iterrows():
        # Map filename to TCGA barcode (e.g., TCGA-02-0001-01)
        # We truncate to 15 chars to match the survival file format
        sample_id = row['sample_id'][:15] 
        mapping[row['file_name']] = sample_id
    return mapping

def load_clinical_data(clinical_path):
    """
    Load survival data.
    Returns a DataFrame indexed by sample ID.
    """
    print(f"Loading clinical data from {clinical_path}...")
    # The file appears to be tab-separated based on previous context
    clinical = pd.read_csv(clinical_path, sep='\t')
    
    # Rename columns to match our standard
    # Expected columns in file: sample, OS, OS.time
    clinical = clinical.rename(columns={
        'sample': 'case',
        'OS': 'vital_status',
        'OS.time': 'survival_months' # Note: This might be days, we'll check/convert if needed
    })
    
    # Set index to case ID
    clinical = clinical.set_index('case')
    
    # Ensure numeric
    clinical['vital_status'] = pd.to_numeric(clinical['vital_status'], errors='coerce')
    clinical['survival_months'] = pd.to_numeric(clinical['survival_months'], errors='coerce')
    
    # Drop missing survival data
    clinical = clinical.dropna(subset=['vital_status', 'survival_months'])
    
    return clinical[['vital_status', 'survival_months']]

def load_grade_data(grade_path):
    """
    Load grade data from clinical matrix.
    Returns a DataFrame indexed by sample ID with 'grade_binary' column.
    """
    print(f"Loading grade data from {grade_path}...")
    df = pd.read_csv(grade_path, sep='\t')
    
    # We need 'sampleID' and 'histological_type' / 'neoplasm_histologic_grade'
    # Create grade_binary: 1 if GBM, 0 if LGG
    # Logic: If histological_type contains 'GBM' or 'Glioblastoma', it's 1. Else 0.
    
    grade_data = []
    for _, row in df.iterrows():
        sample_id = row['sampleID']
        histo_type = str(row['histological_type'])
        
        if 'GBM' in histo_type or 'Glioblastoma' in histo_type:
            grade = 1
        else:
            grade = 0
            
        grade_data.append({'case': sample_id, 'grade_binary': grade})
        
    grade_df = pd.DataFrame(grade_data)
    grade_df = grade_df.set_index('case')
    return grade_df

def process_rna_data():
    # 1. Load Mappings and Targets
    file_to_sample = load_manifest(MANIFEST_FILE)
    clinical_df = load_clinical_data(CLINICAL_FILE)
    grade_df = load_grade_data(GRADE_FILE)
    target_genes = load_target_genes(GENES_FILE)
    
    # 2. Aggregate RNA Files
    print("Aggregating RNA files...")
    files = glob.glob(os.path.join(RNA_DATA_DIR, '*.tsv'))
    
    expression_data = {}
    
    for f in tqdm(files):
        filename = os.path.basename(f)
        
        if filename not in file_to_sample:
            continue
            
        sample_id = file_to_sample[filename]
        
        # Read TSV
        # Skip first 6 rows (header + N_ stats)
        # Columns: gene_id, gene_name, gene_type, unstranded, stranded_first, stranded_second, tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded
        # We want 'fpkm_unstranded' which is at index 7
        # We want 'gene_name' which is at index 1
        try:
            df = pd.read_csv(f, sep='\t', skiprows=6, header=None)
            # Group by gene name (index 1) and take max FPKM (index 7) to handle duplicates
            # This creates a Series with gene_name as index
            sample_genes = df.groupby(1)[7].max()
            
            # Reindex to match target genes, filling missing with 0
            # This ensures we have exactly the genes we want in the correct order
            aligned_genes = sample_genes.reindex(target_genes, fill_value=0)
            
            expression_data[sample_id] = aligned_genes
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Create DataFrame (Rows=Samples, Cols=Genes)
    print("Creating expression matrix...")
    rna_df = pd.DataFrame(expression_data).T
    print(f"Initial shape: {rna_df.shape}")
    
    # Fill NaNs with 0 (should be handled by reindex, but just in case)
    rna_df = rna_df.fillna(0)

    # 3. Normalization
    print("Applying Log2(x+1) normalization...")
    rna_df = np.log2(rna_df + 1)

    # 4. Feature Selection (Already done by reindexing to target_genes)
    print(f"Verifying features match target list ({len(target_genes)} genes)...")
    # Double check columns match target_genes
    if not rna_df.columns.equals(pd.Index(target_genes)):
        print("Warning: Columns do not match target genes exactly. Realigning...")
        rna_df = rna_df[target_genes]
    
    # Rename columns to generic rna_0, rna_1...
    rna_df.columns = [f'rna_{i}' for i in range(len(rna_df.columns))]

    # 5. Merge with Clinical Data and Grade Data
    print("Merging with clinical and grade data...")
    # Inner join to keep only samples with both RNA and Survival info
    # Also join with grade info
    final_df = clinical_df.join(rna_df, how='inner')
    final_df = final_df.join(grade_df, how='left') # Left join to keep samples even if grade is missing (though it shouldn't be)
    
    # Fill missing grade with 0 (LGG) or drop? Let's assume 0 if missing for now, or check.
    # Given the previous check, most samples had grade info.
    final_df['grade_binary'] = final_df['grade_binary'].fillna(0).astype(int)
    
    print(f"Merged shape: {final_df.shape}")

    # 6. Create Survival Bins
    print("Creating survival bins (quartiles)...")
    try:
        final_df['survival_bin'] = pd.qcut(final_df['survival_months'], q=4, labels=False)
    except Exception as e:
        print(f"Binning failed: {e}")
        # Fallback or handle error

    # Reset index to make 'case' a column
    final_df = final_df.reset_index()
    
    # Ensure 'case' column exists (rename 'index' if necessary)
    if 'case' not in final_df.columns and 'index' in final_df.columns:
        final_df = final_df.rename(columns={'index': 'case'})
    
    print("Columns after reset_index:", final_df.columns.tolist())

    # Reorder columns
    # Expected: case, survival_months, vital_status, survival_bin, grade_binary, rna_0...
    cols = ['case', 'survival_months', 'vital_status', 'survival_bin', 'grade_binary'] + [c for c in final_df.columns if 'rna_' in c]
    final_df = final_df[cols]
    
    # 7. Save
    print(f"Saving to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    process_rna_data()
