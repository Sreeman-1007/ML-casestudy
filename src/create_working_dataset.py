import os
import sys
import glob
import gc
import numpy as np
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def build_working_dataset(
    dataset_dir="./dataset/MachineLearningCVE",
    output_path="./dataset/working_dataset.csv",
    target_samples_per_file=25000,
    random_state=42
):
    print("=" * 70)
    print("CREATING REPRODUCIBLE WORKING DATASET FOR CONGESTION PREDICTION")
    print("=" * 70)
    print(f"Source Directory: {dataset_dir}")
    print(f"Output File:      {output_path}")
    print(f"Random State:     {random_state}")
    print(f"Target per file:  {target_samples_per_file}")
    
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
        
    print(f"Found {len(csv_files)} source CSV files.")
    
    # Initialize RNG once with random_state=42 for reproducible selection across files
    rng = np.random.default_rng(seed=random_state)
    
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed previous output file at {output_path}")

    total_sampled = 0
    file_stats = []
    first_write = True
    
    for file_idx, fpath in enumerate(csv_files):
        fname = os.path.basename(fpath)
        print(f"\nProcessing [{file_idx + 1}/{len(csv_files)}]: {fname}")
        
        # 1. Count rows in file without loading into memory
        with open(fpath, "rb") as fp:
            line_count = sum(chunk.count(b"\n") for chunk in iter(lambda: fp.read(1024 * 1024 * 4), b""))
        n_rows = line_count - 1  # subtract header
        print(f"  Total records in file: {n_rows:,}")
        
        # 2. Select uniform random indices without replacement
        sample_size = min(target_samples_per_file, n_rows)
        chosen_indices = set(rng.choice(n_rows, size=sample_size, replace=False))
        print(f"  Selected {len(chosen_indices):,} indices uniformly across file.")
        
        # 3. Read in chunks and filter
        current_row_idx = 0
        file_chunks = []
        chunk_size = 50000
        
        for chunk in pd.read_csv(fpath, chunksize=chunk_size, encoding="utf-8", low_memory=False):
            chunk_len = len(chunk)
            # Find indices belonging to this chunk
            local_indices = [
                i - current_row_idx
                for i in range(current_row_idx, current_row_idx + chunk_len)
                if i in chosen_indices
            ]
            
            if local_indices:
                sub_chunk = chunk.iloc[local_indices].copy()
                # Clean column headers
                sub_chunk.columns = sub_chunk.columns.str.strip()
                
                # Drop duplicate 'Fwd Header Length.1' column if present
                if "Fwd Header Length.1" in sub_chunk.columns:
                    sub_chunk.drop(columns=["Fwd Header Length.1"], inplace=True)
                
                # Clean label strings (strip whitespace and clean non-ASCII chars)
                if "Label" in sub_chunk.columns:
                    sub_chunk["Label"] = (
                        sub_chunk["Label"]
                        .astype(str)
                        .str.strip()
                        .str.replace("\ufffd", "-", regex=False)
                        .str.replace("–", "-", regex=False)
                    )
                
                file_chunks.append(sub_chunk)
                del sub_chunk
                
            current_row_idx += chunk_len
            del chunk
            
        # Combine chunks for this file
        file_df = pd.concat(file_chunks, ignore_index=True)
        del file_chunks
        gc.collect()
        
        # Verify count
        records_collected = len(file_df)
        total_sampled += records_collected
        print(f"  Collected {records_collected:,} records from {fname}")
        
        file_stats.append({
            "file_name": fname,
            "total_file_rows": n_rows,
            "sampled_rows": records_collected
        })
        
        # Write incrementally to output CSV
        file_df.to_csv(
            output_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8"
        )
        first_write = False
        
        del file_df
        gc.collect()
        print(f"  Appended to {output_path} (Cumulative total: {total_sampled:,})")

    print("\n" + "=" * 70)
    print(f"SAMPLING COMPLETE! Total rows written: {total_sampled:,}")
    print("=" * 70)
    return total_sampled, file_stats

if __name__ == "__main__":
    build_working_dataset()
