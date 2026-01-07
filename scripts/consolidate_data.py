import pandas as pd
import glob
import os

def consolidate_csvs(directory, output_file, pattern="*.csv.gz"):
    """
    Reads all gzip-compressed CSV files matching a pattern in a directory 
    and concatenates them into a single CSV file.
    """
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"No files found matching {pattern} in {directory}")
        return

    print(f"Found {len(files)} files. Starting consolidation...")
    
    dfs = []
    for f in sorted(files):
        print(f"Reading {os.path.basename(f)}...")
        # Note: compression is inferred by pandas from the extension
        df = pd.read_csv(f)
        dfs.append(df)
    
    if dfs:
        print("Concatenating dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Use simple name if there are duplicate columns after merge or similar
        # For Airbnb data, these are usually different time snapshots of the same listings
        
        print(f"Saving to {output_file}...")
        combined_df.to_csv(output_file, index=False)
        print("Done!")
    else:
        print("No data found to consolidate.")

if __name__ == "__main__":
    # Base directory
    base_raw = "data/raw"
    
    # Consolidate Calendar files
    calendar_dir = os.path.join(base_raw, "calendar")
    calendar_output = os.path.join(base_raw, "calendar.csv")
    consolidate_csvs(calendar_dir, calendar_output)
    
    # Consolidate Listing files
    listing_dir = os.path.join(base_raw, "listing")
    listing_output = os.path.join(base_raw, "listings.csv")
    consolidate_csvs(listing_dir, listing_output)
