"""
Process Kaggle Airbnb Europe Dataset

Combines city-specific CSV files into a single processed dataset
for use with the pricing engine.
"""
import pandas as pd
import numpy as np
import os
import glob

KAGGLE_DIR = "data/kaggle"
OUTPUT_FILE = "data/processed/europe_airbnb.csv"

# City list from the Kaggle dataset
CITIES = [
    'amsterdam', 'athens', 'barcelona', 'berlin', 'budapest',
    'lisbon', 'london', 'paris', 'rome', 'vienna'
]


def load_all_cities():
    """Load and combine all city CSV files."""
    all_data = []
    
    for city in CITIES:
        for day_type in ['weekdays', 'weekends']:
            file_path = os.path.join(KAGGLE_DIR, f"{city}_{day_type}.csv")
            
            if os.path.exists(file_path):
                print(f"  Loading {city}_{day_type}.csv...")
                df = pd.read_csv(file_path)
                df['city'] = city.capitalize()
                df['is_weekend'] = 1 if day_type == 'weekends' else 0
                all_data.append(df)
            else:
                print(f"  ⚠ Missing: {file_path}")
    
    if not all_data:
        raise FileNotFoundError(
            f"No data files found in {KAGGLE_DIR}. "
            "Please run download_kaggle_data.py first."
        )
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n  Combined: {len(combined):,} rows")
    return combined


def clean_and_process(df):
    """Clean and process the combined dataset."""
    print("\n[2/4] Cleaning data...")
    
    # Rename columns to match our pricing engine expectations
    column_mapping = {
        'realSum': 'price',
        'room_type': 'room_type',
        'person_capacity': 'accommodates',
        'host_is_superhost': 'superhost',
        'guest_satisfaction_overall': 'review_scores_rating',
        'dist': 'distance_to_center',
        'metro_dist': 'distance_to_metro',
        'cleanliness_rating': 'cleanliness_rating',
        'multi': 'is_multi_listing',
        'biz': 'is_business',
        'attr_index': 'attraction_index',
        'rest_index': 'restaurant_index',
        'lng': 'longitude',
        'lat': 'latitude'
    }
    
    # Only rename columns that exist
    existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_cols)
    
    # Clean price
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[(df['price'] >= 10) & (df['price'] <= 2000)]
    
    # Encode room type
    if 'room_type' in df.columns:
        room_type_map = {
            'Entire home/apt': 0,
            'Private room': 1,
            'Shared room': 2
        }
        df['room_type_encoded'] = df['room_type'].map(room_type_map).fillna(1)
    
    # Convert superhost to binary
    if 'superhost' in df.columns:
        df['superhost'] = df['superhost'].apply(
            lambda x: 1 if x in [True, 'True', 't', 1, 'TRUE'] else 0
        )
    
    # Normalize satisfaction to 0-5 scale (if it's on 0-100 scale)
    if 'review_scores_rating' in df.columns:
        if df['review_scores_rating'].max() > 10:
            df['review_scores_rating'] = df['review_scores_rating'] / 20.0
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    print(f"  After cleaning: {len(df):,} rows")
    return df


def add_demand_proxy(df):
    """
    Add a demand/occupancy proxy based on listing characteristics.
    
    We estimate demand using:
    - Review scores (higher = more popular)
    - Superhost status (more popular)
    - Price relative to city median (lower = more demand)
    """
    print("\n[3/4] Adding demand proxy...")
    
    # Calculate price relative to city median
    if 'price' in df.columns and 'city' in df.columns:
        city_medians = df.groupby('city')['price'].transform('median')
        df['price_ratio'] = df['price'] / city_medians
    
    # Create demand score (0-1)
    demand_components = []
    
    if 'review_scores_rating' in df.columns:
        # Higher ratings = more demand
        norm_rating = (df['review_scores_rating'] - df['review_scores_rating'].min()) / \
                      (df['review_scores_rating'].max() - df['review_scores_rating'].min())
        demand_components.append(norm_rating * 0.4)
    
    if 'superhost' in df.columns:
        # Superhost = more demand
        demand_components.append(df['superhost'] * 0.2)
    
    if 'price_ratio' in df.columns:
        # Lower relative price = more demand
        inv_price = 1 / (1 + df['price_ratio'])  # Inverse relationship
        demand_components.append(inv_price * 0.4)
    
    if demand_components:
        df['estimated_demand'] = sum(demand_components)
        df['estimated_demand'] = df['estimated_demand'].clip(0.05, 0.80)
    else:
        df['estimated_demand'] = 0.30  # Default
    
    print(f"  Mean demand: {df['estimated_demand'].mean():.2%}")
    return df


def save_processed(df):
    """Save the processed dataset."""
    print("\n[4/4] Saving processed data...")
    
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✓ Saved to {OUTPUT_FILE}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")


def main():
    print("=" * 60)
    print("PROCESSING KAGGLE AIRBNB EUROPE DATA")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists(KAGGLE_DIR):
        print(f"\n⚠ Data directory not found: {KAGGLE_DIR}")
        print("Please run `python scripts/download_kaggle_data.py` first.")
        return
    
    csv_files = glob.glob(os.path.join(KAGGLE_DIR, "*.csv"))
    if not csv_files:
        print(f"\n⚠ No CSV files found in {KAGGLE_DIR}")
        print("Please download the dataset first.")
        return
    
    print(f"\n[1/4] Loading {len(csv_files)} CSV files...")
    df = load_all_cities()
    
    df = clean_and_process(df)
    df = add_demand_proxy(df)
    save_processed(df)
    
    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Update verify_logic.py to use data/processed/europe_airbnb.csv")
    print("  2. Run: python scripts/verify_logic.py")


if __name__ == "__main__":
    main()
