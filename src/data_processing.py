import pandas as pd
import numpy as np

def clean_price(series):
    """
    Removes currency symbols and commas from price strings and converts to float.
    """
    return series.astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

def load_and_preprocess_data(listings_path, calendar_path, sample_size=None):
    """
    Loads Airbnb listings and calendar data, cleans prices, and merges them.
    
    Args:
        listings_path: Path to listings CSV
        calendar_path: Path to calendar CSV
        sample_size: Optional number of rows to sample from calendar for faster processing
    """
    calendar_dtypes = {
        'listing_id': 'int64',
        'available': 'category',
        'minimum_nights': 'float32',
        'maximum_nights': 'float32'
    }
    
    print("Loading data...")
    listings = pd.read_csv(listings_path)
    
    if sample_size:
        calendar = pd.read_csv(calendar_path, dtype=calendar_dtypes, parse_dates=['date'], nrows=sample_size)
    else:
        calendar = pd.read_csv(calendar_path, dtype=calendar_dtypes, parse_dates=['date'])
    
    # Clean Price
    calendar['price'] = clean_price(calendar['price'])
    listings['price'] = clean_price(listings['price'])
    
    # Fill missing prices in calendar from listings
    listing_prices = listings.set_index('id')['price'].to_dict()
    calendar['price'] = calendar.apply(
        lambda row: listing_prices.get(row['listing_id'], row['price']) if pd.isna(row['price']) else row['price'],
        axis=1
    )
    
    # Backfill any remaining NaN prices within each listing
    calendar['price'] = calendar.groupby('listing_id')['price'].ffill().bfill()
    
    # Filter unrealistic prices (keep prices between $10 and $10,000)
    calendar = calendar[(calendar['price'] >= 10) & (calendar['price'] <= 10000)]
    
    # Feature Engineering on Calendar
    calendar['month'] = calendar['date'].dt.month
    calendar['day_of_week'] = calendar['date'].dt.dayofweek
    calendar['is_weekend'] = calendar['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    calendar['is_booked'] = calendar['available'].apply(lambda x: 1 if x == 'f' else 0)
    
    # Feature Engineering on Listings
    listing_cols = ['id', 'neighbourhood', 'room_type', 'price', 
                    'number_of_reviews', 'reviews_per_month', 'availability_365',
                    'review_scores_rating', 'accommodates']
    
    real_cols = [c for c in listing_cols if c in listings.columns]
    listings_subset = listings[real_cols].rename(columns={'id': 'listing_id', 'price': 'listing_price'})
    listings_subset['listing_id'] = listings_subset['listing_id'].astype('int64')
    
    # Fill NaN values with sensible defaults
    if 'reviews_per_month' in listings_subset.columns:
        listings_subset['reviews_per_month'] = listings_subset['reviews_per_month'].fillna(0)
    if 'review_scores_rating' in listings_subset.columns:
        listings_subset['review_scores_rating'] = listings_subset['review_scores_rating'].fillna(
            listings_subset['review_scores_rating'].median()
        )
    if 'accommodates' in listings_subset.columns:
        listings_subset['accommodates'] = listings_subset['accommodates'].fillna(2)
        
    # Merge
    print("Merging data...")
    df = calendar.merge(listings_subset, on='listing_id', how='left')
    
    if 'listing_price' in df.columns:
        df['price'] = df['price'].fillna(df['listing_price'])
        
    df = df.dropna(subset=['price'])
    
    # Encode categorical room_type
    if 'room_type' in df.columns:
        df['room_type'] = df['room_type'].astype('category').cat.codes
    
    # Fill remaining NaN values for features
    for col in ['review_scores_rating', 'accommodates', 'reviews_per_month']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        
    return df
