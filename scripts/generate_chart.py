import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    listings = pd.read_csv('listings.csv')
    calendar = pd.read_csv('calendar.csv') # Load full or large sample

    # --- Preprocessing (Condensed) ---
    listing_cols = ['id', 'neighbourhood', 'room_type', 'price', 
                    'number_of_reviews', 'reviews_per_month', 'availability_365']
    real_cols = [c for c in listing_cols if c in listings.columns]
    listings_subset = listings[real_cols].rename(columns={'id': 'listing_id', 'price': 'listing_price'})
    listings_subset['listing_id'] = listings_subset['listing_id'].astype('int64') # Fix ID type
    
    if 'reviews_per_month' in listings_subset.columns:
        listings_subset['reviews_per_month'] = listings_subset['reviews_per_month'].fillna(0)

    # Calendar cleanup
    calendar['price'] = calendar['price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    calendar['listing_id'] = calendar['listing_id'].astype('int64') # Fix ID type
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar['month'] = calendar['date'].dt.month
    calendar['day_of_week'] = calendar['date'].dt.dayofweek
    calendar['is_weekend'] = calendar['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    calendar['is_booked'] = calendar['available'].apply(lambda x: 1 if x == 'f' else 0)

    print("Merging...")
    df = calendar.merge(listings_subset, on='listing_id', how='left')
    
    if 'listing_price' in df.columns:
        df['price'] = df['price'].fillna(df['listing_price'])
    
    df = df.dropna(subset=['price'])
    
    # --- features ---
    features = ['price', 'month', 'day_of_week', 'is_weekend', 
                'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']
    
    # Filter features that actually exist in merged df
    model_features = [f for f in features if f in df.columns]
    
    # Sample for speed if needed, but we want decent model
    if len(df) > 100000:
        model_df = df.sample(100000, random_state=42)
    else:
        model_df = df.copy()

    X = model_df[model_features]
    y = model_df['is_booked']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=5, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # --- Find Valid Example ---
    print("Searching for a valid optimization example (Beta > 0)...")
    
    found = False
    for i in range(100): # Try 100 random samples
        sample_row = X_test.iloc[i]
        actual_price = sample_row['price']
        
        # Optimization Logic
        price_range = np.linspace(max(10, actual_price * 0.5), actual_price * 1.5, 50)
        temp_df = pd.DataFrame([sample_row] * len(price_range))
        temp_df['price'] = price_range
        
        preds = model.predict(temp_df[model_features])
        preds = np.clip(preds, 0, 1)
        
        lr = LinearRegression()
        lr.fit(price_range.reshape(-1, 1), preds)
        slope = lr.coef_[0]
        intercept = lr.intercept_
        beta = -slope
        alpha = intercept
        
        if beta > 0.0001: # Ensure positive slope (downward demand)
            opt_price = alpha / (2 * beta)
            print(f"Found valid example at index {i}!")
            print(f"Alpha: {alpha:.4f}, Beta: {beta:.6f}")
            print(f"Actual Price: {actual_price}, Optimal Price: {opt_price:.2f}")
            
            # --- PLOTTING ---
            plt.figure(figsize=(12, 8))
            
            # Plot XGBoost curve
            plt.plot(price_range, preds, label='Demand Estimation (XGBoost)', linewidth=3, color='#007acc')
            
            # Plot Linear Approx
            linear_demand = alpha - beta * price_range
            plt.plot(price_range, linear_demand, '--', label=f'Linear Approx (D = {alpha:.2f} - {beta:.4f}P)', color='orange')
            
            # Plot Points
            plt.scatter([actual_price], [alpha - beta * actual_price], color='red', s=100, zorder=5, label=f'Actual Price ${actual_price:.0f}')
            plt.scatter([opt_price], [alpha - beta * opt_price], color='green', s=150, marker='*', zorder=5, label=f'Optimal Price ${opt_price:.0f}')
            
            # Annotations
            plt.axvline(actual_price, color='red', linestyle=':', alpha=0.5)
            plt.axvline(opt_price, color='green', linestyle=':', alpha=0.5)
            
            plt.title('Dynamic Pricing Optimization: Demand Curve Analysis', fontsize=16)
            plt.xlabel('Price ($)', fontsize=12)
            plt.ylabel('Booking Probability (Demand)', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            output_file = 'demand_curve_chart.png'
            plt.savefig(output_file)
            print(f"\nChart saved to: {output_file}")
            found = True
            break
            
    if not found:
        print("Could not find a clear downward sloping demand curve in the first 100 samples.")

if __name__ == "__main__":
    main()
