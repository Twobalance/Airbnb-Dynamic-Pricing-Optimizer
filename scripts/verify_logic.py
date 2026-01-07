"""
Airbnb Dynamic Pricing - Verification Script (v4 - Kaggle Europe Data)

Uses the processed Kaggle Europe dataset with:
1. Economic demand curve model
2. City-specific pricing segments
3. Realistic occupancy constraints
"""
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# Data paths
KAGGLE_DATA = "data/processed/europe_airbnb.csv"
INSIDE_AIRBNB_DATA = "data/raw/listings.csv"


def exponential_demand(price, base_demand, price_sensitivity, reference_price):
    """
    Exponential demand curve: D(P) = D_base * exp(-sensitivity * (P - P_ref) / P_ref)
    Returns demand probability capped to [0.01, 0.50]
    """
    relative_price_diff = (price - reference_price) / reference_price
    raw_demand = base_demand * np.exp(-price_sensitivity * relative_price_diff)
    return np.clip(raw_demand, 0.01, 0.50)


def estimate_market_parameters(df, price_col='price'):
    """Estimate market parameters from data."""
    reference_price = df[price_col].median()
    
    # Use estimated_demand if available, otherwise calculate from reviews
    if 'estimated_demand' in df.columns:
        base_demand = df['estimated_demand'].mean()
        base_demand = np.clip(base_demand, 0.15, 0.40)
    elif 'review_scores_rating' in df.columns:
        # Higher ratings correlate with demand
        avg_rating = df['review_scores_rating'].mean()
        base_demand = min(avg_rating / 10 * 0.5, 0.40) if avg_rating > 0 else 0.25
    else:
        base_demand = 0.25
    
    # Price sensitivity from variance
    price_cv = df[price_col].std() / df[price_col].mean()
    price_sensitivity = np.clip(1.5 + price_cv, 1.2, 2.5)
    
    return base_demand, price_sensitivity, reference_price


def find_optimal_price(base_demand, price_sensitivity, reference_price,
                       min_price=20, max_price=500):
    """Find revenue-maximizing price."""
    price_range = np.linspace(min_price, max_price, 100)
    demands = exponential_demand(price_range, base_demand, price_sensitivity, reference_price)
    revenues = price_range * demands
    
    idx_max = np.argmax(revenues)
    optimal_price = price_range[idx_max]
    optimal_demand = demands[idx_max]
    max_revenue = revenues[idx_max]
    
    # Calculate elasticity
    if 0 < idx_max < len(price_range) - 1:
        dQ = demands[idx_max + 1] - demands[idx_max - 1]
        dP = price_range[idx_max + 1] - price_range[idx_max - 1]
        elasticity = (dQ / dP) * (optimal_price / optimal_demand)
    else:
        elasticity = -price_sensitivity
    
    return optimal_price, optimal_demand, max_revenue, elasticity, price_range, demands


def main():
    print("=" * 60)
    print("AIRBNB DYNAMIC PRICING - KAGGLE EUROPE DATA")
    print("=" * 60)
    
    # Determine which data to use
    if os.path.exists(KAGGLE_DATA):
        data_path = KAGGLE_DATA
        data_source = "Kaggle Europe"
    elif os.path.exists(INSIDE_AIRBNB_DATA):
        data_path = INSIDE_AIRBNB_DATA
        data_source = "Inside Airbnb"
    else:
        # Try relative path
        if os.path.exists("../" + KAGGLE_DATA):
            data_path = "../" + KAGGLE_DATA
            data_source = "Kaggle Europe"
        else:
            print("⚠ No data found. Run process_kaggle_data.py first.")
            return
    
    print(f"\nUsing data source: {data_source}")
    
    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv(data_path)
    
    # Clean price if needed
    if df['price'].dtype == object:
        df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    
    df = df[(df['price'] >= 10) & (df['price'] <= 2000)]
    
    print(f"   Loaded {len(df):,} listings")
    print(f"   Price range: ${df['price'].min():.0f} - ${df['price'].max():.0f}")
    print(f"   Median price: ${df['price'].median():.0f}")
    
    # City breakdown
    if 'city' in df.columns:
        print(f"\n   Cities: {', '.join(df['city'].unique())}")
    
    # Overall market parameters
    print("\n[2/4] Estimating market parameters...")
    base_demand, price_sensitivity, reference_price = estimate_market_parameters(df)
    
    print(f"   Reference Price (median): ${reference_price:.2f}")
    print(f"   Base Demand: {base_demand:.2%}")
    print(f"   Price Sensitivity (λ): {price_sensitivity:.2f}")
    
    # City-level analysis
    print("\n[3/4] City segment analysis...")
    
    if 'city' in df.columns:
        city_results = []
        for city in df['city'].unique():
            city_df = df[df['city'] == city]
            if len(city_df) > 100:
                c_demand, c_sens, c_ref = estimate_market_parameters(city_df)
                c_opt, c_prob, c_rev, c_elast, _, _ = find_optimal_price(c_demand, c_sens, c_ref)
                city_results.append({
                    'city': city,
                    'count': len(city_df),
                    'median_price': c_ref,
                    'optimal_price': c_opt,
                    'demand': c_prob,
                    'revenue': c_rev
                })
        
        print("\n   City-Level Optimization:")
        print("   " + "-" * 56)
        print(f"   {'City':<12} {'Listings':>8} {'Median':>10} {'Optimal':>10} {'Revenue':>10}")
        print("   " + "-" * 56)
        for r in sorted(city_results, key=lambda x: x['revenue'], reverse=True):
            print(f"   {r['city']:<12} {r['count']:>8,} ${r['median_price']:>8.0f} ${r['optimal_price']:>8.0f} ${r['revenue']:>8.2f}")
        print("   " + "-" * 56)
    
    # Overall optimization
    print("\n[4/4] Running overall revenue optimization...")
    
    optimal_price, optimal_demand, max_revenue, elasticity, price_range, demands = \
        find_optimal_price(base_demand, price_sensitivity, reference_price)
    
    revenue_at_median = reference_price * exponential_demand(
        reference_price, base_demand, price_sensitivity, reference_price)
    revenue_gain = (max_revenue - revenue_at_median) / revenue_at_median * 100

    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\n   Market Median Price:        ${reference_price:.2f}")
    print(f"   Optimal Price:              ${optimal_price:.2f}")
    print(f"   Price Adjustment:           {(optimal_price/reference_price - 1)*100:+.1f}%")
    print(f"\n   Booking Probability:        {optimal_demand:.2%}")
    print(f"   Expected Revenue/Night:     ${max_revenue:.2f}")
    print(f"   Revenue vs Median Pricing:  {revenue_gain:+.1f}%")
    print(f"\n   Price Elasticity:           {elasticity:.2f}")
    print("=" * 60)
    
    # Sanity checks
    print("\nSanity Checks:")
    checks_passed = 0
    
    if 20 <= optimal_price <= 1000:
        print("  ✓ Optimal price is within expected range ($20-$1000)")
        checks_passed += 1
    else:
        print("  ⚠ Warning: Optimal price outside expected range!")
    
    if 0.05 <= optimal_demand <= 0.50:
        print("  ✓ Booking probability is realistic (5-50%)")
        checks_passed += 1
    else:
        print(f"  ⚠ Warning: Booking probability {optimal_demand:.2%} outside range!")
    
    if elasticity < 0:
        print("  ✓ Negative elasticity confirms demand decreases with price")
        checks_passed += 1
    else:
        print("  ⚠ Warning: Elasticity is non-negative!")
    
    if -3 < elasticity < -0.5:
        print("  ✓ Elasticity magnitude is reasonable (-3 to -0.5)")
        checks_passed += 1
    elif elasticity < -3:
        print("  ⚠ Note: High elasticity suggests price-sensitive market")
    
    # Demand curve preview
    print("\n" + "-" * 60)
    print("Demand Curve (Price → Demand → Revenue):")
    print("-" * 60)
    sample_prices = np.array([20, 50, 75, 100, 150, 200, 300, 400])
    sample_prices = np.unique(np.append(sample_prices, optimal_price))
    sample_prices = sample_prices[sample_prices <= max(price_range)]
    sample_prices.sort()
    
    for p in sample_prices:
        d = exponential_demand(p, base_demand, price_sensitivity, reference_price)
        r = p * d
        bar_len = int(d * 50)
        # Mark if it's the optimal price (approximate equality for float)
        marker = " ◀ OPTIMAL" if abs(p - optimal_price) < 0.1 else ""
        print(f"  ${p:>4.0f}: {d:>5.1%} │{'█' * bar_len:<25}│ ${r:>6.2f}{marker}")
    
    print("-" * 60)
    print(f"\n✓ Verification complete. {checks_passed}/4 sanity checks passed.")


if __name__ == "__main__":
    main()
