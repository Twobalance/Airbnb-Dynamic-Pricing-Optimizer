"""
Visualize Airbnb Pricing Results

Generates a dashboard of 4 charts:
1. Demand Curve (Price vs Probability)
2. Revenue Curve (Price vs Expected Revenue)
3. Price Elasticity Curve
4. City-Level Optimal vs Median Prices
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import logic from verify_logic
sys.path.append('scripts')
from verify_logic import estimate_market_parameters, find_optimal_price, exponential_demand

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

OUTPUT_FILE = "pricing_results_chart.png"
DATA_FILE = "data/processed/europe_airbnb.csv"

def generate_charts():
    print("Generating charts...")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return
        
    df = pd.read_csv(DATA_FILE)
    
    # 2. Get Overall Market Parameters
    base_demand, sensitivity, ref_price = estimate_market_parameters(df)
    
    # 3. Get Optimization Results
    opt_price, opt_prob, max_rev, elasticity, price_range, demands = \
        find_optimal_price(base_demand, sensitivity, ref_price, min_price=20, max_price=500)
    
    revenues = price_range * demands
    
    # Calculate Elasticity Curve
    elasticities = []
    for i in range(len(price_range)):
        p = price_range[i]
        d = demands[i]
        # Point elasticity: (dD/dP) * (P/D)
        # derivative of exp demand: -sensitivity/ref_price * demand
        deriv = -sensitivity / ref_price * d
        e = deriv * (p / d)
        elasticities.append(e)
    
    # 4. Prepare City Data
    city_data = []
    for city in df['city'].unique():
        city_df = df[df['city'] == city]
        if len(city_df) > 100:
            c_demand, c_sens, c_ref = estimate_market_parameters(city_df)
            c_opt, _, _, _, _, _ = find_optimal_price(c_demand, c_sens, c_ref)
            city_data.append({
                'City': city,
                'Median Price': c_ref,
                'Optimal Price': c_opt,
                'Difference': (c_opt - c_ref) / c_ref * 100
            })
    city_df = pd.DataFrame(city_data).sort_values('Optimal Price', ascending=False)
    
    # 5. Create Plot
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    
    # Chart A: Demand Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(price_range, demands * 100, color='blue', linewidth=2)
    ax1.axvline(opt_price, color='green', linestyle='--', alpha=0.7, label=f'Optimal: ${opt_price:.0f}')
    ax1.axhline(opt_prob * 100, color='green', linestyle=':', alpha=0.5)
    ax1.scatter([opt_price], [opt_prob * 100], color='green', zorder=5)
    ax1.set_title(f"Demand Curve (Elasticity: {elasticity:.2f})", fontweight='bold')
    ax1.set_xlabel("Price per Night ($)")
    ax1.set_ylabel("Booking Probability (%)")
    ax1.legend()
    ax1.fill_between(price_range, 0, demands * 100, alpha=0.1, color='blue')
    
    # Chart B: Revenue Curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(price_range, revenues, color='purple', linewidth=2)
    ax2.axvline(opt_price, color='green', linestyle='--', alpha=0.7, label=f'Max Revenue: ${max_rev:.0f}')
    ax2.scatter([opt_price], [max_rev], color='green', zorder=5)
    ax2.set_title("Expected Revenue Curve", fontweight='bold')
    ax2.set_xlabel("Price per Night ($)")
    ax2.set_ylabel("Expected Revenue ($)")
    ax2.legend()
    ax2.fill_between(price_range, 0, revenues, alpha=0.1, color='purple')
    
    # Chart C: City Comparison
    ax3 = fig.add_subplot(gs[1, :])  # Span bottom row
    
    x = np.arange(len(city_df))
    width = 0.35
    
    rects1 = ax3.bar(x - width/2, city_df['Median Price'], width, label='Market Median', color='gray', alpha=0.6)
    rects2 = ax3.bar(x + width/2, city_df['Optimal Price'], width, label='Optimal Price', color='green', alpha=0.8)
    
    ax3.set_ylabel('Price ($)')
    ax3.set_title('Market Median vs. Optimal Price by City', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(city_df['City'])
    ax3.legend()
    
    # Add labels on bars
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        diff = city_df.iloc[i]['Difference']
        ax3.text(rect.get_x() + rect.get_width()/2., height + 5,
                f'{diff:+.0f}%',
                ha='center', va='bottom', fontsize=9, color='green' if diff > 0 else 'red')

    # Save
    plt.suptitle("Airbnb Dynamic Pricing Analysis (Europe)", fontsize=16, fontweight='bold', y=1.05)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {os.path.abspath(OUTPUT_FILE)}")

def generate_report():
    """Run verification logic and save output to results.txt."""
    print("Generating detailed text report...")
    output_path = "results.txt"
    
    # Import main from verify_logic to run the full analysis
    from verify_logic import main as verify_main
    from contextlib import redirect_stdout
    
    with open(output_path, 'w') as f:
        with redirect_stdout(f):
            verify_main()
            
    print(f"Report saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_charts()
    generate_report()
