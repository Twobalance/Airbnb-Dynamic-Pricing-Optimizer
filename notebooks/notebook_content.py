# %% [markdown]
# # Dynamic Pricing Optimization for Short-Term Rentals
# 
# **Role:** Senior Data Scientist & Management Engineer
# 
# **Objective:** Replace static pricing with a dynamic algorithm to maximize revenue ($R = P \times D$) using XGBoost and Mathematical Optimization.
# 
# **Architecture:**
# 1.  **Data Processing**: Cleaning and Merging (Internal Logic in `src/data_processing.py`)
# 2.  **Pricing Engine**: XGBoost Demand Estimation & Optimal Price Derivation (`src/pricing_engine.py`)
# 3.  **Visualization**: Professional Charting & Revenue Simulation (`src/visualization.py`)

# %%
import os
import sys
import numpy as np
import pandas as pd
import warnings

# Ensure root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.data_processing import load_and_preprocess_data
from src.pricing_engine import PricingEngine
from src.visualization import plot_demand_curve, print_revenue_comparison

warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Data Preprocessing
# We load the listings and calendar data from the InsideAirbnb dataset.

# %%
# Define paths relative to root or absolute
LISTINGS_PATH = os.path.join(ROOT_DIR, 'data/raw/listings.csv')
CALENDAR_PATH = os.path.join(ROOT_DIR, 'data/raw/calendar.csv')

# Load and process
df = load_and_preprocess_data(LISTINGS_PATH, CALENDAR_PATH)
print(f"Dataset ready with shape: {df.shape}")
df.head()

# %% [markdown]
# ## 2. Demand Estimation (XGBoost)
# We train an XGBoost regressor to estimate the probability of a booking (Demand) based on price and seasonal features.

# %%
features = ['price', 'month', 'day_of_week', 'is_weekend', 
            'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365', 'room_type']

engine = PricingEngine(features)
X_test, y_test = engine.train(df)

# %% [markdown]
# ## 3. Mathematical Optimization & Demand Curve
# We assume a linear demand curve $D(P) = \alpha - \beta P$ locally and solve for the revenue-maximizing price $P^* = \frac{\alpha}{2\beta}$.

# %%
# Select a sample listing to demonstrate optimization
print("Searching for a listing with valid price elasticity...")
found_valid = False

for sample_idx in range(len(X_test)):
    sample_row = X_test.iloc[sample_idx]
    actual_price = sample_row['price']
    
    # Run Optimization
    opt_price, alpha, beta, prices, demands = engine.get_optimal_price(sample_row)
    
    if not np.isnan(opt_price) and opt_price > 0:
        print(f"\nOptimization Successful for Listing index {sample_idx}")
        
        # Visualize Results
        plot_demand_curve(
            prices, demands, alpha, beta, opt_price, actual_price, 
            save_path=os.path.join(ROOT_DIR, 'assets/optimization_chart.png')
        )
        
        # Revenue Simulation
        current_demand = alpha - beta * actual_price
        optimal_demand = alpha - beta * opt_price
        
        print_revenue_comparison(actual_price, current_demand, opt_price, optimal_demand)
        
        found_valid = True
        break

if not found_valid:
    print("Optimization failed to find a valid downward sloping curve in the test set.")

# %% [markdown]
# ## 4. Conclusion
# This dynamic pricing model demonstrates how machine learning can be combined with classical economic optimization to drive significant revenue uplift in the hospitality industry.
