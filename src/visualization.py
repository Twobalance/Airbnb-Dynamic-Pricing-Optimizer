import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_plotting_style():
    """
    Sets up a professional plotting style for academic presentation.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.family'] = 'sans-serif'

def plot_demand_curve(prices, demands, alpha, beta, opt_price, actual_price, save_path=None):
    """
    Plots the predicted demand curve and optimization results.
    """
    setup_plotting_style()
    
    plt.figure()
    plt.plot(prices, demands, label='Predicted Demand (XGBoost)', color='royalblue', linewidth=2)
    plt.plot(prices, alpha - beta * prices, '--', label='Linear Approximation', color='tomato', alpha=0.8)
    
    plt.axvline(opt_price, color='forestgreen', linestyle=':', label=f'Optimal Price ${opt_price:.0f}', linewidth=2)
    plt.axvline(actual_price, color='dimgray', linestyle=':', label=f'Actual Price ${actual_price:.0f}', linewidth=2)
    
    plt.xlabel('Price ($)')
    plt.ylabel('Booking Probability (Demand)')
    plt.title('Demand Curve Reconstruction & Price Optimization')
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def print_revenue_comparison(actual_price, current_demand, opt_price, optimal_demand):
    """
    Prints a formatted revenue comparison.
    """
    current_rev = actual_price * current_demand
    optimal_rev = opt_price * optimal_demand
    uplift = ((optimal_rev - current_rev) / current_rev) * 100
    
    print("\n" + "="*30)
    print("REVENUE SIMULATION RESULTS")
    print("="*30)
    print(f"Current Structure:   ${actual_price:.2f} @ {current_demand:.3f} prompt = ${current_rev:.2f}/night")
    print(f"Optimized Structure: ${opt_price:.2f} @ {optimal_demand:.3f} prompt = ${optimal_rev:.2f}/night")
    print(f"Potential Uplift:    {uplift:.2f}%")
    print("="*30)
