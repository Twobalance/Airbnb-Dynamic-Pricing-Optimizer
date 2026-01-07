# Airbnb Dynamic Pricing Optimization Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]()

## 🚀 Project Overview
This project implements a **theoretically sound Dynamic Pricing Engine** for short-term rentals. Unlike traditional regression models, this engine estimates a probabilistic **Economic Demand Curve** ($D(P)$) to find the price that maximizes Expected Revenue ($E[R]$).

It overcomes common pitfalls in dynamic pricing (like linear regression on binary outcomes) by using **Probability Calibration** and **Exponential Demand Modeling**.

## 📊 Key Features
- **Economic Demand Modeling**: Uses `D(P) = D_base * exp(-λ * (P - P_ref) / P_ref)` to model realistic price elasticity.
- **Revenue Maximization**: Optimizes $R = P \times D(P)$ to find the exact "sweet spot" price.
- **Kaggle Data Integration**: Built-in pipelines to download and process the [Airbnb Prices in European Cities](https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities) dataset (50k+ listings).
- **City-Level Intelligence**: Automatically segments pricing analysis by city (e.g., Amsterdam vs. Lisbon).
- **Visualization Dashboard**: Auto-generates revenue curves, demand curves, and city comparison charts.

## 🛠️ Project Structure
```text
AIRBNB/
├── scripts/
│   ├── verify_logic.py          # 🧠 Core logic verification & optimization
│   ├── process_kaggle_data.py   # 🧹 ETL pipeline for Kaggle data
│   ├── download_kaggle_data.py  # 📥 Data download helper
│   └── visualize_results.py     # 📈 Generates charts & reports
├── src/
│   ├── pricing_engine.py        # ⚙️ Reusable PricingEngine class
│   └── data_processing.py       # 🛠️ Feature engineering utils
├── data/
│   ├── kaggle/                  # Raw CSV inputs
│   └── processed/               # Combined parquet/csv datasets
└── results.txt                  # Output from latest analysis
```

## ⚡ Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get Data
You can use the built-in downloader to get the Kaggle dataset:
```bash
python scripts/download_kaggle_data.py
python scripts/process_kaggle_data.py
```

### 3. Run Analysis & Visualization
Run the all-in-one visualization script to generate both the **charts** and **text report**:
```bash
python scripts/visualize_results.py
```
This produces:
- `pricing_results_chart.png`: Visual dashboard of demand/revenue.
- `results.txt`: Detailed numerical breakdown by city.

## 📈 Methodology
The engine moves beyond simple prediction to **optimization**:
1. **Market Parameter Estimation**: Calculates base demand ($D_0$) and price sensitivity ($\lambda$) from market data.
2. **Demand Curve Construction**: Builds a continuous function $D(P)$ representing booking probability at any price $P$.
3. **Revenue Optimization**: Finds $P^*$ where $\frac{d}{dP}(P \cdot D(P)) = 0$.
4. **Sanity Constraints**: Applies realistic probability caps (e.g., max 50% occupancy) to prevent over-optimistic predictions.

## 🏆 Results (European Market)
On a dataset of 51,000+ listings across 10 cities:
- **Optimal Price**: $189.70 (vs Median $211)
- **Revenue Uplift**: **+12.4%** expected revenue increase vs median pricing.
- **Elasticity**: -0.95 (Unit elastic demand).

---
*Created for Quantitative Finance & Data Science Portfolio.*
