#!/usr/bin/env python3
"""
Script to download and setup Kaggle Airbnb Europe dataset.

This script provides two methods:
1. Using Kaggle API (requires API key)
2. Manual download instructions

Dataset: Airbnb Prices in European Cities
URL: https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities
"""
import os
import subprocess
import sys

DATASET_NAME = "thedevastator/airbnb-prices-in-european-cities"
OUTPUT_DIR = "data/kaggle"

def check_kaggle_api():
    """Check if Kaggle API is configured."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    return os.path.exists(kaggle_json)

def download_with_kaggle_api():
    """Download using Kaggle API."""
    print("Downloading dataset using Kaggle API...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", OUTPUT_DIR, "--unzip"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print(f"\n✓ Dataset downloaded to {OUTPUT_DIR}/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e.stderr}")
        return False

def print_manual_instructions():
    """Print manual download instructions."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           MANUAL DOWNLOAD INSTRUCTIONS                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Go to: https://www.kaggle.com/datasets/thedevastator/       ║
║           airbnb-prices-in-european-cities                       ║
║                                                                  ║
║  2. Sign in / Create a Kaggle account (free)                    ║
║                                                                  ║
║  3. Click "Download" button (top right)                         ║
║                                                                  ║
║  4. Unzip the downloaded file to:                               ║
║     data/kaggle/                                                 ║
║                                                                  ║
║  Expected files:                                                 ║
║     • amsterdam_weekdays.csv, amsterdam_weekends.csv            ║
║     • athens_weekdays.csv, athens_weekends.csv                  ║
║     • barcelona_weekdays.csv, barcelona_weekends.csv            ║
║     • berlin_weekdays.csv, berlin_weekends.csv                  ║
║     • budapest_weekdays.csv, budapest_weekends.csv              ║
║     • lisbon_weekdays.csv, lisbon_weekends.csv                  ║
║     • london_weekdays.csv, london_weekends.csv                  ║
║     • paris_weekdays.csv, paris_weekends.csv                    ║
║     • rome_weekdays.csv, rome_weekends.csv                      ║
║     • vienna_weekdays.csv, vienna_weekends.csv                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

def setup_kaggle_api():
    """Instructions to setup Kaggle API."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           KAGGLE API SETUP INSTRUCTIONS                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Go to: https://www.kaggle.com/settings                      ║
║                                                                  ║
║  2. Scroll to "API" section                                     ║
║                                                                  ║
║  3. Click "Create New Token" - downloads kaggle.json            ║
║                                                                  ║
║  4. Move the file:                                              ║
║     mv ~/Downloads/kaggle.json ~/.kaggle/                       ║
║     chmod 600 ~/.kaggle/kaggle.json                             ║
║                                                                  ║
║  5. Re-run this script                                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

def main():
    print("=" * 60)
    print("KAGGLE AIRBNB EUROPE DATASET DOWNLOADER")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if check_kaggle_api():
        print("\n✓ Kaggle API key found!")
        if download_with_kaggle_api():
            print("\n✓ Download complete! Run `python scripts/process_kaggle_data.py` next.")
        else:
            print("\n⚠ API download failed. Try manual download:")
            print_manual_instructions()
    else:
        print("\n⚠ Kaggle API key not found at ~/.kaggle/kaggle.json")
        print("\nYou have two options:\n")
        print("Option 1: Manual Download (Recommended)")
        print_manual_instructions()
        print("\nOption 2: Setup Kaggle API for automated downloads")
        setup_kaggle_api()

if __name__ == "__main__":
    main()
