#!/usr/bin/env python3
"""
Quick Start Script for Tajikistan School Risk Analysis

This script helps you get started immediately with your geospatial data.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add your project directory to the path
sys.path.append('/Users/alice/Developer/UN-Tech-Over2')

def quick_data_exploration():
    """Quickly explore your available data files."""
    
    data_dir = Path('/Users/alice/Developer/UN-Tech-Over2')
    
    print("=== Data Directory Exploration ===")
    print(f"Base directory: {data_dir}")
    print()
    
    # Check for education facility folders
    edu_folders = [d for d in data_dir.iterdir() if d.is_dir() and 'education_facilities' in d.name]
    print(f"Education facility folders found: {len(edu_folders)}")
    for folder in edu_folders:
        print(f"  - {folder.name}")
        files = list(folder.glob('*'))
        if files:
            for file in files:
                if file.suffix in ['.geojson', '.gpkg', '.shp']:
                    try:
                        gdf = gpd.read_file(file)
                        print(f"    * {file.name}: {len(gdf)} features")
                        if len(gdf) > 0:
                            print(f"      Columns: {list(gdf.columns)[:5]}...")  # First 5 columns
                    except Exception as e:
                        print(f"    * {file.name}: Error reading - {e}")
    
    print()
    
    # Check for Tajikistan base files
    base_files = [f for f in data_dir.iterdir() if f.is_file() and 'tajikistan' in f.name]
    print(f"Tajikistan base files found: {len(base_files)}")
    for file in base_files:
        print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
        
        if file.suffix in ['.geojson', '.csv']:
            try:
                if file.suffix == '.geojson':
                    data = gpd.read_file(file)
                    print(f"    * GeoDataFrame: {len(data)} features")
                elif file.suffix == '.csv':
                    data = pd.read_csv(file)
                    print(f"    * DataFrame: {len(data)} rows, {len(data.columns)} columns")
                
                if len(data) > 0:
                    print(f"      Columns: {list(data.columns)[:8]}...")  # First 8 columns
                    
            except Exception as e:
                print(f"    * Error reading: {e}")


def create_simple_map():
    """Create a simple map with whatever data is available."""
    
    data_dir = Path('/Users/alice/Developer/UN-Tech-Over2')
    
    print("=== Creating Simple Map ===")
    
    # Try to load any available geospatial data
    schools_data = None
    base_data = None
    
    # Try education facilities first
    edu_folders = [d for d in data_dir.iterdir() if d.is_dir() and 'education_facilities' in d.name]
    
    for folder in edu_folders:
        for file in folder.glob('*.geojson'):
            try:
                schools_data = gpd.read_file(file)
                print(f"Loaded schools data from: {file.name}")
                print(f"  Features: {len(schools_data)}")
                print(f"  Columns: {list(schools_data.columns)}")
                break
            except Exception as e:
                print(f"Could not load {file}: {e}")
        if schools_data is not None:
            break
    
    # Try base Tajikistan data
    for file in data_dir.glob('tajikistan*.geojson'):
        try:
            base_data = gpd.read_file(file)
            print(f"Loaded base data from: {file.name}")
            print(f"  Features: {len(base_data)}")
            break
        except Exception as e:
            print(f"Could not load {file}: {e}")
    
    # Create map
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot base data if available
    if base_data is not None:
        base_data.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)
        print("Added base geographic data to map")
    
    # Plot schools if available
    if schools_data is not None:
        schools_data.plot(ax=ax, color='red', markersize=20, alpha=0.7)
        print("Added schools data to map")
    
    # If no data loaded, create sample map
    if schools_data is None and base_data is None:
        print("No geospatial data found. Creating sample map...")
        
        # Create sample data for Tajikistan
        np.random.seed(42)
        n_schools = 50
        
        # Tajikistan approximate bounds
        lon_min, lon_max = 67.0, 75.0
        lat_min, lat_max = 36.5, 41.0
        
        lons = np.random.uniform(lon_min, lon_max, n_schools)
        lats = np.random.uniform(lat_min, lat_max, n_schools)
        
        ax.scatter(lons, lats, c='red', s=50, alpha=0.7, label='Sample Schools')
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
    
    # Customize map
    ax.set_title('Tajikistan Education Facilities Map', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    
    # Save map
    output_file = 'tajikistan_simple_map.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Map saved as: {output_file}")
    
    plt.show()


def setup_environment():
    """Check and setup the required environment."""
    
    print("=== Environment Setup ===")
    
    required_packages = [
        'pandas', 'geopandas', 'matplotlib', 'numpy', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✓ All required packages are available!")
    return True


def main():
    """Main function to run the quick start."""
    
    print("=== Tajikistan School Risk Mapping - Quick Start ===\n")
    
    # Setup environment
    if not setup_environment():
        print("Please install missing packages and try again.")
        return
    
    print()
    
    # Explore available data
    quick_data_exploration()
    
    print()
    
    # Create simple map
    create_simple_map()
    
    print("\n=== Next Steps ===")
    print("1. Review the created map")
    print("2. If data loaded successfully, run the full TajikistanRiskMapper")
    print("3. Customize risk weights and factors based on your requirements")
    print("4. Add your own hospital/healthcare facility data")
    print("5. Integrate real elevation and road network data")


if __name__ == "__main__":
    main()