#!/usr/bin/env python3
"""
Healthcare Distance Analysis for Tajikistan Schools
Following the pattern from getting_started.py

This script provides detailed analysis of distances between schools and healthcare facilities,
creating maps and statistics that can be used as input/output examples for further analysis.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def load_healthcare_facilities(data_dir):
    """Load and clean healthcare facilities data"""
    healthcare_file = Path(data_dir) / "tajikistan.csv"
    
    if not healthcare_file.exists():
        print(f"❌ Healthcare data not found at {healthcare_file}")
        return None
    
    # Load healthcare facilities
    healthcare_df = pd.read_csv(healthcare_file)
    
    # Clean data
    healthcare_df = healthcare_df.dropna(subset=['X', 'Y'])
    healthcare_df = healthcare_df[
        healthcare_df['amenity'].isin([
            'hospital', 'clinic', 'doctors', 'pharmacy', 'dentist'
        ])
    ]
    
    print(f"✓ Loaded {len(healthcare_df)} healthcare facilities")
    print(f"  - Hospitals: {len(healthcare_df[healthcare_df['amenity'] == 'hospital'])}")
    print(f"  - Clinics: {len(healthcare_df[healthcare_df['amenity'] == 'clinic'])}")
    print(f"  - Doctors: {len(healthcare_df[healthcare_df['amenity'] == 'doctors'])}")
    print(f"  - Pharmacies: {len(healthcare_df[healthcare_df['amenity'] == 'pharmacy'])}")
    print(f"  - Dentists: {len(healthcare_df[healthcare_df['amenity'] == 'dentist'])}")
    
    return healthcare_df

def load_education_facilities(data_dir):
    """Load and clean education facilities data"""
    education_file = Path(data_dir) / "hotosm_tjk_education_facilities_points_geojson.geojson"
    
    if not education_file.exists():
        print(f"❌ Education data not found at {education_file}")
        return None
    
    # Load education facilities
    education_gdf = gpd.read_file(education_file)
    
    # Filter for schools only
    schools_gdf = education_gdf[education_gdf['amenity'] == 'school'].copy()
    
    print(f"✓ Loaded {len(schools_gdf)} schools")
    print(f"  - Total education facilities: {len(education_gdf)}")
    print(f"  - Schools: {len(schools_gdf)}")
    print(f"  - Kindergartens: {len(education_gdf[education_gdf['amenity'] == 'kindergarten'])}")
    print(f"  - Universities: {len(education_gdf[education_gdf['amenity'] == 'university'])}")
    
    return schools_gdf

def calculate_distances_to_healthcare(schools_gdf, healthcare_df):
    """
    Calculate distances from each school to nearest healthcare facilities
    Returns comprehensive distance analysis
    """
    print("\n=== Calculating Distances to Healthcare Facilities ===")
    
    # Facility priority weights (for calculating weighted nearest facility)
    facility_priorities = {
        'hospital': 1.0,
        'clinic': 0.8,
        'doctors': 0.6,
        'pharmacy': 0.4,
        'dentist': 0.3
    }
    
    schools_coords = np.array([
        [point.x, point.y] for point in schools_gdf.geometry
    ])
    
    # Prepare results storage
    results = {
        'school_id': [],
        'school_name': [],
        'school_lon': [],
        'school_lat': [],
        'nearest_hospital_dist': [],
        'nearest_clinic_dist': [],
        'nearest_pharmacy_dist': [],
        'nearest_any_facility_dist': [],
        'nearest_facility_type': [],
        'nearest_facility_name': [],
        'hospitals_within_5km': [],
        'clinics_within_5km': [],
        'total_facilities_within_10km': [],
        'weighted_access_score': []
    }
    
    for idx, (school_idx, school) in enumerate(schools_gdf.iterrows()):
        school_coord = schools_coords[idx]
        
        results['school_id'].append(school_idx)
        results['school_name'].append(school.get('name', 'Unnamed School'))
        results['school_lon'].append(school_coord[0])
        results['school_lat'].append(school_coord[1])
        
        # Calculate distances to different facility types
        min_distances = {}
        facility_counts = {}
        
        for facility_type in facility_priorities.keys():
            facilities = healthcare_df[healthcare_df['amenity'] == facility_type]
            
            if len(facilities) > 0:
                facility_coords = np.array([
                    [row['X'], row['Y']] for _, row in facilities.iterrows()
                ])
                
                distances = cdist([school_coord], facility_coords, metric='euclidean')[0]
                min_distances[facility_type] = np.min(distances)
                
                # Count facilities within different radii
                # 0.05 degrees ≈ 5km, 0.1 degrees ≈ 10km (approximate)
                facility_counts[f'{facility_type}_5km'] = np.sum(distances < 0.05)
                facility_counts[f'{facility_type}_10km'] = np.sum(distances < 0.1)
            else:
                min_distances[facility_type] = float('inf')
                facility_counts[f'{facility_type}_5km'] = 0
                facility_counts[f'{facility_type}_10km'] = 0
        
        # Store specific facility distances
        results['nearest_hospital_dist'].append(
            min_distances.get('hospital', float('inf'))
        )
        results['nearest_clinic_dist'].append(
            min_distances.get('clinic', float('inf'))
        )
        results['nearest_pharmacy_dist'].append(
            min_distances.get('pharmacy', float('inf'))
        )
        
        # Find overall nearest facility
        nearest_dist = float('inf')
        nearest_type = 'None'
        nearest_name = 'Unknown'
        
        for facility_type, distance in min_distances.items():
            if distance < nearest_dist:
                nearest_dist = distance
                nearest_type = facility_type
                
                # Find the specific facility name
                facilities = healthcare_df[healthcare_df['amenity'] == facility_type]
                if len(facilities) > 0:
                    facility_coords = np.array([
                        [row['X'], row['Y']] for _, row in facilities.iterrows()
                    ])
                    distances = cdist([school_coord], facility_coords, metric='euclidean')[0]
                    nearest_idx = np.argmin(distances)
                    nearest_name = facilities.iloc[nearest_idx].get('name', 'Unnamed Facility')
        
        results['nearest_any_facility_dist'].append(nearest_dist if nearest_dist != float('inf') else None)
        results['nearest_facility_type'].append(nearest_type)
        results['nearest_facility_name'].append(nearest_name)
        
        # Store facility counts
        results['hospitals_within_5km'].append(facility_counts.get('hospital_5km', 0))
        results['clinics_within_5km'].append(facility_counts.get('clinic_5km', 0))
        
        total_facilities_10km = sum([
            facility_counts.get(f'{ft}_10km', 0) 
            for ft in facility_priorities.keys()
        ])
        results['total_facilities_within_10km'].append(total_facilities_10km)
        
        # Calculate weighted access score (lower is better access)
        access_score = 0
        for facility_type, priority in facility_priorities.items():
            if min_distances[facility_type] != float('inf'):
                # Weight by priority and distance
                access_score += min_distances[facility_type] / priority
        
        results['weighted_access_score'].append(access_score)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add risk categories based on access scores
    access_scores = results_df['weighted_access_score'].values
    access_scores = access_scores[~np.isinf(access_scores)]  # Remove infinities
    
    if len(access_scores) > 0:
        # Create risk categories based on quartiles
        q25, q75 = np.percentile(access_scores, [25, 75])
        
        def categorize_risk(score):
            if np.isinf(score):
                return 'Very High Risk'
            elif score <= q25:
                return 'Low Risk'
            elif score <= q75:
                return 'Medium Risk'
            else:
                return 'High Risk'
        
        results_df['access_risk_category'] = results_df['weighted_access_score'].apply(categorize_risk)
    
    print(f"✓ Distance analysis complete for {len(results_df)} schools")
    return results_df


def create_distance_visualizations(schools_gdf, healthcare_df, distance_results, output_dir="distance_analysis"):
    """Create comprehensive visualizations of distance analysis"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n=== Creating Distance Analysis Visualizations ===")
    
    # 1. Main distance map
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot schools colored by access risk
    risk_colors = {
        'Low Risk': 'green',
        'Medium Risk': 'yellow', 
        'High Risk': 'orange',
        'Very High Risk': 'red'
    }
    
    for risk_category, color in risk_colors.items():
        mask = distance_results['access_risk_category'] == risk_category
        if mask.any():
            ax.scatter(
                distance_results.loc[mask, 'school_lon'],
                distance_results.loc[mask, 'school_lat'],
                c=color, s=60, alpha=0.8, 
                label=f'{risk_category} ({mask.sum()} schools)',
                edgecolors='black', linewidth=0.5
            )
    
    # Plot healthcare facilities with different symbols
    facility_symbols = {
        'hospital': ('s', 'darkred', 100, 'Hospitals'),
        'clinic': ('D', 'blue', 60, 'Clinics'),
        'doctors': ('o', 'cyan', 40, 'Doctors'),
        'pharmacy': ('+', 'purple', 30, 'Pharmacies'),
        'dentist': ('x', 'orange', 25, 'Dentists')
    }
    
    for facility_type, (marker, color, size, label) in facility_symbols.items():
        facilities = healthcare_df[healthcare_df['amenity'] == facility_type]
        if len(facilities) > 0:
            ax.scatter(facilities['X'], facilities['Y'], 
                      marker=marker, c=color, s=size, alpha=0.7,
                      label=f'{label} ({len(facilities)})')
    
    # Draw connections for highest risk schools
    high_risk_schools = distance_results[
        distance_results['access_risk_category'].isin(['High Risk', 'Very High Risk'])
    ].head(10)  # Show top 10 highest risk
    
    for _, school in high_risk_schools.iterrows():
        if not np.isinf(school['nearest_any_facility_dist']):
            # Find the nearest facility coordinates
            nearest_type = school['nearest_facility_type']
            facilities = healthcare_df[healthcare_df['amenity'] == nearest_type]
            
            if len(facilities) > 0:
                school_coord = [school['school_lon'], school['school_lat']]
                facility_coords = np.array([
                    [row['X'], row['Y']] for _, row in facilities.iterrows()
                ])
                distances = cdist([school_coord], facility_coords, metric='euclidean')[0]
                nearest_idx = np.argmin(distances)
                nearest_facility = facilities.iloc[nearest_idx]
                
                ax.plot([school['school_lon'], nearest_facility['X']], 
                       [school['school_lat'], nearest_facility['Y']], 
                       'r--', alpha=0.5, linewidth=1)
    
    ax.set_title('Tajikistan Schools: Healthcare Access Analysis\n' + 
                'Distance-Based Risk Assessment', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'healthcare_distance_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Distance distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distance to nearest facility histogram
    valid_distances = distance_results['nearest_any_facility_dist'].dropna()
    axes[0,0].hist(valid_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of Distances to Nearest Healthcare Facility')
    axes[0,0].set_xlabel('Distance (degrees)')
    axes[0,0].set_ylabel('Number of Schools')
    axes[0,0].grid(True, alpha=0.3)
    
    # Risk category distribution
    risk_counts = distance_results['access_risk_category'].value_counts()
    colors = ['green', 'yellow', 'orange', 'red']
    axes[0,1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                 colors=colors[:len(risk_counts)], startangle=90)
    axes[0,1].set_title('Healthcare Access Risk Distribution')
    
    # Facility counts within 5km
    facility_counts_5km = [
        distance_results['hospitals_within_5km'].sum(),
        distance_results['clinics_within_5km'].sum()
    ]
    axes[1,0].bar(['Hospitals within 5km', 'Clinics within 5km'], facility_counts_5km,
                 color=['darkred', 'blue'], alpha=0.7)
    axes[1,0].set_title('Healthcare Facilities within 5km of Schools')
    axes[1,0].set_ylabel('Total Count')
    axes[1,0].grid(True, alpha=0.3)
    
    # Distance by facility type boxplot
    distance_by_type = []
    facility_types = []
    
    for ftype in ['hospital', 'clinic', 'pharmacy']:
        col = f'nearest_{ftype}_dist'
        if col in distance_results.columns:
            valid_vals = distance_results[col][~np.isinf(distance_results[col])]
            distance_by_type.extend(valid_vals)
            facility_types.extend([ftype.title()] * len(valid_vals))
    
    if distance_by_type:
        df_plot = pd.DataFrame({
            'Distance': distance_by_type,
            'Facility Type': facility_types
        })
        
        import seaborn as sns
        sns.boxplot(data=df_plot, x='Facility Type', y='Distance', ax=axes[1,1])
        axes[1,1].set_title('Distance Distribution by Facility Type')
        axes[1,1].set_ylabel('Distance (degrees)')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'distance_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Visualizations saved to {output_path}")

def export_distance_analysis(distance_results, output_dir="distance_analysis"):
    """Export distance analysis results"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Export detailed results
    distance_results.to_csv(output_path / 'school_healthcare_distances.csv', index=False)
    
    # Create summary statistics
    summary_stats = {
        'total_schools': len(distance_results),
        'schools_with_healthcare_access': len(distance_results.dropna(subset=['nearest_any_facility_dist'])),
        'avg_distance_to_nearest_facility': float(distance_results['nearest_any_facility_dist'].mean()),
        'median_distance_to_nearest_facility': float(distance_results['nearest_any_facility_dist'].median()),
        'max_distance_to_nearest_facility': float(distance_results['nearest_any_facility_dist'].max()),
        'schools_within_5km_of_hospital': int(distance_results['hospitals_within_5km'].sum()),
        'schools_within_5km_of_clinic': int(distance_results['clinics_within_5km'].sum()),
        'risk_distribution': distance_results['access_risk_category'].value_counts().to_dict(),
        'top_10_highest_risk_schools': distance_results.nlargest(10, 'weighted_access_score')[
            ['school_name', 'nearest_facility_type', 'nearest_any_facility_dist', 'access_risk_category']
        ].to_dict('records')
    }
    
    import json
    with open(output_path / 'distance_analysis_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print(f"\n=== Distance Analysis Summary ===")
    print(f"📊 Total schools analyzed: {summary_stats['total_schools']}")
    print(f"🏥 Schools with healthcare access: {summary_stats['schools_with_healthcare_access']}")
    print(f"📏 Average distance to nearest facility: {summary_stats['avg_distance_to_nearest_facility']:.4f} degrees")
    print(f"📏 Median distance to nearest facility: {summary_stats['median_distance_to_nearest_facility']:.4f} degrees")
    print(f"🔴 High/Very High Risk schools: {summary_stats['risk_distribution'].get('High Risk', 0) + summary_stats['risk_distribution'].get('Very High Risk', 0)}")
    
    return summary_stats

def main():
    """Main function following getting_started.py pattern"""
    
    print("=== Tajikistan Healthcare Distance Analysis ===")
    print("Building on getting_started.py methodology\n")
    
    data_dir = "data"
    
    # Load data
    print("1. Loading healthcare facilities...")
    healthcare_df = load_healthcare_facilities(data_dir)
    if healthcare_df is None:
        return
    
    print("\n2. Loading education facilities...")
    schools_gdf = load_education_facilities(data_dir)
    if schools_gdf is None:
        return
    
    # Calculate distances
    print("\n3. Calculating distances...")
    distance_results = calculate_distances_to_healthcare(schools_gdf, healthcare_df)
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    create_distance_visualizations(schools_gdf, healthcare_df, distance_results)
    
    # Export results
    print("\n5. Exporting results...")
    summary_stats = export_distance_analysis(distance_results)
    
    print("\n" + "="*50)
    print("✅ HEALTHCARE DISTANCE ANALYSIS COMPLETE!")
    print("📁 Results saved to: distance_analysis/")
    print("📊 Use these results as input for composite risk analysis")
    print("🔗 Compatible with getting_started.py workflow")
    
    return distance_results, summary_stats

if __name__ == "__main__":
    results, summary = main()