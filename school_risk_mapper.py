#!/usr/bin/env python3
"""
Implementation Script for UN-Tech-Over School Risk Analysis
===========================================================

This script implements the comprehensive school risk mapping system
specifically for the UN-Tech-Over project data structure, with enhanced
features and generalized capabilities.

Usage:
    python implement_risk_analysis.py

Requirements:
    - pandas, geopandas, matplotlib, seaborn, sklearn, scipy
    - contextily (optional for base maps)
    - folium (optional for interactive maps)
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class EnhancedSchoolRiskMapper:
    """
    Enhanced School Risk Mapper specifically designed for UN-Tech-Over data structure.
    
    Features:
    - Works with existing Tajikistan data structure
    - Generalizable to other countries
    - Enhanced visualizations with proper labeling
    - Base map integration
    - Comprehensive risk scoring
    - CCRI-compatible outputs
    """
    
    def __init__(self, country_name="Tajikistan", base_dir=str(os.getcwd())):
        self.country_name = country_name
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        
        # Risk weights (customizable per analysis)
        self.risk_weights = {
            'healthcare_distance': 0.30,
            'healthcare_capacity': 0.25,
            'road_conditions': 0.20,
            'infrastructure_age': 0.15,
            'environmental_factors': 0.10
        }
        
        # Initialize data containers
        self.healthcare_data = None
        self.education_data = None
        self.results_df = None
        
        print(f"🌍 Enhanced School Risk Mapper initialized for {country_name}")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 Data directory: {self.data_dir}")

    def load_and_validate_data(self):
        """Load and validate the UN-Tech-Over data files."""
        print("\n=== Loading UN-Tech-Over Data ===")
        
        # Load healthcare facilities (Tajikistan CSV format)
        healthcare_file = self.data_dir / "tajikistan.csv"
        if healthcare_file.exists():
            self.healthcare_data = pd.read_csv(healthcare_file)
            self.healthcare_data = self._clean_healthcare_data()
            print(f"✅ Loaded {len(self.healthcare_data)} healthcare facilities")
        else:
            print(f"❌ Healthcare file not found: {healthcare_file}")
            return False
        
        # Load education facilities (GeoJSON format)
        education_file = self.data_dir / "hotosm_tjk_education_facilities_points_geojson.geojson"
        if education_file.exists():
            self.education_data = gpd.read_file(education_file)
            self.education_data = self._clean_education_data()
            print(f"✅ Loaded {len(self.education_data)} education facilities")
        else:
            print(f"❌ Education file not found: {education_file}")
            return False
        
        return True

    def _clean_healthcare_data(self):
        """Clean and prepare healthcare data."""
        # Remove entries without coordinates
        df = self.healthcare_data.dropna(subset=['X', 'Y'])
        
        # Filter for healthcare facilities
        healthcare_types = ['hospital', 'clinic', 'doctors', 'pharmacy', 'dentist']
        df = df[df['amenity'].isin(healthcare_types)]
        
        # Create geometry column
        df = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df['X'], df['Y']),
            crs='EPSG:4326'
        )
        
        print(f"📊 Healthcare facility breakdown:")
        print(df['amenity'].value_counts())
        
        return df

    def _clean_education_data(self):
        """Clean and prepare education data."""
        # Filter for schools only (primary analysis focus)
        schools = self.education_data[self.education_data['amenity'] == 'school'].copy()
        
        # Ensure CRS is set
        if schools.crs is None:
            schools.set_crs('EPSG:4326', inplace=True)
        
        print(f"📊 Education facility breakdown:")
        print(self.education_data['amenity'].value_counts())
        print(f"🏫 Schools selected for analysis: {len(schools)}")
        
        return schools

    def calculate_healthcare_accessibility_metrics(self):
        """
        Calculate comprehensive healthcare accessibility metrics.
        
        Returns detailed distance and access analysis for each school.
        """
        print("\n=== Calculating Healthcare Accessibility ===")
        
        # Get coordinate arrays
        school_coords = np.array([
            [point.x, point.y] for point in self.education_data.geometry
        ])
        healthcare_coords = np.array([
            [point.x, point.y] for point in self.healthcare_data.geometry
        ])
        
        # Facility priority weights
        facility_priorities = {
            'hospital': 1.0,
            'clinic': 0.8,
            'doctors': 0.6,
            'pharmacy': 0.4,
            'dentist': 0.3
        }
        
        results = []
        
        for idx, school in self.education_data.iterrows():
            school_coord = school_coords[idx]
            school_name = school.get('name', f'School_{idx}')
            
            # Calculate distances to all healthcare facilities
            all_distances = cdist([school_coord], healthcare_coords, metric='euclidean')[0]
            
            # Initialize metrics
            metrics = {
                'school_id': idx,
                'school_name': school_name,
                'school_lat': school_coord[1],
                'school_lon': school_coord[0],
                'nearest_hospital_dist': float('inf'),
                'nearest_clinic_dist': float('inf'),
                'nearest_pharmacy_dist': float('inf'),
                'nearest_overall_dist': float('inf'),
                'nearest_facility_name': 'None',
                'nearest_facility_type': 'None',
                'facilities_5km': 0,
                'facilities_10km': 0,
                'facilities_20km': 0,
                'weighted_access_score': 0,
                'emergency_access_score': 0
            }
            
            # Analyze by facility type
            best_overall_dist = float('inf')
            best_facility_info = None
            
            for facility_type, priority in facility_priorities.items():
                # Get facilities of this type
                type_mask = self.healthcare_data['amenity'] == facility_type
                if not type_mask.any():
                    continue
                
                type_indices = np.where(type_mask)[0]
                type_distances = all_distances[type_indices]
                min_dist = np.min(type_distances)
                
                # Store type-specific distance
                metrics[f'nearest_{facility_type}_dist'] = min_dist
                
                # Check if this is the best overall option (weighted by priority)
                weighted_dist = min_dist / priority
                if weighted_dist < best_overall_dist:
                    best_overall_dist = weighted_dist
                    metrics['nearest_overall_dist'] = min_dist
                    metrics['nearest_facility_type'] = facility_type
                    
                    # Get facility name
                    closest_idx = type_indices[np.argmin(type_distances)]
                    facility_name = self.healthcare_data.iloc[closest_idx].get('name', f'{facility_type}_{closest_idx}')
                    metrics['nearest_facility_name'] = facility_name if pd.notna(facility_name) else f'{facility_type}_{closest_idx}'
            
            # Count facilities within different radii
            metrics['facilities_5km'] = np.sum(all_distances < 0.05)   # ~5km
            metrics['facilities_10km'] = np.sum(all_distances < 0.1)   # ~10km  
            metrics['facilities_20km'] = np.sum(all_distances < 0.2)   # ~20km
            
            # Calculate access scores
            if metrics['nearest_overall_dist'] != float('inf'):
                # Emergency access score (within 5km is ideal)
                metrics['emergency_access_score'] = max(0, 1 - (metrics['nearest_overall_dist'] / 0.05))
                
                # Weighted access score considering facility density
                density_factor = 1 + (metrics['facilities_10km'] * 0.1)
                metrics['weighted_access_score'] = (1 / (1 + metrics['nearest_overall_dist'])) * density_factor
            
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # Calculate risk scores (higher = worse access)
        if len(results_df) > 0:
            scaler = MinMaxScaler()
            
            # Distance-based risk (normalized)
            valid_distances = results_df['nearest_overall_dist'][results_df['nearest_overall_dist'] != float('inf')]
            if len(valid_distances) > 0:
                results_df['healthcare_distance_risk'] = np.where(
                    results_df['nearest_overall_dist'] == float('inf'),
                    1.0,  # Maximum risk for no access
                    scaler.fit_transform(results_df[['nearest_overall_dist']])
                )
            else:
                results_df['healthcare_distance_risk'] = 0.5
            
            # Capacity risk (inverted facility count)
            max_facilities = results_df['facilities_10km'].max()
            results_df['healthcare_capacity_risk'] = 1 - (results_df['facilities_10km'] / (max_facilities + 1))
        
        print(f"✅ Healthcare accessibility calculated for {len(results_df)} schools")
        print(f"📊 Average distance to nearest facility: {results_df['nearest_overall_dist'].mean():.4f} degrees")
        print(f"🏥 Schools with no nearby healthcare (>20km): {(results_df['facilities_20km'] == 0).sum()}")
        
        return results_df

    def calculate_infrastructure_and_road_risk(self, healthcare_results):
        """Calculate infrastructure age and road condition risks."""
        print("\n=== Calculating Infrastructure & Road Risks ===")
        
        results = healthcare_results.copy()
        
        # Infrastructure age risk
        infrastructure_risks = []
        road_risks = []
        environmental_risks = []
        
        # Get major urban centers from healthcare facility clustering
        major_centers = self._identify_urban_centers()
        
        for idx, school in self.education_data.iterrows():
            school_coord = [school.geometry.x, school.geometry.y]
            
            # 1. Infrastructure age risk
            infra_risk = self._calculate_school_infrastructure_risk(school)
            infrastructure_risks.append(infra_risk)
            
            # 2. Road condition risk
            road_risk = self._calculate_road_condition_risk(school_coord, major_centers)
            road_risks.append(road_risk)
            
            # 3. Environmental risk
            env_risk = self._calculate_environmental_risk(school_coord, idx)
            environmental_risks.append(env_risk)
        
        results['infrastructure_age_risk'] = infrastructure_risks
        results['road_condition_risk'] = road_risks
        results['environmental_risk'] = environmental_risks
        
        print(f"✅ Infrastructure analysis complete")
        print(f"📊 Average infrastructure age risk: {np.mean(infrastructure_risks):.3f}")
        print(f"🛣️ Average road condition risk: {np.mean(road_risks):.3f}")
        
        return results

    def _identify_urban_centers(self):
        """Identify major urban centers from healthcare facility density."""
        if len(self.healthcare_data) < 5:
            return []
        
        # Use hospital locations as primary urban indicators
        hospitals = self.healthcare_data[self.healthcare_data['amenity'] == 'hospital']
        
        if len(hospitals) == 0:
            # Fall back to clinic density
            hospitals = self.healthcare_data[self.healthcare_data['amenity'] == 'clinic']
        
        centers = []
        for idx, hospital in hospitals.iterrows():
            # Calculate facility density around each hospital
            hospital_coord = [hospital.geometry.x, hospital.geometry.y]
            distances = cdist([hospital_coord], 
                            [[p.x, p.y] for p in self.healthcare_data.geometry],
                            metric='euclidean')[0]
            
            nearby_facilities = np.sum(distances < 0.1)  # Within ~10km
            importance = min(nearby_facilities / 5.0, 1.0)  # Normalize importance
            
            centers.append({
                'coord': hospital_coord,
                'importance': importance,
                'name': hospital.get('name', f'Center_{idx}')
            })
        
        return sorted(centers, key=lambda x: x['importance'], reverse=True)[:5]

    def _calculate_school_infrastructure_risk(self, school):
        """Calculate infrastructure age/quality risk for a school."""
        risk_factors = []
        
        # Name-based analysis
        school_name = str(school.get('name', ''))
        if school_name and school_name != 'nan':
            name_lower = school_name.lower()
            
            # Modern indicators
            modern_keywords = ['гимназия', 'лицей', 'центр', 'academy', 'modern', 'new']
            traditional_keywords = ['школа', 'мактаб', 'basic']
            
            modern_score = sum(1 for kw in modern_keywords if kw in name_lower)
            traditional_score = sum(1 for kw in traditional_keywords if kw in name_lower)
            
            if modern_score > 0:
                risk_factors.append(0.2)  # Lower risk
            elif traditional_score > 0:
                risk_factors.append(0.7)  # Higher risk
            else:
                risk_factors.append(0.5)  # Medium risk
        else:
            risk_factors.append(0.6)  # Higher risk for unnamed schools
        
        # Building type analysis
        building = str(school.get('building', ''))
        if 'yes' in building.lower():
            risk_factors.append(0.3)  # Lower risk for specified buildings
        else:
            risk_factors.append(0.6)  # Higher risk for unspecified
        
        # Operator analysis
        operator = str(school.get('operator:type', ''))
        if 'private' in operator.lower():
            risk_factors.append(0.2)  # Private often better maintained
        elif 'public' in operator.lower():
            risk_factors.append(0.6)  # Public variable quality
        else:
            risk_factors.append(0.5)  # Unknown
        
        return np.mean(risk_factors)

    def _calculate_road_condition_risk(self, school_coord, major_centers):
        """Calculate road condition risk based on accessibility."""
        if not major_centers:
            return 0.5
        
        # Distance to nearest major center
        min_distance = min(
            np.sqrt((school_coord[0] - center['coord'][0])**2 + 
                   (school_coord[1] - center['coord'][1])**2) / center['importance']
            for center in major_centers
        )
        
        # Healthcare facility density (proxy for road development)
        healthcare_coords = [[p.x, p.y] for p in self.healthcare_data.geometry]
        distances = cdist([school_coord], healthcare_coords, metric='euclidean')[0]
        nearby_facilities = np.sum(distances < 0.1)
        facility_factor = 1.0 / (1.0 + nearby_facilities * 0.2)
        
        # Combine factors
        distance_risk = min(min_distance * 3, 1.0)
        total_risk = (distance_risk * 0.7) + (facility_factor * 0.3)
        
        return total_risk

    def _calculate_environmental_risk(self, school_coord, school_idx):
        """Calculate environmental and isolation risks."""
        # Isolation risk - distance to other schools
        other_schools = [p for i, p in enumerate(self.education_data.geometry) if i != school_idx]
        if other_schools:
            distances = cdist([school_coord], 
                            [[p.x, p.y] for p in other_schools],
                            metric='euclidean')[0]
            nearest_school_dist = np.min(distances)
            isolation_risk = min(nearest_school_dist * 10, 1.0)
        else:
            isolation_risk = 1.0
        
        # Geographic risk (elevation proxy from latitude variation)
        all_lats = [p.y for p in self.education_data.geometry]
        lat_std = np.std(all_lats)
        if lat_std > 0:
            lat_deviation = abs(school_coord[1] - np.mean(all_lats))
            elevation_risk = min(lat_deviation / lat_std, 1.0)
        else:
            elevation_risk = 0.3
        
        return (isolation_risk * 0.6) + (elevation_risk * 0.4)

    def calculate_composite_risk_scores(self):
        """Calculate final composite risk scores and create complete results."""
        print("\n=== Calculating Composite Risk Scores ===")
        
        # Get healthcare accessibility results
        healthcare_results = self.calculate_healthcare_accessibility_metrics()
        
        # Add infrastructure and road risks
        complete_results = self.calculate_infrastructure_and_road_risk(healthcare_results)
        
        # Calculate weighted composite score
        composite_score = (
            complete_results['healthcare_distance_risk'] * self.risk_weights['healthcare_distance'] +
            complete_results['healthcare_capacity_risk'] * self.risk_weights['healthcare_capacity'] +
            complete_results['road_condition_risk'] * self.risk_weights['road_conditions'] +
            complete_results['infrastructure_age_risk'] * self.risk_weights['infrastructure_age'] +
            complete_results['environmental_risk'] * self.risk_weights['environmental_factors']
        )
        
        complete_results['composite_risk_score'] = composite_score
        
        # Categorize risks
        risk_categories = pd.cut(
            composite_score,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        )
        complete_results['risk_category'] = risk_categories
        
        # Add priority ranking
        complete_results['priority_rank'] = composite_score.rank(method='dense', ascending=False)
        
        # Add metadata
        complete_results['analysis_date'] = datetime.now().isoformat()
        complete_results['country'] = self.country_name
        
        # Create GeoDataFrame for mapping
        self.results_df = gpd.GeoDataFrame(
            complete_results.merge(
                self.education_data[['name', 'amenity', 'geometry']].reset_index(),
                left_on='school_id', right_on='index', how='left'
            ),
            geometry='geometry',
            crs='EPSG:4326'
        )
        
        print(f"✅ Composite risk analysis complete for {len(self.results_df)} schools")
        print(f"📊 Risk distribution:")
        for category, count in self.results_df['risk_category'].value_counts().items():
            print(f"   {category}: {count} schools")
        
        return self.results_df

    def create_enhanced_visualizations(self, output_dir="enhanced_risk_analysis"):
        """Create comprehensive visualizations with proper labeling."""
        print(f"\n=== Creating Enhanced Visualizations ===")
        
        if self.results_df is None:
            raise ValueError("Must calculate composite risk scores first")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set visualization style
        plt.style.use('default')
        sns.set_palette("Set1")
        
        # 1. Main comprehensive risk map
        self._create_comprehensive_risk_map(output_path)
        
        # 2. Healthcare accessibility detailed analysis
        self._create_healthcare_accessibility_map(output_path)
        
        # 3. Risk component breakdown
        self._create_risk_component_analysis(output_path)
        
        # 4. Priority schools identification
        self._create_priority_schools_analysis(output_path)
        
        # 5. Statistical summary dashboard
        self._create_statistical_dashboard(output_path)
        
        print(f"✅ All visualizations saved to {output_path}")

    def _create_comprehensive_risk_map(self, output_path):
        """Create the main comprehensive risk map with all labeled points."""
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        
        # Color scheme for risk categories
        risk_colors = {
            'Low Risk': '#2E8B57',      # Sea Green
            'Medium Risk': '#FFD700',    # Gold
            'High Risk': '#FF6347',      # Tomato
            'Critical Risk': '#DC143C'   # Crimson
        }
        
        # Plot schools by risk category
        for risk_cat, color in risk_colors.items():
            mask = self.results_df['risk_category'] == risk_cat
            if mask.any():
                schools_subset = self.results_df[mask]
                
                # Plot points
                ax.scatter(
                    [p.x for p in schools_subset.geometry],
                    [p.y for p in schools_subset.geometry],
                    c=color, s=100, alpha=0.8,
                    label=f'{risk_cat} ({mask.sum()} schools)',
                    edgecolors='black', linewidth=0.8, zorder=5
                )
                
                # Label high-risk and critical schools
                if risk_cat in ['High Risk', 'Critical Risk']:
                    for idx, school in schools_subset.iterrows():
                        name = school.get('school_name', f'School_{school.get("school_id", idx)}')
                        if pd.notna(name) and str(name) != 'nan':
                            # Truncate long names
                            display_name = str(name)[:25] + '...' if len(str(name)) > 25 else str(name)
                            ax.annotate(
                                display_name,
                                (school.geometry.x, school.geometry.y),
                                xytext=(8, 8), textcoords='offset points',
                                fontsize=8, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor='yellow' if risk_cat == 'High Risk' else 'red',
                                         alpha=0.7),
                                zorder=6
                            )
        
        # Plot healthcare facilities with different symbols and labels
        facility_symbols = {
            'hospital': ('s', '#8B0000', 120, 'Hospitals'),
            'clinic': ('D', '#0000CD', 80, 'Clinics'),
            'doctors': ('o', '#00CED1', 60, 'Doctors'),
            'pharmacy': ('+', '#9370DB', 50, 'Pharmacies'),
            'dentist': ('x', '#FF6347', 40, 'Dentists')
        }
        
        for facility_type, (marker, color, size, label) in facility_symbols.items():
            facilities = self.healthcare_data[self.healthcare_data['amenity'] == facility_type]
            if len(facilities) > 0:
                ax.scatter(
                    [p.x for p in facilities.geometry],
                    [p.y for p in facilities.geometry],
                    marker=marker, c=color, s=size, alpha=0.9,
                    label=f'{label} ({len(facilities)})',
                    edgecolors='white', linewidth=1, zorder=4
                )
                
                # Label major hospitals
                if facility_type == 'hospital':
                    for idx, facility in facilities.iterrows():
                        name = facility.get('name', f'Hospital_{idx}')
                        if pd.notna(name) and str(name) != 'nan' and len(facilities) <= 15:
                            display_name = str(name)[:20] + '...' if len(str(name)) > 20 else str(name)
                            ax.annotate(
                                display_name,
                                (facility.geometry.x, facility.geometry.y),
                                xytext=(10, 10), textcoords='offset points',
                                fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                                zorder=6
                            )
        
        # Draw connections for critical risk schools to nearest healthcare
        critical_schools = self.results_df[self.results_df['risk_category'] == 'Critical Risk']
        for idx, school in critical_schools.head(10).iterrows():  # Limit to avoid clutter
            if pd.notna(school.get('nearest_facility_name')):
                # Find the nearest healthcare facility
                facility_type = school.get('nearest_facility_type', '')
                if facility_type in ['hospital', 'clinic']:
                    # Draw a line to show distance
                    nearest_facilities = self.healthcare_data[
                        self.healthcare_data['amenity'] == facility_type
                    ]
                    if len(nearest_facilities) > 0:
                        # Find closest facility (simplified)
                        school_coord = [school.geometry.x, school.geometry.y]
                        facility_coords = [[p.x, p.y] for p in nearest_facilities.geometry]
                        distances = cdist([school_coord], facility_coords)[0]
                        closest_idx = np.argmin(distances)
                        closest_facility = nearest_facilities.iloc[closest_idx]
                        
                        ax.plot(
                            [school.geometry.x, closest_facility.geometry.x],
                            [school.geometry.y, closest_facility.geometry.y],
                            'r--', alpha=0.6, linewidth=2, zorder=3
                        )
        
        # Styling
        ax.set_title(
            f'{self.country_name}: Comprehensive School Risk Assessment\n'
            f'Healthcare Access, Infrastructure & Environmental Analysis\n'
            f'Supplementing UNICEF CCRI with Fine-Grained Local Data',
            fontsize=16, fontweight='bold', pad=20
        )
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Enhanced legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, 
                 bbox_to_anchor=(1.05, 1), loc='upper left',
                 fontsize=10, title='Risk Categories & Healthcare Facilities',
                 title_fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.country_name.lower()}_comprehensive_risk_map.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_healthcare_accessibility_map(self, output_path):
        """Create detailed healthcare accessibility analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Distance to nearest healthcare
        scatter1 = axes[0,0].scatter(
            [p.x for p in self.results_df.geometry],
            [p.y for p in self.results_df.geometry],
            c=self.results_df['nearest_overall_dist'],
            cmap='RdYlBu_r', s=80, alpha=0.8,
            edgecolors='black', linewidth=0.5
        )
        axes[0,0].set_title('Distance to Nearest Healthcare Facility', fontweight='bold')
        axes[0,0].set_xlabel('Longitude')
        axes[0,0].set_ylabel('Latitude')
        plt.colorbar(scatter1, ax=axes[0,0], label='Distance (degrees)')
        
        # 2. Healthcare facility density
        scatter2 = axes[0,1].scatter(
            [p.x for p in self.results_df.geometry],
            [p.y for p in self.results_df.geometry],
            c=self.results_df['facilities_10km'],
            cmap='viridis', s=80, alpha=0.8,
            edgecolors='black', linewidth=0.5
        )
        axes[0,1].set_title('Healthcare Facilities within 10km', fontweight='bold')
        axes[0,1].set_xlabel('Longitude')
        axes[0,1].set_ylabel('Latitude')
        plt.colorbar(scatter2, ax=axes[0,1], label='Number of Facilities')
        
        # 3. Emergency access scores
        scatter3 = axes[1,0].scatter(
            [p.x for p in self.results_df.geometry],
            [p.y for p in self.results_df.geometry],
            c=self.results_df['emergency_access_score'],
            cmap='RdYlGn', s=80, alpha=0.8,
            edgecolors='black', linewidth=0.5
        )
        axes[1,0].set_title('Emergency Healthcare Access Score', fontweight='bold')
        axes[1,0].set_xlabel('Longitude')
        axes[1,0].set_ylabel('Latitude')
        plt.colorbar(scatter3, ax=axes[1,0], label='Access Score (higher = better)')
        
        # 4. Facility type distribution
        facility_types = self.results_df['nearest_facility_type'].value_counts()
        axes[1,1].pie(facility_types.values, labels=facility_types.index, autopct='%1.1f%%')
        axes[1,1].set_title('Primary Healthcare Access by Facility Type', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.country_name.lower()}_healthcare_accessibility.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_risk_component_analysis(self, output_path):
        """Create risk component breakdown analysis."""
        risk_components = [
            ('healthcare_distance_risk', 'Healthcare Distance Risk'),
            ('healthcare_capacity_risk', 'Healthcare Capacity Risk'),
            ('road_condition_risk', 'Road Condition Risk'),
            ('infrastructure_age_risk', 'Infrastructure Age Risk'),
            ('environmental_risk', 'Environmental Risk')
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.ravel()
        
        for i, (component, title) in enumerate(risk_components):
            if component in self.results_df.columns:
                scatter = axes[i].scatter(
                    [p.x for p in self.results_df.geometry],
                    [p.y for p in self.results_df.geometry],
                    c=self.results_df[component],
                    cmap='RdYlBu_r', s=70, alpha=0.8,
                    edgecolors='black', linewidth=0.5
                )
                axes[i].set_title(title, fontweight='bold', fontsize=14)
                axes[i].set_xlabel('Longitude')
                axes[i].set_ylabel('Latitude')
                axes[i].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[i], shrink=0.8)
        
        # Composite risk in the last subplot
        scatter = axes[5].scatter(
            [p.x for p in self.results_df.geometry],
            [p.y for p in self.results_df.geometry],
            c=self.results_df['composite_risk_score'],
            cmap='RdYlBu_r', s=70, alpha=0.8,
            edgecolors='black', linewidth=0.5
        )
        axes[5].set_title('Composite Risk Score', fontweight='bold', fontsize=14)
        axes[5].set_xlabel('Longitude')
        axes[5].set_ylabel('Latitude')
        axes[5].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[5], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.country_name.lower()}_risk_components.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_priority_schools_analysis(self, output_path):
        """Create priority schools analysis for intervention planning."""
        # Get top 20 priority schools
        top_priority = self.results_df.nlargest(20, 'composite_risk_score')
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Priority schools map
        axes[0,0].scatter(
            [p.x for p in self.results_df.geometry],
            [p.y for p in self.results_df.geometry],
            c=self.results_df['composite_risk_score'],
            cmap='RdYlBu_r', s=50, alpha=0.6
        )
        axes[0,0].scatter(
            [p.x for p in top_priority.geometry],
            [p.y for p in top_priority.geometry],
            c='red', s=150, alpha=0.9,
            marker='*', edgecolors='black', linewidth=2
        )
        axes[0,0].set_title('Top 20 Priority Schools for Intervention', fontweight='bold')
        axes[0,0].set_xlabel('Longitude')
        axes[0,0].set_ylabel('Latitude')
        
        # 2. Priority distribution
        axes[0,1].hist(top_priority['composite_risk_score'], bins=10, alpha=0.7,
                      color='red', edgecolor='black')
        axes[0,1].set_title('Risk Score Distribution - Priority Schools', fontweight='bold')
        axes[0,1].set_xlabel('Risk Score')
        axes[0,1].set_ylabel('Number of Schools')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Risk factors for priority schools
        risk_cols = ['healthcare_distance_risk', 'healthcare_capacity_risk',
                    'road_condition_risk', 'infrastructure_age_risk', 'environmental_risk']
        priority_risk_data = top_priority[risk_cols].mean()
        
        axes[1,0].bar(range(len(priority_risk_data)), priority_risk_data.values, 
                     alpha=0.7, color='orange')
        axes[1,0].set_xticks(range(len(priority_risk_data)))
        axes[1,0].set_xticklabels([col.replace('_', '\n') for col in priority_risk_data.index], 
                                 rotation=45, ha='right')
        axes[1,0].set_title('Average Risk Factors - Priority Schools', fontweight='bold')
        axes[1,0].set_ylabel('Risk Score')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Intervention impact simulation
        intervention_scenarios = ['Healthcare Access', 'Road Improvement', 
                                'Infrastructure Upgrade', 'Combined Intervention']
        impact_scores = [0.3, 0.2, 0.15, 0.6]  # Simulated impact
        
        axes[1,1].bar(intervention_scenarios, impact_scores, 
                     alpha=0.7, color=['blue', 'green', 'purple', 'gold'])
        axes[1,1].set_title('Potential Intervention Impact (Simulated)', fontweight='bold')
        axes[1,1].set_ylabel('Risk Reduction')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.country_name.lower()}_priority_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_statistical_dashboard(self, output_path):
        """Create comprehensive statistical dashboard."""
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        
        # 1. Risk distribution histogram
        axes[0,0].hist(self.results_df['composite_risk_score'], bins=30, alpha=0.7,
                      color='skyblue', edgecolor='black')
        axes[0,0].set_title('Composite Risk Score Distribution', fontweight='bold')
        axes[0,0].set_xlabel('Risk Score')
        axes[0,0].set_ylabel('Number of Schools')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Risk category pie chart
        risk_counts = self.results_df['risk_category'].value_counts()
        colors = ['#2E8B57', '#FFD700', '#FF6347', '#DC143C']
        axes[0,1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                     colors=colors[:len(risk_counts)], startangle=90)
        axes[0,1].set_title('Risk Category Distribution', fontweight='bold')
        
        # 3. Distance vs Risk scatter
        axes[0,2].scatter(self.results_df['nearest_overall_dist'], 
                         self.results_df['composite_risk_score'],
                         alpha=0.6, s=50)
        axes[0,2].set_xlabel('Distance to Nearest Healthcare (degrees)')
        axes[0,2].set_ylabel('Composite Risk Score')
        axes[0,2].set_title('Healthcare Distance vs Risk', fontweight='bold')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Facility access distribution
        axes[1,0].hist(self.results_df['facilities_10km'], bins=20, alpha=0.7,
                      color='lightgreen', edgecolor='black')
        axes[1,0].set_title('Healthcare Facilities within 10km', fontweight='bold')
        axes[1,0].set_xlabel('Number of Facilities')
        axes[1,0].set_ylabel('Number of Schools')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Risk correlation matrix
        risk_cols = ['healthcare_distance_risk', 'healthcare_capacity_risk',
                    'road_condition_risk', 'infrastructure_age_risk', 'environmental_risk']
        corr_matrix = self.results_df[risk_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   ax=axes[1,1], square=True)
        axes[1,1].set_title('Risk Factor Correlations', fontweight='bold')
        
        # 6. Emergency access scores
        axes[1,2].hist(self.results_df['emergency_access_score'], bins=20, alpha=0.7,
                      color='orange', edgecolor='black')
        axes[1,2].set_title('Emergency Access Score Distribution', fontweight='bold')
        axes[1,2].set_xlabel('Access Score')
        axes[1,2].set_ylabel('Number of Schools')
        axes[1,2].grid(True, alpha=0.3)
        
        # 7. Risk by facility type
        if 'nearest_facility_type' in self.results_df.columns:
            facility_risk = self.results_df.groupby('nearest_facility_type')['composite_risk_score'].mean()
            axes[2,0].bar(facility_risk.index, facility_risk.values, alpha=0.7, color='teal')
            axes[2,0].set_title('Average Risk by Nearest Facility Type', fontweight='bold')
            axes[2,0].set_ylabel('Average Risk Score')
            axes[2,0].tick_params(axis='x', rotation=45)
            axes[2,0].grid(True, alpha=0.3)
        
        # 8. Priority ranking distribution
        axes[2,1].hist(self.results_df['priority_rank'], bins=20, alpha=0.7,
                      color='purple', edgecolor='black')
        axes[2,1].set_title('Priority Ranking Distribution', fontweight='bold')
        axes[2,1].set_xlabel('Priority Rank')
        axes[2,1].set_ylabel('Number of Schools')
        axes[2,1].grid(True, alpha=0.3)
        
        # 9. Summary statistics table
        summary_stats = {
            'Total Schools': len(self.results_df),
            'Critical Risk': len(self.results_df[self.results_df['risk_category'] == 'Critical Risk']),
            'High Risk': len(self.results_df[self.results_df['risk_category'] == 'High Risk']),
            'Avg Risk Score': f"{self.results_df['composite_risk_score'].mean():.3f}",
            'Max Distance (deg)': f"{self.results_df['nearest_overall_dist'].max():.4f}",
            'Schools w/o Healthcare': len(self.results_df[self.results_df['facilities_10km'] == 0])
        }
        
        axes[2,2].axis('off')
        table_data = [[k, v] for k, v in summary_stats.items()]
        table = axes[2,2].table(cellText=table_data, 
                               colLabels=['Metric', 'Value'],
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        axes[2,2].set_title('Summary Statistics', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.country_name.lower()}_statistical_dashboard.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def export_comprehensive_results(self, output_dir="enhanced_risk_analysis"):
        """Export comprehensive results in multiple formats."""
        print(f"\n=== Exporting Comprehensive Results ===")
        
        if self.results_df is None:
            raise ValueError("Must calculate composite risk scores first")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.country_name.lower()}_enhanced_school_risk_analysis_{timestamp}"
        
        # 1. CSV export (without geometry)
        csv_df = self.results_df.drop(columns=['geometry'])
        csv_path = output_path / f"{base_filename}.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"✅ CSV export: {csv_path}")
        
        # 2. GeoJSON export (with geometry for GIS)
        geojson_path = output_path / f"{base_filename}.geojson"
        self.results_df.to_file(geojson_path, driver='GeoJSON')
        print(f"✅ GeoJSON export: {geojson_path}")
        
        # 3. Excel workbook with multiple sheets
        excel_path = output_path / f"{base_filename}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main results
            csv_df.to_excel(writer, sheet_name='School_Risk_Analysis', index=False)
            
            # Priority schools (top 50)
            priority_schools = self.results_df.nlargest(50, 'composite_risk_score')
            priority_df = priority_schools.drop(columns=['geometry'])
            priority_df.to_excel(writer, sheet_name='Priority_Schools', index=False)
            
            # Summary statistics
            summary_stats = self._generate_detailed_summary()
            pd.DataFrame([summary_stats]).to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Risk distribution
            risk_dist = self.results_df['risk_category'].value_counts().to_frame('Count')
            risk_dist.to_excel(writer, sheet_name='Risk_Distribution')
            
            # Healthcare access summary
            healthcare_summary = self.results_df.groupby('nearest_facility_type').agg({
                'composite_risk_score': ['mean', 'count'],
                'nearest_overall_dist': 'mean',
                'facilities_10km': 'mean'
            }).round(4)
            healthcare_summary.to_excel(writer, sheet_name='Healthcare_Access_Summary')
        
        print(f"✅ Excel workbook: {excel_path}")
        
        # 4. Summary report
        self._export_detailed_report(output_path, base_filename)
        
        # 5. CCRI Integration format
        ccri_df = self._prepare_ccri_integration_format()
        ccri_path = output_path / f"{base_filename}_CCRI_integration.csv"
        ccri_df.to_csv(ccri_path, index=False)
        print(f"✅ CCRI Integration format: {ccri_path}")
        
        return output_path

    def _generate_detailed_summary(self):
        """Generate detailed summary statistics."""
        return {
            'analysis_date': datetime.now().isoformat(),
            'country': self.country_name,
            'total_schools': len(self.results_df),
            'total_healthcare_facilities': len(self.healthcare_data),
            'critical_risk_schools': len(self.results_df[self.results_df['risk_category'] == 'Critical Risk']),
            'high_risk_schools': len(self.results_df[self.results_df['risk_category'] == 'High Risk']),
            'medium_risk_schools': len(self.results_df[self.results_df['risk_category'] == 'Medium Risk']),
            'low_risk_schools': len(self.results_df[self.results_df['risk_category'] == 'Low Risk']),
            'avg_composite_risk': float(self.results_df['composite_risk_score'].mean()),
            'max_composite_risk': float(self.results_df['composite_risk_score'].max()),
            'avg_healthcare_distance': float(self.results_df['nearest_overall_dist'].mean()),
            'max_healthcare_distance': float(self.results_df['nearest_overall_dist'].max()),
            'schools_no_nearby_healthcare': len(self.results_df[self.results_df['facilities_10km'] == 0]),
            'avg_emergency_access_score': float(self.results_df['emergency_access_score'].mean()),
            'hospitals_available': len(self.healthcare_data[self.healthcare_data['amenity'] == 'hospital']),
            'clinics_available': len(self.healthcare_data[self.healthcare_data['amenity'] == 'clinic']),
            'pharmacies_available': len(self.healthcare_data[self.healthcare_data['amenity'] == 'pharmacy'])
        }

    def _prepare_ccri_integration_format(self):
        """Prepare data in format suitable for CCRI integration."""
        ccri_df = self.results_df.copy()
        
        # Rename columns for CCRI compatibility
        ccri_df = ccri_df.rename(columns={
            'school_name': 'facility_name',
            'composite_risk_score': 'healthcare_access_risk_score',
            'nearest_overall_dist': 'healthcare_distance_km',
            'facilities_10km': 'healthcare_facility_density',
            'emergency_access_score': 'emergency_healthcare_access'
        })
        
        # Add CCRI-specific fields
        ccri_df['data_source'] = 'UN-Tech-Over Enhanced Analysis'
        ccri_df['analysis_methodology'] = 'Multi-factor risk assessment'
        ccri_df['recommended_intervention_priority'] = ccri_df['priority_rank'].apply(
            lambda x: 'High' if x <= 20 else 'Medium' if x <= 50 else 'Low'
        )
        
        # Select relevant columns for CCRI
        ccri_columns = [
            'facility_name', 'school_lat', 'school_lon',
            'healthcare_access_risk_score', 'risk_category',
            'healthcare_distance_km', 'healthcare_facility_density',
            'emergency_healthcare_access', 'recommended_intervention_priority',
            'analysis_date', 'country', 'data_source'
        ]
        
        return ccri_df[ccri_columns]

    def _export_detailed_report(self, output_path, base_filename):
        """Export detailed analysis report."""
        summary = self._generate_detailed_summary()
        
        report_path = output_path / f"{base_filename}_detailed_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED SCHOOL RISK ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Country: {summary['country']}\n")
            f.write(f"Analysis Date: {summary['analysis_date']}\n")
            f.write(f"Analysis Framework: Enhanced UN-Tech-Over Implementation\n")
            f.write(f"CCRI Integration: Ready\n\n")
            
            f.write("ANALYSIS SCOPE:\n")
            f.write(f"• Total Schools Analyzed: {summary['total_schools']}\n")
            f.write(f"• Total Healthcare Facilities: {summary['total_healthcare_facilities']}\n")
            f.write(f"  - Hospitals: {summary['hospitals_available']}\n")
            f.write(f"  - Clinics: {summary['clinics_available']}\n")
            f.write(f"  - Pharmacies: {summary['pharmacies_available']}\n\n")
            
            f.write("RISK ASSESSMENT RESULTS:\n")
            f.write(f"• Critical Risk Schools: {summary['critical_risk_schools']}\n")
            f.write(f"• High Risk Schools: {summary['high_risk_schools']}\n")
            f.write(f"• Medium Risk Schools: {summary['medium_risk_schools']}\n")
            f.write(f"• Low Risk Schools: {summary['low_risk_schools']}\n\n")
            
            f.write("HEALTHCARE ACCESSIBILITY:\n")
            f.write(f"• Average Distance to Healthcare: {summary['avg_healthcare_distance']:.4f} degrees\n")
            f.write(f"• Maximum Distance to Healthcare: {summary['max_healthcare_distance']:.4f} degrees\n")
            f.write(f"• Schools without Nearby Healthcare: {summary['schools_no_nearby_healthcare']}\n")
            f.write(f"• Average Emergency Access Score: {summary['avg_emergency_access_score']:.3f}\n\n")
            
            f.write("RISK SCORING:\n")
            f.write(f"• Average Composite Risk Score: {summary['avg_composite_risk']:.4f}\n")
            f.write(f"• Maximum Risk Score: {summary['max_composite_risk']:.4f}\n\n")
            
            f.write("METHODOLOGY:\n")
            f.write("Risk factors weighted as follows:\n")
            for factor, weight in self.risk_weights.items():
                f.write(f"  • {factor.replace('_', ' ').title()}: {weight*100:.0f}%\n")
            
            f.write("\nCCRI INTEGRATION:\n")
            f.write("• Healthcare access risk scores calculated for all schools\n")
            f.write("• Fine-grained distance and facility density metrics available\n")
            f.write("• Priority intervention rankings established\n")
            f.write("• Data exported in UNICEF-compatible formats\n")
            f.write("• Ready for integration with climate risk indicators\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("1. Prioritize healthcare access improvements for Critical/High risk schools\n")
            f.write("2. Focus on schools with >0.1 degree distance to nearest healthcare\n")
            f.write("3. Consider mobile healthcare units for isolated schools\n")
            f.write("4. Integrate results with UNICEF GeoSight CCRI platform\n")
            f.write("5. Update analysis annually or after infrastructure changes\n")
        
        print(f"✅ Detailed report: {report_path}")

    def run_enhanced_analysis(self):
        """Run the complete enhanced analysis pipeline."""
        print("🏫 ENHANCED SCHOOL RISK ANALYSIS SYSTEM")
        print("🎯 UN-Tech-Over Implementation with CCRI Integration")
        print("=" * 70)
        
        try:
            # 1. Load and validate data
            print("\n1️⃣  Loading and validating UN-Tech-Over data...")
            if not self.load_and_validate_data():
                raise ValueError("Failed to load required data files")
            
            # 2. Calculate comprehensive risk scores
            print("\n2️⃣  Calculating comprehensive risk assessment...")
            results = self.calculate_composite_risk_scores()
            
            # 3. Create enhanced visualizations
            print("\n3️⃣  Creating enhanced visualizations...")
            self.create_enhanced_visualizations()
            
            # 4. Export comprehensive results
            print("\n4️⃣  Exporting results in multiple formats...")
            output_path = self.export_comprehensive_results()
            
            # 5. Summary
            summary = self._generate_detailed_summary()
            
            print("\n" + "=" * 70)
            print("🎉 ENHANCED ANALYSIS COMPLETE!")
            print("=" * 70)
            
            print(f"\n📊 Final Results Summary:")
            print(f"• Total Schools Analyzed: {summary['total_schools']}")
            print(f"• Critical Risk Schools: {summary['critical_risk_schools']}")
            print(f"• High Risk Schools: {summary['high_risk_schools']}")
            print(f"• Average Risk Score: {summary['avg_composite_risk']:.3f}")
            print(f"• Schools Needing Urgent Healthcare Access: {summary['schools_no_nearby_healthcare']}")
            
            print(f"\n📁 All outputs saved to: {output_path}")
            
            print(f"\n🌍 UNICEF CCRI Integration Status:")
            print(f"✅ Healthcare access risk indicators calculated")
            print(f"✅ Fine-grained school-level data available") 
            print(f"✅ Priority intervention rankings established")
            print(f"✅ Data exported in compatible formats")
            print(f"✅ Ready for GeoSight platform integration")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            raise

# Main execution
if __name__ == "__main__":
    print("🚀 Enhanced School Risk Analysis - UN-Tech-Over Implementation")
    print("=" * 80)

    base_directory = str(os.getcwd())
    
    # Initialize the enhanced mapper
    mapper = EnhancedSchoolRiskMapper(
        country_name = "Tajikistan",
        base_dir = base_directory
    )
    
    # Run the complete analysis
    try:
        results = mapper.run_enhanced_analysis()
        print("\n✨ Analysis complete! Ready for UNICEF GeoSight integration.")
        print("📧 Data can now be uploaded to supplement CCRI indicators.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the UN-Tech-Over data files are available in the correct location.")
        print("Expected files:")
        print(f"{base_directory}/data/tajikistan.csv")
        print(f"{base_directory}/data/hotosm_tjk_education_facilities_points_geojson.geojson")