#!/usr/bin/env python3
"""
Comprehensive School Risk Mapping System
=========================================

A generalized framework for analyzing risks to schoolchildren that supplements and extends
UNICEF GeoSight's "Children's Climate and Environmental Risk Index (CCRI)" with fine-grained,
location-specific data analysis.

Key Features:
- Works with any country's data input
- Healthcare accessibility analysis
- Road condition assessment
- Infrastructure age evaluation
- Composite risk scoring with map overlays
- Comprehensive labeling and visualization
- Base map integration

Author: Enhanced from UN-Tech-Over framework
Version: 2.0
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon
import warnings
import json
from datetime import datetime
import contextily as ctx
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class SchoolRiskMapper:
    """
    Comprehensive risk mapping system for schoolchildren that works with any country data.
    
    This system analyzes multiple risk factors affecting children's access to healthcare
    and educational safety, providing fine-grained analysis to supplement UNICEF's CCRI.
    """
    
    def __init__(self, country_name="Unknown", data_dir=None, config=None):
        """
        Initialize the risk mapper with country-specific or general configuration.
        
        Parameters:
        -----------
        country_name : str
            Name of the country being analyzed
        data_dir : str or Path
            Directory containing the input data files
        config : dict
            Configuration dictionary for risk weights and parameters
        """
        self.country_name = country_name
        self.data_dir = Path(data_dir) if data_dir else Path("./data")
        
        # Default risk weights (can be customized per country/analysis)
        default_risk_weights = {
            'healthcare_distance': 0.30,      # Distance to nearest healthcare facility
            'healthcare_capacity': 0.20,      # Healthcare facility density/capacity
            'road_conditions': 0.25,          # Road infrastructure quality
            'infrastructure_age': 0.15,       # Age and condition of built environment
            'environmental_hazards': 0.10,    # Environmental risk factors
        }
        
        # Merge provided risk weights with defaults
        if config and 'risk_weights' in config:
            self.risk_weights = {**default_risk_weights, **config['risk_weights']}
        else:
            self.risk_weights = default_risk_weights
        
        # Default analysis parameters
        default_analysis_params = {
            'distance_thresholds': {
                'emergency': 0.05,    # ~5km in degrees
                'routine': 0.1,       # ~10km in degrees
                'accessible': 0.2     # ~20km in degrees
            },
            'facility_priorities': {
                'hospital': 1.0,
                'emergency_hospital': 1.2,
                'clinic': 0.8,
                'health_center': 0.7,
                'doctors': 0.6,
                'pharmacy': 0.4,
                'dentist': 0.3
            },
            'infrastructure_age_indicators': {
                'modern_keywords': ['гимназия', 'лицей', 'центр', 'modern', 'new', 'academy'],
                'traditional_keywords': ['школа', 'мактаб', 'school', 'basic'],
                'building_quality_indicators': ['brick', 'concrete', 'permanent', 'temporary']
            }
        }
        
        # Merge provided analysis params with defaults
        if config and 'analysis_params' in config:
            self.analysis_params = default_analysis_params.copy()
            for key, value in config['analysis_params'].items():
                if isinstance(value, dict) and key in self.analysis_params:
                    # Merge nested dictionaries
                    self.analysis_params[key] = {**self.analysis_params[key], **value}
                else:
                    # Replace completely for non-dict values
                    self.analysis_params[key] = value
        else:
            self.analysis_params = default_analysis_params
        
        # Initialize data containers
        self.healthcare_facilities = None
        self.education_facilities = None
        self.roads_data = None
        self.population_data = None
        self.results_df = None
        
        # Scalers for normalization
        self.scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        
        print(f"🌍 Initialized School Risk Mapper for {self.country_name}")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"⚖️  Risk weights: {self.risk_weights}")

    def load_data(self, poi_file=None, education_file=None, infrastructure_file=None, roads_file=None, auto_detect=True):
        """
        Load healthcare, education, and supporting data files.
        
        Parameters:
        -----------
        poi_file : str
            Path to healthcare facilities file (CSV, GeoJSON, or Shapefile)
        education_file : str  
            Path to education facilities file (GeoJSON, Shapefile, or CSV)
        roads_file : str
            Path to roads/infrastructure file (optional)
        auto_detect : bool
            Whether to automatically detect and load files from data directory
        """
        print("=== Loading Data Files ===")
        
        if auto_detect:
            # Auto-detect common file patterns
            poi_file = self._auto_detect_file(['healthcare', 'health', 'hospital', 'clinic'], 
                                                   ['csv', 'geojson', 'shp'])
            education_file = self._auto_detect_file(['education', 'school', 'university', 'college'],
                                                  ['geojson', 'shp', 'csv'])
            roads_file = self._auto_detect_file(['road', 'transport', 'infrastructure'],
                                              ['geojson', 'shp', 'csv'])
        
        # Load healthcare facilities
        if poi_file and Path(poi_file).exists():
            self.healthcare_facilities = self._load_geospatial_file(poi_file)
            if self.healthcare_facilities is not None:
                self.healthcare_facilities = self._clean_healthcare_data(self.healthcare_facilities)
                print(f"✅ Loaded {len(self.healthcare_facilities)} healthcare facilities")
        
        # Load education facilities  
        if education_file and Path(education_file).exists():
            self.education_facilities = self._load_geospatial_file(education_file)
            if self.education_facilities is not None:
                self.education_facilities = self._clean_education_data(self.education_facilities)
                print(f"✅ Loaded {len(self.education_facilities)} education facilities")
        
        # Load roads/infrastructure (optional)
        if roads_file and Path(roads_file).exists():
            self.roads_data = self._load_geospatial_file(roads_file)
            if self.roads_data is not None:
                print(f"✅ Loaded {len(self.roads_data)} road/infrastructure features")
        
        # Validate essential data
        if self.healthcare_facilities is None or self.education_facilities is None:
            raise ValueError("❌ Failed to load essential data files. Please check file paths and formats.")
        
        return True

    def _auto_detect_file(self, keywords, extensions):
        """Auto-detect files based on keywords and extensions"""
        for keyword in keywords:
            for ext in extensions:
                pattern = f"*{keyword}*{ext}"
                matches = list(self.data_dir.glob(pattern))
                if matches:
                    return matches[0]
        return None

    def _load_geospatial_file(self, file_path):
        """Load geospatial file (GeoJSON, Shapefile, or CSV with coordinates)"""
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)

                # 1) If you have explicit X/Y columns, use them:
                if 'X' in df.columns and 'Y' in df.columns:
                    lon_col, lat_col = 'X', 'Y'
                else:
                    # otherwise fall back to your detector
                    coords = self._detect_coordinate_columns(df)
                    if coords:
                        lon_col, lat_col = coords
                    else:
                        # no coords at all—just return raw df
                        return df

                # 2) Force X/Y to numeric, drop any rows that fail
                df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
                df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
                df = df.dropna(subset=[lon_col, lat_col])

                # 3) Build your GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                    crs="EPSG:4326"
                )
                return gdf
            else:
                return gpd.read_file(file_path)
        except Exception as e:
            print(f"⚠️  Warning: Could not load {file_path}: {e}")
            return None

    def _detect_coordinate_columns(self, df):
        """Detect longitude/latitude columns in a DataFrame"""
        lon_candidates = ['lon', 'lng', 'longitude', 'x', 'X', 'long']
        lat_candidates = ['lat', 'latitude', 'y', 'Y']
        
        lon_col = None
        lat_col = None
        
        for col in df.columns:
            if any(candidate.lower() in col.lower() for candidate in lon_candidates):
                lon_col = col
            elif any(candidate.lower() in col.lower() for candidate in lat_candidates):
                lat_col = col
        
        return (lon_col, lat_col) if lon_col and lat_col else None

    def _clean_healthcare_data(self, df):
        """Clean and standardize healthcare facilities data"""
        # Remove rows without coordinates
        if hasattr(df, 'geometry'):
            df = df.dropna(subset=['geometry'])
        else:
            coord_cols = self._detect_coordinate_columns(df)
            if coord_cols:
                df = df.dropna(subset=list(coord_cols))
        
        # Standardize amenity/facility type column
        if 'amenity' not in df.columns:
            # Try to find facility type column
            type_candidates = ['type', 'facility_type', 'healthcare', 'category']
            for col in df.columns:
                if any(candidate in col.lower() for candidate in type_candidates):
                    df['amenity'] = df[col]
                    break
        
        # Filter healthcare-related facilities
        if 'amenity' in df.columns:
            healthcare_types = ['hospital', 'clinic', 'doctors', 'pharmacy', 'dentist', 
                              'health_center', 'medical_center', 'emergency']
            df = df[df['amenity'].str.contains('|'.join(healthcare_types), case=False, na=False)]
        
        return df

    def _clean_education_data(self, df):
        """Clean and standardize education facilities data"""
        # Remove rows without coordinates
        if hasattr(df, 'geometry'):
            df = df.dropna(subset=['geometry'])
        
        # Filter for schools only (exclude kindergartens and universities for primary analysis)
        if 'amenity' in df.columns:
            school_types = ['school']
            df = df[df['amenity'].isin(school_types)]
        
        return df

    def calculate_healthcare_distance_risk(self):
        """
        Calculate comprehensive healthcare accessibility risk scores.
        
        Returns:
        --------
        dict: Dictionary containing various distance metrics and risk scores
        """
        print("\n=== Calculating Healthcare Distance Risk ===")
        
        if self.healthcare_facilities is None or self.education_facilities is None:
            raise ValueError("Healthcare and education data must be loaded first")
        
        # Get coordinates
        schools_coords = np.array([
            [point.x, point.y] for point in self.education_facilities.geometry
        ])
        
        healthcare_coords = np.array([
            [point.x, point.y] for point in self.healthcare_facilities.geometry
        ])
        
        results = {
            'school_id': [],
            'school_name': [],
            'school_coordinates': [],
            'nearest_hospital_dist': [],
            'nearest_clinic_dist': [],
            'nearest_pharmacy_dist': [],
            'nearest_overall_dist': [],
            'nearest_facility_type': [],
            'nearest_facility_name': [],
            'facilities_within_5km': [],
            'facilities_within_10km': [],
            'facilities_within_20km': [],
            'emergency_access_score': [],
            'routine_access_score': [],
            'weighted_access_score': []
        }
        
        for idx, (school_idx, school) in enumerate(self.education_facilities.iterrows()):
            school_coord = schools_coords[idx]
            
            # Basic information
            results['school_id'].append(school_idx)
            results['school_name'].append(school.get('name', f'School_{school_idx}'))
            results['school_coordinates'].append(f"{school_coord[1]:.6f}, {school_coord[0]:.6f}")
            
            # Calculate distances to different facility types
            facility_distances = {}
            facility_counts = {'5km': 0, '10km': 0, '20km': 0}
            
            distances_all = cdist([school_coord], healthcare_coords, metric='euclidean')[0]
            
            # Process each facility type
            for facility_type, priority in self.analysis_params['facility_priorities'].items():
                type_mask = self.healthcare_facilities['amenity'].str.contains(
                    facility_type, case=False, na=False
                )
                
                if type_mask.any():
                    type_distances = distances_all[type_mask]
                    facility_distances[facility_type] = np.min(type_distances)
                    
                    # Count facilities within different radii
                    facility_counts['5km'] += np.sum(type_distances < self.analysis_params['distance_thresholds']['emergency'])
                    facility_counts['10km'] += np.sum(type_distances < self.analysis_params['distance_thresholds']['routine'])
                    facility_counts['20km'] += np.sum(type_distances < self.analysis_params['distance_thresholds']['accessible'])
                else:
                    facility_distances[facility_type] = float('inf')
            
            # Store specific distances
            results['nearest_hospital_dist'].append(
                facility_distances.get('hospital', float('inf'))
            )
            results['nearest_clinic_dist'].append(
                facility_distances.get('clinic', float('inf'))
            )
            results['nearest_pharmacy_dist'].append(
                facility_distances.get('pharmacy', float('inf'))
            )
            
            # Find overall nearest facility with priority weighting
            best_weighted_dist = float('inf')
            nearest_type = 'None'
            nearest_name = 'Unknown'
            nearest_overall = float('inf')
            
            for facility_type, distance in facility_distances.items():
                if distance < float('inf'):
                    priority = self.analysis_params['facility_priorities'].get(facility_type, 0.5)
                    weighted_dist = distance / priority
                    
                    if weighted_dist < best_weighted_dist:
                        best_weighted_dist = weighted_dist
                        nearest_type = facility_type
                        nearest_overall = distance
                        
                        # Find specific facility name
                        type_mask = self.healthcare_facilities['amenity'].str.contains(
                            facility_type, case=False, na=False
                        )
                        if type_mask.any():
                            type_coords = healthcare_coords[type_mask]
                            type_distances = cdist([school_coord], type_coords, metric='euclidean')[0]
                            nearest_idx = np.argmin(type_distances)
                            type_facilities = self.healthcare_facilities[type_mask]
                            nearest_name = type_facilities.iloc[nearest_idx].get('name', 'Unnamed Facility')
            
            results['nearest_overall_dist'].append(nearest_overall if nearest_overall != float('inf') else None)
            results['nearest_facility_type'].append(nearest_type)
            results['nearest_facility_name'].append(nearest_name)
            
            # Store facility counts
            results['facilities_within_5km'].append(facility_counts['5km'])
            results['facilities_within_10km'].append(facility_counts['10km'])
            results['facilities_within_20km'].append(facility_counts['20km'])
            
            # Calculate access scores
            emergency_score = self._calculate_access_score(
                nearest_overall, 
                self.analysis_params['distance_thresholds']['emergency']
            )
            routine_score = self._calculate_access_score(
                nearest_overall,
                self.analysis_params['distance_thresholds']['routine']
            )
            
            # Weighted access score considering facility type priorities
            weighted_score = best_weighted_dist if best_weighted_dist != float('inf') else 1.0
            
            results['emergency_access_score'].append(emergency_score)
            results['routine_access_score'].append(routine_score)
            results['weighted_access_score'].append(min(weighted_score, 1.0))
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Normalize risk scores (higher score = higher risk)
        if len(results_df) > 0:
            # Emergency access risk (inverted score)
            results_df['emergency_risk'] = 1 - results_df['emergency_access_score']
            
            # Routine access risk (inverted score)  
            results_df['routine_risk'] = 1 - results_df['routine_access_score']
            
            # Weighted distance risk (normalized)
            valid_weighted = results_df['weighted_access_score'][results_df['weighted_access_score'] < 1.0]
            if len(valid_weighted) > 0:
                results_df['healthcare_distance_risk'] = self.scaler.fit_transform(
                    results_df[['weighted_access_score']]
                ).flatten()
            else:
                results_df['healthcare_distance_risk'] = 0.5
        
        print(f"✅ Healthcare distance analysis complete for {len(results_df)} schools")
        print(f"📊 Average distance to nearest facility: {results_df['nearest_overall_dist'].mean():.4f} degrees")
        
        return results_df

    def _calculate_access_score(self, distance, threshold):
        """Calculate access score based on distance and threshold"""
        if distance is None or distance == float('inf'):
            return 0.0
        return max(0.0, 1.0 - (distance / threshold))

    def calculate_road_condition_risk(self):
        """
        Estimate road condition risk using multiple indicators.
        
        Returns:
        --------
        np.array: Road condition risk scores (0=good, 1=poor)
        """
        print("\n=== Calculating Road Condition Risk ===")
        
        schools_coords = np.array([
            [point.x, point.y] for point in self.education_facilities.geometry
        ])
        
        road_risk_scores = []
        
        # Define major urban centers (can be expanded based on country)
        major_centers = self._identify_major_centers()
        
        for school_coord in schools_coords:
            risk_factors = []
            
            # 1. Distance from major urban centers (proxy for road development)
            if major_centers:
                center_distances = []
                for center_lon, center_lat, center_name, importance in major_centers:
                    dist = np.sqrt((school_coord[0] - center_lon)**2 + (school_coord[1] - center_lat)**2)
                    weighted_dist = dist / importance  # Weight by city importance
                    center_distances.append(weighted_dist)
                
                min_center_distance = min(center_distances)
                urban_access_risk = min(min_center_distance * 2, 1.0)  # Normalize
                risk_factors.append(urban_access_risk)
            
            # 2. Healthcare facility density (proxy for infrastructure development)
            if self.healthcare_facilities is not None:
                healthcare_coords = np.array([
                    [point.x, point.y] for point in self.healthcare_facilities.geometry
                ])
                
                distances = cdist([school_coord], healthcare_coords, metric='euclidean')[0]
                nearby_facilities = np.sum(distances < 0.1)  # Within ~10km
                facility_density_risk = 1.0 / (1.0 + nearby_facilities * 0.1)
                risk_factors.append(facility_density_risk)
            
            # 3. Terrain/elevation proxy (based on latitude variation)
            # Schools in mountainous areas typically have worse road access
            baseline_lat = np.mean([coord[1] for coord in schools_coords])
            elevation_proxy = abs(school_coord[1] - baseline_lat) * 3
            terrain_risk = min(elevation_proxy, 1.0)
            risk_factors.append(terrain_risk)
            
            # 4. Road network density (if roads data available)
            if self.roads_data is not None:
                road_density_risk = self._calculate_road_network_risk(school_coord)
                risk_factors.append(road_density_risk)
            
            # Combine all risk factors
            overall_road_risk = np.mean(risk_factors) if risk_factors else 0.5
            road_risk_scores.append(overall_road_risk)
        
        # Normalize scores
        road_risk_normalized = self.scaler.fit_transform(
            np.array(road_risk_scores).reshape(-1, 1)
        ).flatten()
        
        print(f"✅ Road condition risk calculated")
        print(f"📊 Average road risk score: {np.mean(road_risk_normalized):.3f}")
        
        return road_risk_normalized

    def _identify_major_centers(self):
        """Identify major urban centers for road analysis"""
        # This can be expanded with actual city data or extracted from healthcare/education density
        if self.healthcare_facilities is None:
            return []
        
        # Use healthcare facility clustering to identify major centers
        healthcare_coords = np.array([
            [point.x, point.y] for point in self.healthcare_facilities.geometry
        ])
        
        if len(healthcare_coords) < 3:
            return []
        
        # Use KMeans to find facility clusters
        n_clusters = min(5, len(healthcare_coords) // 10)  # Adaptive number of clusters
        if n_clusters < 1:
            n_clusters = 1
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(healthcare_coords)
        
        centers = []
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > 0:
                center_coord = kmeans.cluster_centers_[i]
                importance = min(cluster_size / len(healthcare_coords) * 2, 1.0)
                centers.append((center_coord[0], center_coord[1], f"Center_{i}", importance))
        
        return centers

    def _calculate_road_network_risk(self, school_coord):
        """Calculate road network density risk around a school"""
        # This would analyze actual road network data if available
        # For now, return a placeholder based on facility access
        return 0.5

    def calculate_infrastructure_age_risk(self):
        """
        Estimate infrastructure age and quality risk using multiple indicators.
        
        Returns:
        --------
        np.array: Infrastructure age risk scores (0=modern, 1=old/poor)
        """
        print("\n=== Calculating Infrastructure Age Risk ===")
        
        infrastructure_risk_scores = []
        
        for idx, school in self.education_facilities.iterrows():
            risk_factors = []
            
            # 1. School naming pattern analysis
            school_name = str(school.get('name', ''))
            if school_name and school_name != 'nan':
                name_risk = self._analyze_naming_pattern(school_name)
                risk_factors.append(name_risk)
            
            # 2. Building type/quality indicators
            building_risk = self._analyze_building_quality(school)
            risk_factors.append(building_risk)
            
            # 3. Operator type analysis
            operator_risk = self._analyze_operator_type(school)
            risk_factors.append(operator_risk)
            
            # 4. Geographic infrastructure proxy
            school_coord = [school.geometry.x, school.geometry.y]
            location_risk = self._calculate_location_infrastructure_risk(school_coord)
            risk_factors.append(location_risk)
            
            # 5. Facility density proxy (more facilities = better infrastructure)
            if self.healthcare_facilities is not None:
                density_risk = self._calculate_infrastructure_density_risk(school_coord)
                risk_factors.append(density_risk)
            
            # Combine all risk factors
            overall_infrastructure_risk = np.mean(risk_factors)
            infrastructure_risk_scores.append(overall_infrastructure_risk)
        
        print(f"✅ Infrastructure age risk calculated")
        print(f"📊 Average infrastructure age risk: {np.mean(infrastructure_risk_scores):.3f}")
        
        return np.array(infrastructure_risk_scores)

    def _analyze_naming_pattern(self, school_name):
        """Analyze school naming patterns for age/quality indicators"""
        name_lower = school_name.lower()
        modern_indicators = self.analysis_params['infrastructure_age_indicators']['modern_keywords']
        traditional_indicators = self.analysis_params['infrastructure_age_indicators']['traditional_keywords']
        
        modern_score = sum(1 for indicator in modern_indicators if indicator.lower() in name_lower)
        traditional_score = sum(1 for indicator in traditional_indicators if indicator.lower() in name_lower)
        
        if modern_score > 0:
            return 0.2  # Lower risk for modern-named schools
        elif traditional_score > 0:
            return 0.7  # Higher risk for traditionally-named schools
        else:
            return 0.5  # Medium risk for unclear naming

    def _analyze_building_quality(self, school):
        """Analyze building quality indicators"""
        building_info = str(school.get('building', ''))
        
        if 'yes' in building_info.lower() or 'permanent' in building_info.lower():
            return 0.3  # Lower risk for permanent buildings
        elif 'temporary' in building_info.lower():
            return 0.8  # Higher risk for temporary buildings
        else:
            return 0.6  # Medium risk for unspecified

    def _analyze_operator_type(self, school):
        """Analyze operator type for infrastructure quality proxy"""
        operator_type = str(school.get('operator:type', ''))
        operator = str(school.get('operator', ''))
        
        operator_info = (operator_type + ' ' + operator).lower()
        
        if 'private' in operator_info:
            return 0.2  # Lower risk for private schools (often better maintained)
        elif 'public' in operator_info or 'government' in operator_info:
            return 0.6  # Medium risk for public schools
        else:
            return 0.5  # Medium risk for unknown

    def _calculate_location_infrastructure_risk(self, school_coord):
        """Calculate infrastructure risk based on location"""
        # Use distance from identified major centers
        major_centers = self._identify_major_centers()
        
        if not major_centers:
            return 0.5
        
        min_distance = float('inf')
        for center_lon, center_lat, _, importance in major_centers:
            dist = np.sqrt((school_coord[0] - center_lon)**2 + (school_coord[1] - center_lat)**2)
            weighted_dist = dist / importance
            min_distance = min(min_distance, weighted_dist)
        
        return min(min_distance * 2, 1.0)  # Normalize to 0-1

    def _calculate_infrastructure_density_risk(self, school_coord):
        """Calculate infrastructure risk based on facility density"""
        healthcare_coords = np.array([
            [point.x, point.y] for point in self.healthcare_facilities.geometry
        ])
        
        distances = cdist([school_coord], healthcare_coords, metric='euclidean')[0]
        nearby_facilities = np.sum(distances < 0.1)  # Within ~10km
        
        # More facilities = better infrastructure
        return 1.0 / (1.0 + nearby_facilities * 0.2)

    def calculate_environmental_risk(self):
        """
        Calculate environmental and climate-related risk factors.
        
        Returns:
        --------
        np.array: Environmental risk scores
        """
        print("\n=== Calculating Environmental Risk Factors ===")
        
        environmental_risk_scores = []
        
        schools_coords = np.array([
            [point.x, point.y] for point in self.education_facilities.geometry
        ])
        
        for school_coord in schools_coords:
            risk_factors = []
            
            # 1. Elevation-based risk (proxy for landslide/flood risk)
            elevation_risk = self._calculate_elevation_risk(school_coord, schools_coords)
            risk_factors.append(elevation_risk)
            
            # 2. Isolation risk (distance from other schools/facilities)
            isolation_risk = self._calculate_isolation_risk(school_coord, schools_coords)
            risk_factors.append(isolation_risk)
            
            # 3. Climate exposure (based on geographic position)
            climate_risk = self._calculate_climate_exposure_risk(school_coord)
            risk_factors.append(climate_risk)
            
            # Combine risk factors
            overall_environmental_risk = np.mean(risk_factors)
            environmental_risk_scores.append(overall_environmental_risk)
        
        print(f"✅ Environmental risk calculated")
        print(f"📊 Average environmental risk: {np.mean(environmental_risk_scores):.3f}")
        
        return np.array(environmental_risk_scores)

    def _calculate_elevation_risk(self, school_coord, all_coords):
        """Calculate elevation-based risk using coordinate variation"""
        # Use latitude variation as proxy for elevation changes
        lat_std = np.std([coord[1] for coord in all_coords])
        school_lat_deviation = abs(school_coord[1] - np.mean([coord[1] for coord in all_coords]))
        
        if lat_std > 0:
            return min(school_lat_deviation / lat_std, 1.0)
        return 0.5

    def _calculate_isolation_risk(self, school_coord, all_coords):
        """Calculate isolation risk based on distance to other schools"""
        distances = cdist([school_coord], all_coords, metric='euclidean')[0]
        distances = distances[distances > 0]  # Remove self-distance
        
        if len(distances) > 0:
            nearest_school_distance = np.min(distances)
            return min(nearest_school_distance * 5, 1.0)  # Normalize
        return 1.0

    def _calculate_climate_exposure_risk(self, school_coord):
        """Calculate climate exposure risk based on geographic position"""
        # Simplified climate risk based on latitude (can be enhanced with actual climate data)
        lat = abs(school_coord[1])
        
        # Higher risk for extreme latitudes or specific climate zones
        if lat > 40 or lat < 30:  # Simplified example
            return 0.7
        return 0.3

    def calculate_composite_risk_score(self):
        """
        Calculate comprehensive composite risk scores combining all factors.
        
        Returns:
        --------
        gpd.GeoDataFrame: Complete results with all risk scores and metadata
        """
        print("\n=== Calculating Composite Risk Scores ===")
        
        # Calculate all individual risk components
        healthcare_results = self.calculate_healthcare_distance_risk()
        road_risk = self.calculate_road_condition_risk()
        infrastructure_risk = self.calculate_infrastructure_age_risk()
        environmental_risk = self.calculate_environmental_risk()
        
        # Create comprehensive results GeoDataFrame
        results_gdf = self.education_facilities.copy()
        
        # Add all risk scores
        results_gdf['healthcare_distance_risk'] = healthcare_results['healthcare_distance_risk']
        results_gdf['healthcare_capacity_risk'] = 1 - (healthcare_results['facilities_within_10km'] / 
                                                     (healthcare_results['facilities_within_10km'].max() + 1))
        results_gdf['road_condition_risk'] = road_risk
        results_gdf['infrastructure_age_risk'] = infrastructure_risk
        results_gdf['environmental_risk'] = environmental_risk
        
        # Add detailed healthcare metrics
        for col in ['nearest_overall_dist', 'nearest_facility_type', 'nearest_facility_name',
                   'facilities_within_5km', 'facilities_within_10km', 'emergency_access_score']:
            if col in healthcare_results.columns:
                results_gdf[col] = healthcare_results[col]
        
        # Calculate weighted composite risk score
        composite_risk = (
            results_gdf['healthcare_distance_risk'] * self.risk_weights['healthcare_distance'] +
            results_gdf['healthcare_capacity_risk'] * self.risk_weights['healthcare_capacity'] +
            results_gdf['road_condition_risk'] * self.risk_weights['road_conditions'] +
            results_gdf['infrastructure_age_risk'] * self.risk_weights['infrastructure_age'] +
            results_gdf['environmental_risk'] * self.risk_weights['environmental_hazards']
        )
        
        results_gdf['composite_risk_score'] = composite_risk
        
        # Categorize risk levels
        risk_categories = pd.cut(
            composite_risk,
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low Risk', 'Medium-Low Risk', 'Medium-High Risk', 'High Risk']
        )
        results_gdf['risk_category'] = risk_categories
        
        # Add priority ranking
        results_gdf['priority_rank'] = results_gdf['composite_risk_score'].rank(method='dense', ascending=False)
        
        # Add CCRI integration fields
        results_gdf['ccri_supplement_ready'] = True
        results_gdf['analysis_date'] = datetime.now().isoformat()
        results_gdf['country'] = self.country_name
        
        self.results_df = results_gdf
        
        print(f"✅ Composite risk analysis complete")
        print(f"📊 Risk distribution:")
        print(results_gdf['risk_category'].value_counts())
        
        return results_gdf

    def create_comprehensive_visualizations(self, output_dir="risk_analysis_outputs", include_interactive=True):
        """
        Create comprehensive visualizations including static and interactive maps.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualization outputs
        include_interactive : bool
            Whether to create interactive Folium maps
        """
        print(f"\n=== Creating Comprehensive Visualizations ===")
        
        if self.results_df is None:
            raise ValueError("Must run calculate_composite_risk_score() first")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("viridis")
        
        # 1. Main composite risk map with base map
        self._create_main_risk_map(output_path)
        
        # 2. Individual risk component maps
        self._create_component_maps(output_path)
        
        # 3. Statistical analysis plots
        self._create_statistical_plots(output_path)
        
        # 4. Healthcare accessibility analysis
        self._create_healthcare_analysis(output_path)
        
        # 5. Interactive maps (if requested)
        if include_interactive:
            self._create_interactive_maps(output_path)
        
        # 6. Priority schools identification
        self._create_priority_analysis(output_path)
        
        print(f"✅ All visualizations saved to {output_path}")

    def _create_main_risk_map(self, output_path):
        """Create main composite risk map with base map"""
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        
        # Plot schools colored by risk with larger points
        risk_colors = {'Low Risk': '#2E8B57', 'Medium-Low Risk': '#FFD700', 
                      'Medium-High Risk': '#FF8C00', 'High Risk': '#DC143C'}
        
        for risk_cat, color in risk_colors.items():
            mask = self.results_df['risk_category'] == risk_cat
            if mask.any():
                schools_subset = self.results_df[mask]
                ax.scatter(
                    [point.x for point in schools_subset.geometry],
                    [point.y for point in schools_subset.geometry],
                    c=color, s=120, alpha=0.8, 
                    label=f'{risk_cat} ({mask.sum()} schools)',
                    edgecolors='black', linewidth=0.8
                )
                
                # Add labels for high-risk schools
                if risk_cat in ['High Risk', 'Medium-High Risk']:
                    for idx, school in schools_subset.iterrows():
                        school_name = school.get('name', f'School_{idx}')
                        if pd.notna(school_name) and school_name != 'nan':
                            ax.annotate(school_name[:20], 
                                      (school.geometry.x, school.geometry.y),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, alpha=0.7,
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        # Plot healthcare facilities with different symbols and labels
        if self.healthcare_facilities is not None:
            facility_symbols = {
                'hospital': ('s', '#8B0000', 150, 'Hospitals'),
                'clinic': ('D', '#0000CD', 80, 'Clinics'),
                'doctors': ('o', '#00CED1', 60, 'Doctors'),
                'pharmacy': ('+', '#9370DB', 40, 'Pharmacies'),
                'dentist': ('x', '#FF6347', 30, 'Dentists')
            }
            
            for facility_type, (marker, color, size, label) in facility_symbols.items():
                type_mask = self.healthcare_facilities['amenity'].str.contains(
                    facility_type, case=False, na=False
                )
                if type_mask.any():
                    facilities = self.healthcare_facilities[type_mask]
                    ax.scatter([point.x for point in facilities.geometry],
                             [point.y for point in facilities.geometry],
                             marker=marker, c=color, s=size, alpha=0.9,
                             label=f'{label} ({type_mask.sum()})',
                             edgecolors='white', linewidth=1)
                    
                    # Add labels for hospitals and major clinics
                    if facility_type in ['hospital'] and len(facilities) <= 20:
                        for idx, facility in facilities.iterrows():
                            facility_name = facility.get('name', f'{facility_type}_{idx}')
                            if pd.notna(facility_name) and facility_name != 'nan':
                                ax.annotate(facility_name[:25],
                                          (facility.geometry.x, facility.geometry.y),
                                          xytext=(8, 8), textcoords='offset points',
                                          fontsize=9, fontweight='bold',
                                          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        # Add base map context if possible
        try:
            ctx.add_basemap(ax, crs=self.results_df.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)
        except Exception as e:
            print(f"Note: Could not add base map: {e}")
        
        ax.set_title(f'{self.country_name}: Comprehensive School Risk Assessment\n'
                    f'Healthcare Access & Infrastructure Analysis (CCRI Supplement)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=14)
        ax.set_ylabel('Latitude', fontsize=14)
        
        # Create legend with two columns
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', 
                 ncol=1, fontsize=10, title='Risk Categories & Facilities', title_fontsize=12)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'{self.country_name.lower()}_comprehensive_risk_map.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_component_maps(self, output_path):
        """Create individual risk component maps"""
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
                    [point.x for point in self.results_df.geometry],
                    [point.y for point in self.results_df.geometry],
                    c=self.results_df[component],
                    cmap='RdYlBu_r',
                    s=80,
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=0.5
                )
                axes[i].set_title(title, fontweight='bold', fontsize=14)
                axes[i].set_xlabel('Longitude')
                axes[i].set_ylabel('Latitude')
                axes[i].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[i], shrink=0.8)
        
        # Hide the extra subplot
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.country_name.lower()}_risk_components.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_statistical_plots(self, output_path):
        """Create statistical analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Risk distribution histogram
        axes[0,0].hist(self.results_df['composite_risk_score'], bins=30, alpha=0.7, 
                      color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribution of Composite Risk Scores', fontweight='bold')
        axes[0,0].set_xlabel('Risk Score')
        axes[0,0].set_ylabel('Number of Schools')
        axes[0,0].grid(True, alpha=0.3)
        
        # Risk category pie chart
        risk_counts = self.results_df['risk_category'].value_counts()
        colors = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']
        axes[0,1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                     colors=colors[:len(risk_counts)], startangle=90)
        axes[0,1].set_title('Risk Category Distribution', fontweight='bold')
        
        # Risk factor correlation matrix
        risk_cols = ['healthcare_distance_risk', 'healthcare_capacity_risk',
                    'road_condition_risk', 'infrastructure_age_risk', 'environmental_risk']
        risk_cols = [col for col in risk_cols if col in self.results_df.columns]
        
        if len(risk_cols) > 1:
            corr_matrix = self.results_df[risk_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes[0,2], square=True, cbar_kws={'shrink': 0.8})
            axes[0,2].set_title('Risk Factor Correlations', fontweight='bold')
        
        # Healthcare distance vs risk
        if 'nearest_overall_dist' in self.results_df.columns:
            valid_distances = self.results_df['nearest_overall_dist'].dropna()
            valid_risks = self.results_df.loc[valid_distances.index, 'composite_risk_score']
            
            axes[1,0].scatter(valid_distances, valid_risks, alpha=0.6, s=50)
            axes[1,0].set_xlabel('Distance to Nearest Healthcare (degrees)')
            axes[1,0].set_ylabel('Composite Risk Score')
            axes[1,0].set_title('Healthcare Distance vs Overall Risk', fontweight='bold')
            axes[1,0].grid(True, alpha=0.3)
        
        # Priority ranking distribution
        axes[1,1].hist(self.results_df['priority_rank'], bins=20, alpha=0.7,
                      color='orange', edgecolor='black')
        axes[1,1].set_title('Priority Ranking Distribution', fontweight='bold')
        axes[1,1].set_xlabel('Priority Rank')
        axes[1,1].set_ylabel('Number of Schools')
        axes[1,1].grid(True, alpha=0.3)
        
        # Risk by facility access
        if 'facilities_within_10km' in self.results_df.columns:
            facility_bins = [0, 1, 3, 5, float('inf')]
            facility_labels = ['0-1', '2-3', '4-5', '6+']
            self.results_df['facility_access_group'] = pd.cut(
                self.results_df['facilities_within_10km'], 
                bins=facility_bins, labels=facility_labels
            )
            
            sns.boxplot(data=self.results_df, x='facility_access_group', y='composite_risk_score',
                       ax=axes[1,2])
            axes[1,2].set_title('Risk by Healthcare Facility Access', fontweight='bold')
            axes[1,2].set_xlabel('Facilities within 10km')
            axes[1,2].set_ylabel('Composite Risk Score')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.country_name.lower()}_statistical_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_healthcare_analysis(self, output_path):
        """Create detailed healthcare accessibility analysis"""
        if 'nearest_overall_dist' not in self.results_df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Distance distribution by risk category
        risk_categories = self.results_df['risk_category'].unique()
        distance_data = []
        categories = []
        
        for category in risk_categories:
            mask = self.results_df['risk_category'] == category
            distances = self.results_df.loc[mask, 'nearest_overall_dist'].dropna()
            distance_data.extend(distances)
            categories.extend([category] * len(distances))
        
        if distance_data:
            distance_df = pd.DataFrame({'Distance': distance_data, 'Risk Category': categories})
            sns.boxplot(data=distance_df, x='Risk Category', y='Distance', ax=axes[0,0])
            axes[0,0].set_title('Healthcare Distance by Risk Category', fontweight='bold')
            axes[0,0].set_ylabel('Distance to Nearest Healthcare (degrees)')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Facility type accessibility
        if 'nearest_facility_type' in self.results_df.columns:
            facility_access = self.results_df['nearest_facility_type'].value_counts()
            axes[0,1].pie(facility_access.values, labels=facility_access.index, autopct='%1.1f%%')
            axes[0,1].set_title('Primary Healthcare Access by Facility Type', fontweight='bold')
        
        # Geographic distribution of healthcare access
        if 'emergency_access_score' in self.results_df.columns:
            scatter = axes[1,0].scatter(
                [point.x for point in self.results_df.geometry],
                [point.y for point in self.results_df.geometry],
                c=self.results_df['emergency_access_score'],
                cmap='RdYlGn',
                s=60,
                alpha=0.7
            )
            axes[1,0].set_title('Emergency Healthcare Access Scores', fontweight='bold')
            axes[1,0].set_xlabel('Longitude')
            axes[1,0].set_ylabel('Latitude')
            plt.colorbar(scatter, ax=axes[1,0], label='Emergency Access Score')
        
        # Facility density analysis
        if 'facilities_within_10km' in self.results_df.columns:
            density_dist = self.results_df['facilities_within_10km'].value_counts().sort_index()
            axes[1,1].bar(density_dist.index, density_dist.values, alpha=0.7, color='teal')
            axes[1,1].set_title('Healthcare Facility Density Distribution', fontweight='bold')
            axes[1,1].set_xlabel('Number of Facilities within 10km')
            axes[1,1].set_ylabel('Number of Schools')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.country_name.lower()}_healthcare_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _create_interactive_maps(self, output_path):
        """Create interactive Folium maps"""
        try:
            # Calculate map center
            bounds = self.results_df.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            
            # Create main interactive map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
            
            # Add school markers with risk information
            risk_colors = {'Low Risk': 'green', 'Medium-Low Risk': 'yellow',
                          'Medium-High Risk': 'orange', 'High Risk': 'red'}
            
            for idx, school in self.results_df.iterrows():
                risk_cat = school['risk_category']
                school_name = school.get('name', f'School_{idx}')
                
                popup_text = f"""
                <b>{school_name}</b><br>
                Risk Category: {risk_cat}<br>
                Composite Risk: {school['composite_risk_score']:.3f}<br>
                Priority Rank: {school['priority_rank']}<br>
                Nearest Healthcare: {school.get('nearest_facility_name', 'Unknown')}<br>
                Distance: {school.get('nearest_overall_dist', 'Unknown'):.4f} degrees
                """
                
                folium.CircleMarker(
                    location=[school.geometry.y, school.geometry.x],
                    radius=8,
                    popup=folium.Popup(popup_text, max_width=300),
                    color='black',
                    fillColor=risk_colors.get(risk_cat, 'blue'),
                    fillOpacity=0.7,
                    weight=1
                ).add_to(m)
            
            # Add healthcare facilities
            if self.healthcare_facilities is not None:
                for idx, facility in self.healthcare_facilities.iterrows():
                    facility_name = facility.get('name', f'Healthcare_{idx}')
                    facility_type = facility.get('amenity', 'Unknown')
                    
                    popup_text = f"""
                    <b>{facility_name}</b><br>
                    Type: {facility_type}
                    """
                    
                    folium.Marker(
                        location=[facility.geometry.y, facility.geometry.x],
                        popup=folium.Popup(popup_text, max_width=200),
                        icon=folium.Icon(color='blue', icon='plus-sign')
                    ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save interactive map
            m.save(output_path / f'{self.country_name.lower()}_interactive_risk_map.html')
            print(f"✅ Interactive map saved as HTML")
            
        except Exception as e:
            print(f"⚠️  Could not create interactive map: {e}")

    def _create_priority_analysis(self, output_path):
        """Create priority schools analysis"""
        # Get top 20 highest risk schools
        top_priority = self.results_df.nlargest(20, 'composite_risk_score')
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Top priority schools map
        scatter = axes[0,0].scatter(
            [point.x for point in self.results_df.geometry],
            [point.y for point in self.results_df.geometry],
            c=self.results_df['composite_risk_score'],
            cmap='RdYlBu_r',
            s=60,
            alpha=0.6
        )
        
        # Highlight top priority schools
        axes[0,0].scatter(
            [point.x for point in top_priority.geometry],
            [point.y for point in top_priority.geometry],
            c='red',
            s=150,
            alpha=0.9,
            edgecolors='black',
            linewidth=2,
            marker='*'
        )
        
        axes[0,0].set_title('Top 20 Priority Schools for Intervention', fontweight='bold')
        axes[0,0].set_xlabel('Longitude')
        axes[0,0].set_ylabel('Latitude')
        plt.colorbar(scatter, ax=axes[0,0], label='Risk Score')
        
        # Priority schools table
        priority_table = top_priority[['name', 'risk_category', 'composite_risk_score',
                                     'nearest_facility_name', 'nearest_overall_dist']].head(10)
        
        axes[0,1].axis('tight')
        axes[0,1].axis('off')
        table = axes[0,1].table(cellText=priority_table.values,
                               colLabels=['School Name', 'Risk Category', 'Risk Score',
                                        'Nearest Healthcare', 'Distance'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        axes[0,1].set_title('Top 10 Priority Schools', fontweight='bold')
        
        # Risk score distribution for priority schools
        axes[1,0].hist(top_priority['composite_risk_score'], bins=10, alpha=0.7,
                      color='red', edgecolor='black')
        axes[1,0].set_title('Risk Score Distribution - Top 20 Priority Schools', fontweight='bold')
        axes[1,0].set_xlabel('Risk Score')
        axes[1,0].set_ylabel('Number of Schools')
        axes[1,0].grid(True, alpha=0.3)
        
        # Intervention cost-benefit analysis (simulated)
        axes[1,1].scatter(top_priority['composite_risk_score'], 
                         range(len(top_priority)),
                         s=100, alpha=0.7, color='orange')
        axes[1,1].set_title('Priority Ranking vs Risk Score', fontweight='bold')
        axes[1,1].set_xlabel('Composite Risk Score')
        axes[1,1].set_ylabel('Priority Rank')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{self.country_name.lower()}_priority_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

    def export_results(self, output_dir="risk_analysis_outputs", formats=['csv', 'geojson', 'excel']):
        """
        Export comprehensive analysis results in multiple formats.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save exported files
        formats : list
            List of export formats ('csv', 'geojson', 'excel', 'shapefile')
        """
        print(f"\n=== Exporting Results ===")
        
        if self.results_df is None:
            raise ValueError("Must run calculate_composite_risk_score() first")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.country_name.lower()}_school_risk_analysis_{timestamp}"
        
        # CSV export (without geometry)
        if 'csv' in formats:
            csv_df = self.results_df.drop(columns=['geometry'])
            csv_path = output_path / f"{base_filename}.csv"
            csv_df.to_csv(csv_path, index=False)
            print(f"✅ CSV export: {csv_path}")
        
        # GeoJSON export (with geometry)
        if 'geojson' in formats:
            geojson_path = output_path / f"{base_filename}.geojson"
            self.results_df.to_file(geojson_path, driver='GeoJSON')
            print(f"✅ GeoJSON export: {geojson_path}")
        
        # Excel export with multiple sheets
        if 'excel' in formats:
            excel_path = output_path / f"{base_filename}.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Main results
                csv_df = self.results_df.drop(columns=['geometry'])
                csv_df.to_excel(writer, sheet_name='School_Risk_Analysis', index=False)
                
                # Summary statistics
                summary_stats = self._generate_summary_statistics()
                pd.DataFrame([summary_stats]).to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Priority schools
                priority_schools = self.results_df.nlargest(50, 'composite_risk_score').drop(columns=['geometry'])
                priority_schools.to_excel(writer, sheet_name='Priority_Schools', index=False)
                
                # Risk distribution
                risk_dist = self.results_df['risk_category'].value_counts().to_frame('Count')
                risk_dist.to_excel(writer, sheet_name='Risk_Distribution')
            
            print(f"✅ Excel export: {excel_path}")
        
        # Shapefile export
        if 'shapefile' in formats:
            try:
                shp_path = output_path / f"{base_filename}.shp"
                self.results_df.to_file(shp_path, driver='ESRI Shapefile')
                print(f"✅ Shapefile export: {shp_path}")
            except Exception as e:
                print(f"⚠️  Could not export shapefile: {e}")
        
        # Export summary report
        self._export_summary_report(output_path, base_filename)
        
        return output_path

    def _generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        summary = {
            'country': self.country_name,
            'analysis_date': datetime.now().isoformat(),
            'total_schools_analyzed': len(self.results_df),
            'total_healthcare_facilities': len(self.healthcare_facilities) if self.healthcare_facilities is not None else 0,
            'high_risk_schools': len(self.results_df[self.results_df['risk_category'] == 'High Risk']),
            'medium_high_risk_schools': len(self.results_df[self.results_df['risk_category'] == 'Medium-High Risk']),
            'medium_low_risk_schools': len(self.results_df[self.results_df['risk_category'] == 'Medium-Low Risk']),
            'low_risk_schools': len(self.results_df[self.results_df['risk_category'] == 'Low Risk']),
            'average_composite_risk': float(self.results_df['composite_risk_score'].mean()),
            'median_composite_risk': float(self.results_df['composite_risk_score'].median()),
            'max_composite_risk': float(self.results_df['composite_risk_score'].max()),
            'min_composite_risk': float(self.results_df['composite_risk_score'].min()),
            'avg_healthcare_distance': float(self.results_df['nearest_overall_dist'].mean()) if 'nearest_overall_dist' in self.results_df.columns else None,
            'schools_without_nearby_healthcare': len(self.results_df[self.results_df['facilities_within_10km'] == 0]) if 'facilities_within_10km' in self.results_df.columns else None,
            'risk_weights_used': self.risk_weights,
            'analysis_parameters': self.analysis_params
        }
        
        return summary

    def _export_summary_report(self, output_path, base_filename):
        """Export comprehensive summary report"""
        summary_stats = self._generate_summary_statistics()
        
        # JSON export
        json_path = output_path / f"{base_filename}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        # Text report
        report_path = output_path / f"{base_filename}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"SCHOOL RISK ANALYSIS REPORT\n")
            f.write(f"={'='*50}\n\n")
            f.write(f"Country: {summary_stats['country']}\n")
            f.write(f"Analysis Date: {summary_stats['analysis_date']}\n")
            f.write(f"Total Schools Analyzed: {summary_stats['total_schools_analyzed']}\n")
            f.write(f"Total Healthcare Facilities: {summary_stats['total_healthcare_facilities']}\n\n")
            
            f.write(f"RISK DISTRIBUTION:\n")
            f.write(f"- High Risk: {summary_stats['high_risk_schools']} schools\n")
            f.write(f"- Medium-High Risk: {summary_stats['medium_high_risk_schools']} schools\n")
            f.write(f"- Medium-Low Risk: {summary_stats['medium_low_risk_schools']} schools\n")
            f.write(f"- Low Risk: {summary_stats['low_risk_schools']} schools\n\n")
            
            f.write(f"RISK STATISTICS:\n")
            f.write(f"- Average Risk Score: {summary_stats['average_composite_risk']:.4f}\n")
            f.write(f"- Median Risk Score: {summary_stats['median_composite_risk']:.4f}\n")
            f.write(f"- Maximum Risk Score: {summary_stats['max_composite_risk']:.4f}\n")
            f.write(f"- Minimum Risk Score: {summary_stats['min_composite_risk']:.4f}\n\n")
            
            if summary_stats['avg_healthcare_distance']:
                f.write(f"HEALTHCARE ACCESS:\n")
                f.write(f"- Average Distance to Healthcare: {summary_stats['avg_healthcare_distance']:.4f} degrees\n")
                if summary_stats['schools_without_nearby_healthcare'] is not None:
                    f.write(f"- Schools without Healthcare within 10km: {summary_stats['schools_without_nearby_healthcare']}\n\n")
            
            f.write(f"METHODOLOGY:\n")
            f.write(f"Risk Weights Used:\n")
            for factor, weight in summary_stats['risk_weights_used'].items():
                f.write(f"  - {factor}: {weight}\n")
        
        print(f"✅ Summary report: {report_path}")
        print(f"✅ JSON summary: {json_path}")

    def run_complete_analysis(self, output_dir="risk_analysis_outputs", **kwargs):
        """
        Run the complete risk analysis pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Directory for all outputs
        **kwargs : dict
            Additional parameters for analysis components
        """
        print(f"🌍 COMPREHENSIVE SCHOOL RISK ANALYSIS")
        print(f"🎯 Country: {self.country_name}")
        print(f"📊 Supplementing UNICEF CCRI with Fine-Grained Assessment")
        print("=" * 70)
        
        try:
            # Load data
            print("\n1️⃣  Loading and validating data...")
            self.load_data(**kwargs.get('load_params', {}))
            
            # Calculate composite risk
            print("\n2️⃣  Calculating comprehensive risk scores...")
            results = self.calculate_composite_risk_score()
            
            # Create visualizations
            print("\n3️⃣  Creating comprehensive visualizations...")
            self.create_comprehensive_visualizations(output_dir, **kwargs.get('viz_params', {}))
            
            # Export results
            print("\n4️⃣  Exporting results in multiple formats...")
            export_path = self.export_results(output_dir, **kwargs.get('export_params', {}))
            
            # Generate final summary
            summary_stats = self._generate_summary_statistics()
            
            print("\n" + "=" * 70)
            print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
            print("=" * 70)
            
            print(f"\n📊 Analysis Summary:")
            print(f"• Total Schools Analyzed: {summary_stats['total_schools_analyzed']}")
            print(f"• High Risk Schools: {summary_stats['high_risk_schools']}")
            print(f"• Medium-High Risk Schools: {summary_stats['medium_high_risk_schools']}")
            print(f"• Average Risk Score: {summary_stats['average_composite_risk']:.3f}")
            print(f"• Healthcare Facilities: {summary_stats['total_healthcare_facilities']}")
            
            print(f"\n📁 All outputs saved to: {export_path}")
            
            print(f"\n🔗 CCRI Integration Ready:")
            print(f"• Composite risk scores calculated for all schools")
            print(f"• Fine-grained healthcare accessibility metrics available")
            print(f"• Infrastructure quality assessments completed")
            print(f"• Priority rankings established for targeted interventions")
            print(f"• Data exported in UNICEF-compatible formats")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            raise


# Example usage and configuration
if __name__ == "__main__":
    
    # Example: Tajikistan Analysis
    print("🏫 SCHOOL RISK MAPPING SYSTEM - DEMONSTRATION")
    print("=" * 60)
    
    # Configuration for Tajikistan
    tajikistan_config = {
        'risk_weights': {
            'healthcare_distance': 0.35,
            'healthcare_capacity': 0.20,
            'road_conditions': 0.25,
            'infrastructure_age': 0.15,
            'environmental_hazards': 0.05
        },
        'analysis_params': {
            'distance_thresholds': {
                'emergency': 0.05,    # 5km
                'routine': 0.1,       # 10km  
                'accessible': 0.2     # 20km
            },
            'facility_priorities': {
                'hospital': 1.0,
                'clinic': 0.8,
                'doctors': 0.6,
                'pharmacy': 0.4,
                'dentist': 0.3
            }
        }
    }
    
    data_dir = Path(os.getcwd()) / "data"
    mapper = SchoolRiskMapper(
        country_name="Tajikistan",
        data_dir=str(data_dir),
        config=tajikistan_config
    )

    try:
        results = mapper.run_complete_analysis(
            output_dir="tajikistan_comprehensive_analysis",
            load_params={
                'poi_file': str(data_dir / "hotosm_tjk_points_of_interest_points_geojson.geojson"),
                'education_file': str(data_dir / "hotosm_tjk_education_facilities_points_geojson.geojson"),
                'infrastructure_file': str(data_dir / "hotosm_tjk_buildings_polygons_geojson.geojson"),
                'roads_file': str(data_dir / "hotosm_tjk_roads_lines_geojson.geojson"),
                'auto_detect': False
            },
            viz_params={ 'include_interactive': True },
            export_params={ 'formats': ['csv','geojson','excel'] }
        )
        print("✨ Ready for UNICEF GeoSight integration!")
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("Please ensure data files are available and paths are correct.")
