#!/usr/bin/env python3
"""
Configuration Templates and Utility Scripts
============================================

This file contains configuration templates and utility functions to support
the Enhanced School Risk Mapping System implementation.

Contents:
- Configuration templates for different countries/scenarios
- Data validation and preparation utilities
- Batch processing functions
- Quality assurance tools
"""

import json
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================
# CONFIGURATION TEMPLATES
# ================================

class ConfigurationManager:
    """Manages configuration templates for different analysis scenarios."""
    
    @staticmethod
    def get_country_config(country_name: str) -> Dict:
        """Get country-specific configuration."""
        configs = {
            "tajikistan": {
                "risk_weights": {
                    "healthcare_distance": 0.35,
                    "healthcare_capacity": 0.20,
                    "road_conditions": 0.25,
                    "infrastructure_age": 0.15,
                    "environmental_hazards": 0.05
                },
                "distance_thresholds": {
                    "emergency": 0.05,    # ~5km in mountainous terrain
                    "routine": 0.1,       # ~10km
                    "accessible": 0.2     # ~20km maximum
                },
                "facility_priorities": {
                    "hospital": 1.0,
                    "clinic": 0.8,
                    "doctors": 0.6,
                    "pharmacy": 0.4,
                    "dentist": 0.3
                },
                "infrastructure_keywords": {
                    "modern": ["гимназия", "лицей", "центр", "academy", "modern"],
                    "traditional": ["школа", "мактаб", "basic", "elementary"]
                }
            },
            
            "kenya": {
                "risk_weights": {
                    "healthcare_distance": 0.30,
                    "healthcare_capacity": 0.25,
                    "road_conditions": 0.20,
                    "infrastructure_age": 0.15,
                    "environmental_hazards": 0.10
                },
                "distance_thresholds": {
                    "emergency": 0.03,    # ~3km in more accessible terrain
                    "routine": 0.08,      # ~8km
                    "accessible": 0.15    # ~15km maximum
                },
                "facility_priorities": {
                    "hospital": 1.0,
                    "health_center": 0.9,
                    "clinic": 0.8,
                    "dispensary": 0.6,
                    "pharmacy": 0.4
                },
                "infrastructure_keywords": {
                    "modern": ["academy", "institute", "college", "modern", "new"],
                    "traditional": ["primary", "secondary", "basic", "community"]
                }
            },
            
            "afghanistan": {
                "risk_weights": {
                    "healthcare_distance": 0.40,    # Critical in conflict areas
                    "healthcare_capacity": 0.25,
                    "road_conditions": 0.20,
                    "infrastructure_age": 0.10,
                    "environmental_hazards": 0.05
                },
                "distance_thresholds": {
                    "emergency": 0.08,    # Longer distances due to terrain/security
                    "routine": 0.15,
                    "accessible": 0.30
                },
                "facility_priorities": {
                    "hospital": 1.0,
                    "clinic": 0.8,
                    "basic_health_unit": 0.7,
                    "mobile_clinic": 0.6,
                    "pharmacy": 0.3
                },
                "infrastructure_keywords": {
                    "modern": ["institute", "academy", "modern", "new", "reconstruction"],
                    "traditional": ["madrassa", "basic", "community", "temporary"]
                }
            },
            
            "generic_mountainous": {
                "risk_weights": {
                    "healthcare_distance": 0.35,
                    "healthcare_capacity": 0.20,
                    "road_conditions": 0.30,    # Roads critical in mountains
                    "infrastructure_age": 0.10,
                    "environmental_hazards": 0.05
                },
                "distance_thresholds": {
                    "emergency": 0.06,
                    "routine": 0.12,
                    "accessible": 0.25
                },
                "facility_priorities": {
                    "hospital": 1.0,
                    "clinic": 0.8,
                    "health_post": 0.6,
                    "pharmacy": 0.4
                }
            },
            
            "generic_urban": {
                "risk_weights": {
                    "healthcare_distance": 0.25,
                    "healthcare_capacity": 0.30,    # Many facilities available
                    "road_conditions": 0.20,
                    "infrastructure_age": 0.15,
                    "environmental_hazards": 0.10
                },
                "distance_thresholds": {
                    "emergency": 0.02,    # Short distances in cities
                    "routine": 0.05,
                    "accessible": 0.10
                },
                "facility_priorities": {
                    "emergency_hospital": 1.2,
                    "hospital": 1.0,
                    "clinic": 0.8,
                    "health_center": 0.7,
                    "pharmacy": 0.5
                }
            }
        }
        
        return configs.get(country_name.lower(), configs["generic_mountainous"])
    
    @staticmethod
    def get_scenario_config(scenario: str) -> Dict:
        """Get scenario-specific configuration."""
        scenarios = {
            "emergency_response": {
                "risk_weights": {
                    "healthcare_distance": 0.50,    # Primary concern in emergencies
                    "healthcare_capacity": 0.30,
                    "road_conditions": 0.15,
                    "infrastructure_age": 0.05,
                    "environmental_hazards": 0.00
                },
                "priority_facilities": ["hospital", "emergency_clinic", "mobile_unit"]
            },
            
            "development_planning": {
                "risk_weights": {
                    "healthcare_distance": 0.25,
                    "healthcare_capacity": 0.25,
                    "road_conditions": 0.25,
                    "infrastructure_age": 0.25,
                    "environmental_hazards": 0.00
                },
                "focus": "long_term_infrastructure"
            },
            
            "climate_adaptation": {
                "risk_weights": {
                    "healthcare_distance": 0.20,
                    "healthcare_capacity": 0.20,
                    "road_conditions": 0.20,
                    "infrastructure_age": 0.15,
                    "environmental_hazards": 0.25    # Climate focus
                },
                "climate_factors": ["flood_risk", "drought_risk", "extreme_weather"]
            }
        }
        
        return scenarios.get(scenario, scenarios["development_planning"])

# ================================
# DATA VALIDATION UTILITIES
# ================================

class DataValidator:
    """Comprehensive data validation for school risk analysis."""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_healthcare_data(self, df: pd.DataFrame, 
                                expected_columns: List[str] = None) -> Dict:
        """Validate healthcare facilities data."""
        if expected_columns is None:
            expected_columns = ['X', 'Y', 'amenity', 'name']
        
        results = {
            "total_records": len(df),
            "missing_coordinates": 0,
            "missing_amenity": 0,
            "missing_names": 0,
            "invalid_coordinates": 0,
            "duplicate_records": 0,
            "facility_type_distribution": {},
            "coordinate_bounds": {},
            "quality_score": 0,
            "recommendations": []
        }
        
        # Check for missing coordinates
        coord_cols = ['X', 'Y'] if 'X' in df.columns else ['longitude', 'latitude']
        if all(col in df.columns for col in coord_cols):
            missing_coords = df[coord_cols].isnull().any(axis=1).sum()
            results["missing_coordinates"] = missing_coords
            
            # Check coordinate validity
            if len(coord_cols) == 2:
                lon_col, lat_col = coord_cols
                invalid_lon = ((df[lon_col] < -180) | (df[lon_col] > 180)).sum()
                invalid_lat = ((df[lat_col] < -90) | (df[lat_col] > 90)).sum()
                results["invalid_coordinates"] = invalid_lon + invalid_lat
                
                # Calculate bounds
                valid_coords = df.dropna(subset=coord_cols)
                if len(valid_coords) > 0:
                    results["coordinate_bounds"] = {
                        "min_lon": float(valid_coords[lon_col].min()),
                        "max_lon": float(valid_coords[lon_col].max()),
                        "min_lat": float(valid_coords[lat_col].min()),
                        "max_lat": float(valid_coords[lat_col].max())
                    }
        
        # Check amenity field
        if 'amenity' in df.columns:
            results["missing_amenity"] = df['amenity'].isnull().sum()
            results["facility_type_distribution"] = df['amenity'].value_counts().to_dict()
        
        # Check names
        if 'name' in df.columns:
            results["missing_names"] = df['name'].isnull().sum()
        
        # Check for duplicates
        if all(col in df.columns for col in coord_cols):
            duplicates = df.duplicated(subset=coord_cols).sum()
            results["duplicate_records"] = duplicates
        
        # Calculate quality score
        total_records = results["total_records"]
        if total_records > 0:
            completeness = 1 - (results["missing_coordinates"] + 
                               results["missing_amenity"]) / (total_records * 2)
            validity = 1 - results["invalid_coordinates"] / total_records
            uniqueness = 1 - results["duplicate_records"] / total_records
            results["quality_score"] = (completeness + validity + uniqueness) / 3
        
        # Generate recommendations
        if results["missing_coordinates"] > 0:
            results["recommendations"].append(
                f"Fix {results['missing_coordinates']} records with missing coordinates"
            )
        if results["invalid_coordinates"] > 0:
            results["recommendations"].append(
                f"Correct {results['invalid_coordinates']} records with invalid coordinates"
            )
        if results["missing_amenity"] > 0:
            results["recommendations"].append(
                f"Add facility type for {results['missing_amenity']} records"
            )
        
        return results
    
    def validate_education_data(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Validate education facilities data."""
        results = {
            "total_records": len(gdf),
            "missing_geometry": 0,
            "invalid_geometry": 0,
            "missing_amenity": 0,
            "missing_names": 0,
            "education_type_distribution": {},
            "coordinate_bounds": {},
            "quality_score": 0,
            "recommendations": []
        }
        
        # Check geometry
        if 'geometry' in gdf.columns:
            missing_geom = gdf['geometry'].isnull().sum()
            results["missing_geometry"] = missing_geom
            
            # Check geometry validity
            valid_geom = gdf[gdf['geometry'].notnull()]
            if len(valid_geom) > 0:
                invalid_geom = (~valid_geom['geometry'].is_valid).sum()
                results["invalid_geometry"] = invalid_geom
                
                # Calculate bounds
                bounds = valid_geom.total_bounds
                results["coordinate_bounds"] = {
                    "min_lon": float(bounds[0]),
                    "min_lat": float(bounds[1]),
                    "max_lon": float(bounds[2]),
                    "max_lat": float(bounds[3])
                }
        
        # Check amenity field
        if 'amenity' in gdf.columns:
            results["missing_amenity"] = gdf['amenity'].isnull().sum()
            results["education_type_distribution"] = gdf['amenity'].value_counts().to_dict()
        
        # Check names
        if 'name' in gdf.columns:
            results["missing_names"] = gdf['name'].isnull().sum()
        
        # Calculate quality score
        total_records = results["total_records"]
        if total_records > 0:
            completeness = 1 - (results["missing_geometry"] + 
                               results["missing_amenity"]) / (total_records * 2)
            validity = 1 - results["invalid_geometry"] / total_records
            results["quality_score"] = (completeness + validity) / 2
        
        # Generate recommendations
        if results["missing_geometry"] > 0:
            results["recommendations"].append(
                f"Fix {results['missing_geometry']} records with missing geometry"
            )
        if results["invalid_geometry"] > 0:
            results["recommendations"].append(
                f"Repair {results['invalid_geometry']} records with invalid geometry"
            )
        
        return results
    
    def generate_validation_report(self, healthcare_results: Dict, 
                                 education_results: Dict) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Healthcare data section
        report.append("HEALTHCARE FACILITIES DATA:")
        report.append(f"  Total Records: {healthcare_results['total_records']}")
        report.append(f"  Quality Score: {healthcare_results['quality_score']:.2f}/1.0")
        report.append(f"  Missing Coordinates: {healthcare_results['missing_coordinates']}")
        report.append(f"  Invalid Coordinates: {healthcare_results['invalid_coordinates']}")
        report.append(f"  Missing Facility Types: {healthcare_results['missing_amenity']}")
        report.append(f"  Duplicate Records: {healthcare_results['duplicate_records']}")
        report.append("")
        
        if healthcare_results['facility_type_distribution']:
            report.append("  Facility Distribution:")
            for facility_type, count in healthcare_results['facility_type_distribution'].items():
                report.append(f"    {facility_type}: {count}")
        report.append("")
        
        # Education data section
        report.append("EDUCATION FACILITIES DATA:")
        report.append(f"  Total Records: {education_results['total_records']}")
        report.append(f"  Quality Score: {education_results['quality_score']:.2f}/1.0")
        report.append(f"  Missing Geometry: {education_results['missing_geometry']}")
        report.append(f"  Invalid Geometry: {education_results['invalid_geometry']}")
        report.append(f"  Missing Facility Types: {education_results['missing_amenity']}")
        report.append("")
        
        if education_results['education_type_distribution']:
            report.append("  Education Type Distribution:")
            for edu_type, count in education_results['education_type_distribution'].items():
                report.append(f"    {edu_type}: {count}")
        report.append("")
        
        # Recommendations
        all_recommendations = (healthcare_results['recommendations'] + 
                             education_results['recommendations'])
        if all_recommendations:
            report.append("RECOMMENDATIONS:")
            for i, rec in enumerate(all_recommendations, 1):
                report.append(f"  {i}. {rec}")
        else:
            report.append("✅ No major data quality issues detected!")
        
        return "\n".join(report)

# ================================
# DATA PREPARATION UTILITIES
# ================================

class DataPreprocessor:
    """Utilities for data cleaning and preparation."""
    
    @staticmethod
    def standardize_healthcare_data(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize healthcare data format."""
        df_clean = df.copy()
        
        # Standardize coordinate columns
        coord_mappings = {
            ('lon', 'lat'): ('X', 'Y'),
            ('longitude', 'latitude'): ('X', 'Y'),
            ('lng', 'lat'): ('X', 'Y')
        }
        
        for (lon_col, lat_col), (target_lon, target_lat) in coord_mappings.items():
            if lon_col in df_clean.columns and lat_col in df_clean.columns:
                df_clean = df_clean.rename(columns={lon_col: target_lon, lat_col: target_lat})
                break
        
        # Standardize amenity values
        amenity_mappings = {
            'hospitals': 'hospital',
            'medical_center': 'hospital',
            'health_center': 'clinic',
            'health_centre': 'clinic',
            'dispensary': 'clinic',
            'medical_clinic': 'clinic',
            'drug_store': 'pharmacy',
            'chemist': 'pharmacy'
        }
        
        if 'amenity' in df_clean.columns:
            df_clean['amenity'] = df_clean['amenity'].str.lower()
            df_clean['amenity'] = df_clean['amenity'].replace(amenity_mappings)
        
        # Remove invalid coordinates
        if 'X' in df_clean.columns and 'Y' in df_clean.columns:
            valid_coords = (
                (df_clean['X'] >= -180) & (df_clean['X'] <= 180) &
                (df_clean['Y'] >= -90) & (df_clean['Y'] <= 90) &
                df_clean['X'].notnull() & df_clean['Y'].notnull()
            )
            df_clean = df_clean[valid_coords]
        
        return df_clean
    
    @staticmethod
    def standardize_education_data(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardize education data format."""
        gdf_clean = gdf.copy()
        
        # Standardize amenity values
        amenity_mappings = {
            'primary_school': 'school',
            'secondary_school': 'school',
            'elementary_school': 'school',
            'high_school': 'school',
            'middle_school': 'school'
        }
        
        if 'amenity' in gdf_clean.columns:
            gdf_clean['amenity'] = gdf_clean['amenity'].str.lower()
            gdf_clean['amenity'] = gdf_clean['amenity'].replace(amenity_mappings)
        
        # Remove invalid geometries
        if 'geometry' in gdf_clean.columns:
            valid_geom = gdf_clean['geometry'].notnull() & gdf_clean['geometry'].is_valid
            gdf_clean = gdf_clean[valid_geom]
        
        # Ensure CRS is set
        if gdf_clean.crs is None:
            gdf_clean.set_crs('EPSG:4326', inplace=True)
        
        return gdf_clean
    
    @staticmethod
    def merge_healthcare_sources(sources: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple healthcare data sources."""
        if not sources:
            return pd.DataFrame()
        
        # Standardize all sources
        standardized = [DataPreprocessor.standardize_healthcare_data(df) for df in sources]
        
        # Combine all sources
        combined = pd.concat(standardized, ignore_index=True)
        
        # Remove duplicates based on coordinates and name
        duplicate_cols = ['X', 'Y']
        if 'name' in combined.columns:
            duplicate_cols.append('name')
        
        combined = combined.drop_duplicates(subset=duplicate_cols, keep='first')
        
        return combined

# ================================
# BATCH PROCESSING UTILITIES
# ================================

class BatchProcessor:
    """Utilities for processing multiple countries or datasets."""
    
    def __init__(self, base_output_dir: str = "batch_analysis_outputs"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.results_summary = []
    
    def process_multiple_countries(self, country_configs: List[Dict]) -> Dict:
        """Process multiple countries in batch."""
        """
        country_configs format:
        [
            {
                'name': 'Tajikistan',
                'data_dir': '/path/to/tajikistan/data',
                'healthcare_file': 'tajikistan.csv',
                'education_file': 'schools.geojson',
                'config': {...}  # Optional custom config
            },
            ...
        ]
        """
        
        batch_results = {
            'successful': [],
            'failed': [],
            'summary_statistics': {},
            'processing_log': []
        }
        
        for country_config in country_configs:
            country_name = country_config['name']
            logger.info(f"Processing {country_name}...")
            
            try:
                # Import here to avoid circular imports
                from school_risk_mapper import SchoolRiskMapper
                
                # Get configuration
                config = country_config.get('config', 
                                          ConfigurationManager.get_country_config(country_name))
                
                # Initialize mapper
                mapper = SchoolRiskMapper(
                    country_name=country_name,
                    data_dir=country_config['data_dir'],
                    config=config
                )
                
                # Create country-specific output directory
                country_output_dir = self.base_output_dir / f"{country_name.lower()}_analysis"
                
                # Run analysis
                results = mapper.run_complete_analysis(
                    output_dir=str(country_output_dir),
                    load_params={
                        'healthcare_file': country_config.get('healthcare_file'),
                        'education_file': country_config.get('education_file'),
                        'auto_detect': True
                    }
                )
                
                # Store success
                country_summary = {
                    'country': country_name,
                    'status': 'success',
                    'total_schools': len(results),
                    'high_risk_schools': len(results[results['risk_category'].isin(['High Risk', 'Critical Risk'])]),
                    'output_directory': str(country_output_dir)
                }
                
                batch_results['successful'].append(country_summary)
                self.results_summary.append(country_summary)
                
                logger.info(f"✅ {country_name} processed successfully")
                
            except Exception as e:
                error_summary = {
                    'country': country_name,
                    'status': 'failed',
                    'error': str(e)
                }
                
                batch_results['failed'].append(error_summary)
                logger.error(f"❌ {country_name} processing failed: {e}")
        
        # Generate batch summary
        batch_results['summary_statistics'] = self._generate_batch_summary()
        
        # Save batch results
        self._save_batch_results(batch_results)
        
        return batch_results
    
    def _generate_batch_summary(self) -> Dict:
        """Generate summary statistics for batch processing."""
        if not self.results_summary:
            return {}
        
        total_schools = sum(r['total_schools'] for r in self.results_summary)
        total_high_risk = sum(r['high_risk_schools'] for r in self.results_summary)
        
        return {
            'countries_processed': len(self.results_summary),
            'total_schools_analyzed': total_schools,
            'total_high_risk_schools': total_high_risk,
            'average_schools_per_country': total_schools / len(self.results_summary) if self.results_summary else 0,
            'high_risk_percentage': (total_high_risk / total_schools * 100) if total_schools > 0 else 0
        }
    
    def _save_batch_results(self, batch_results: Dict):
        """Save batch processing results."""
        # Save as JSON
        with open(self.base_output_dir / 'batch_processing_results.json', 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        # Save summary as CSV
        if self.results_summary:
            summary_df = pd.DataFrame(self.results_summary)
            summary_df.to_csv(self.base_output_dir / 'batch_summary.csv', index=False)

# ================================
# QUALITY ASSURANCE UTILITIES
# ================================

class QualityAssurance:
    """Quality assurance and testing utilities."""
    
    @staticmethod
    def validate_risk_scores(results_df: pd.DataFrame) -> Dict:
        """Validate risk score calculations."""
        qa_results = {
            'total_records': len(results_df),
            'risk_score_issues': [],
            'statistical_checks': {},
            'recommendations': []
        }
        
        # Check risk score ranges
        risk_columns = [col for col in results_df.columns if 'risk' in col.lower()]
        
        for col in risk_columns:
            if col in results_df.columns:
                min_val = results_df[col].min()
                max_val = results_df[col].max()
                
                # Risk scores should be between 0 and 1
                if min_val < 0 or max_val > 1:
                    qa_results['risk_score_issues'].append(
                        f"{col}: values outside [0,1] range ({min_val:.3f} to {max_val:.3f})"
                    )
        
        # Statistical checks
        if 'composite_risk_score' in results_df.columns:
            composite_scores = results_df['composite_risk_score']
            qa_results['statistical_checks'] = {
                'mean': float(composite_scores.mean()),
                'median': float(composite_scores.median()),
                'std': float(composite_scores.std()),
                'min': float(composite_scores.min()),
                'max': float(composite_scores.max()),
                'null_count': int(composite_scores.isnull().sum())
            }
            
            # Check for unusual distributions
            if composite_scores.std() < 0.1:
                qa_results['recommendations'].append(
                    "Low variance in risk scores - check if risk factors are properly differentiated"
                )
            
            if composite_scores.isnull().sum() > 0:
                qa_results['recommendations'].append(
                    f"Found {composite_scores.isnull().sum()} null risk scores"
                )
        
        return qa_results
    
    @staticmethod
    def cross_validate_results(results_df: pd.DataFrame, 
                             known_high_risk_areas: List[Tuple[float, float]] = None) -> Dict:
        """Cross-validate results against known high-risk areas."""
        validation_results = {
            'geographic_consistency': True,
            'known_area_validation': {},
            'outlier_detection': {}
        }
        
        # Check geographic consistency
        if 'nearest_overall_dist' in results_df.columns and 'composite_risk_score' in results_df.columns:
            # Risk should generally increase with distance to healthcare
            correlation = results_df['nearest_overall_dist'].corr(results_df['composite_risk_score'])
            validation_results['geographic_consistency'] = correlation > 0.1
        
        # Validate against known high-risk areas
        if known_high_risk_areas and 'school_lat' in results_df.columns and 'school_lon' in results_df.columns:
            high_risk_validation = []
            
            for known_lat, known_lon in known_high_risk_areas:
                # Find schools near known high-risk area
                distances = np.sqrt(
                    (results_df['school_lat'] - known_lat)**2 + 
                    (results_df['school_lon'] - known_lon)**2
                )
                
                nearby_schools = results_df[distances < 0.05]  # Within ~5km
                if len(nearby_schools) > 0:
                    avg_risk = nearby_schools['composite_risk_score'].mean()
                    high_risk_validation.append(avg_risk > 0.6)  # Should be high risk
            
            validation_results['known_area_validation'] = {
                'areas_checked': len(known_high_risk_areas),
                'correctly_identified': sum(high_risk_validation),
                'accuracy': sum(high_risk_validation) / len(high_risk_validation) if high_risk_validation else 0
            }
        
        return validation_results

# ================================
# EXAMPLE USAGE AND TESTING
# ================================

def example_usage():
    """Example usage of configuration and utility functions."""
    
    print("🔧 Configuration and Utilities Example")
    print("=" * 50)
    
    # 1. Get country configuration
    print("\n1. Country Configuration:")
    config = ConfigurationManager.get_country_config("tajikistan")
    print(f"Risk weights: {config['risk_weights']}")
    
    # 2. Validate sample data (if available)
    print("\n2. Data Validation Example:")
    # This would typically use real data files
    sample_healthcare = pd.DataFrame({
        'X': [68.78, 69.01, 68.76],
        'Y': [38.54, 39.91, 38.58],
        'amenity': ['hospital', 'clinic', 'pharmacy'],
        'name': ['Hospital 1', 'Clinic A', 'Pharmacy X']
    })
    
    validator = DataValidator()
    healthcare_validation = validator.validate_healthcare_data(sample_healthcare)
    print(f"Healthcare data quality score: {healthcare_validation['quality_score']:.2f}")
    
    # 3. Configuration for different scenarios
    print("\n3. Scenario Configurations:")
    emergency_config = ConfigurationManager.get_scenario_config("emergency_response")
    print(f"Emergency response weights: {emergency_config['risk_weights']}")
    
    print("\n✅ Configuration and utilities example complete!")

if __name__ == "__main__":
    example_usage()