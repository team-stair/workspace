# Tajikistan Schoolchildren Hospital Access Risk Mapping

## Overview

This project creates comprehensive risk maps for schoolchildren in Tajikistan, where **risk represents the difficulty of getting transported to a hospital in times of need**. The system uses geospatial analysis to combine multiple risk factors into a composite risk score.

## Risk Factors Considered

The composite risk scoring system evaluates:

1. **Distance to Nearest Hospital (35% weight)** - Primary factor
2. **Road Network Accessibility (25% weight)** - Infrastructure quality
3. **Population Density (20% weight)** - Competition for healthcare resources
4. **Terrain Difficulty (15% weight)** - Geographic barriers (mountains, elevation)
5. **Facility Capacity (5% weight)** - Local resource availability

## Your Data Structure

Based on your project directory, you have:

```
/Users/alice/Developer/UN-Tech-Over2/
├── hotosm_tjk_education_facilities_points_geojson/
├── hotosm_tjk_education_facilities_points_gpkg/
├── hotosm_tjk_education_facilities_points_shp/
├── hotosm_tjk_education_facilities_polygons_geojson/
├── hotosm_tjk_education_facilities_polygons_gpkg/
├── hotosm_tjk_education_facilities_polygons_shp/
├── tajikistan.geojson
├── tajikistan.csv
├── tajikistan_hxl.geojson
├── tajikistan_hxl.csv
└── challenge2_sample_notebook.ipynb
```

## Quick Start (Option 1)

### Step 1: Run the Quick Start Script

```python
# Save the quick_start_script.py and run:
python quick_start_script.py
```

This will:
- Check your environment setup
- Explore your available data files
- Create a simple map with whatever data is accessible
- Provide next steps guidance

### Step 2: Run the Full Risk Analysis

```python
# Save the tjk_risk_mapper.py and run:
python tjk_risk_mapper.py
```

This will:
- Load your education facilities data
- Calculate all risk components
- Create comprehensive risk maps
- Export results to CSV

## Manual Setup (Option 2)

### Step 1: Install Required Packages

```bash
pip install pandas geopandas matplotlib numpy scipy scikit-learn seaborn
```

### Step 2: Basic Usage Example

```python
from tjk_risk_mapper import TajikistanRiskMapper

# Initialize the mapper
mapper = TajikistanRiskMapper("/Users/alice/Developer/UN-Tech-Over2")

# Load your education facilities
schools = mapper.load_education_facilities()
print(f"Loaded {len(schools)} education facilities")

# Calculate composite risk
risk_scores = mapper.calculate_composite_risk()

# Create maps
mapper.create_risk_maps(output_dir="my_risk_maps")

# Export results
mapper.export_risk_data("school_risk_analysis.csv")
```

## Understanding the Output

### Maps Created

1. **`composite_risk_map.png`** - Main risk visualization
   - Red points = High risk schools
   - Yellow points = Low risk schools
   - Blue crosses = Hospitals

2. **`component_risk_maps.png`** - Individual risk factors
   - Shows each risk component separately
   - Helps identify which factors drive overall risk

3. **`hospital_access_map.png`** - Healthcare accessibility
   - Lines connect high-risk schools to nearest hospitals
   - Visualizes transportation challenges

4. **`risk_statistics_summary.png`** - Statistical analysis
   - Risk distribution histograms
   - Correlation analysis between factors
   - Summary statistics

### Data Export

The CSV export includes:
- School coordinates and properties
- Individual risk scores for each factor
- Composite risk score
- Risk category classification

## Customizing the Analysis

### Adjusting Risk Weights

```python
# Modify risk factor importance
mapper.risk_weights = {
    'distance_to_hospital': 0.40,    # Increase hospital distance weight
    'road_accessibility': 0.30,     # Increase road importance
    'population_density': 0.15,     # Decrease population weight
    'terrain_difficulty': 0.10,     # Decrease terrain weight
    'facility_capacity': 0.05       # Keep facility weight same
}
```

### Adding Real Hospital Data

```python
# Load your hospital/healthcare facility data
hospitals = gpd.read_file("path/to/hospital_data.geojson")
mapper.hospitals = hospitals

# Recalculate with real hospital locations
risk_scores = mapper.calculate_composite_risk()
```

### Using Real Cities Data for Population Analysis

The system includes default major cities, but you can get real data:

**Option 1: Use the provided cities data collector**
```python
# Run the cities data collection script
python get_cities_data.py

# Then use in your analysis
mapper = TajikistanRiskMapper()
mapper.load_custom_cities_data(cities_file='tajikistan_cities.csv')
```

**Option 2: Provide your own cities data**
```python
# Custom cities (longitude, latitude, name, population)
custom_cities = [
    (68.78, 38.54, 'Dushanbe', 863400),
    (69.62, 40.29, 'Khujand', 181600),
    (69.78, 37.91, 'Kulob', 106400),
    # Add more cities...
]

mapper = TajikistanRiskMapper()
mapper.load_custom_cities_data(cities_data=custom_cities)
```

**Option 3: Create a CSV file**
```csv
name,longitude,latitude,population
Dushanbe,68.78,38.54,863400
Khujand,69.62,40.29,181600
Kulob,69.78,37.91,106400
Kurgan-Tyube,68.78,37.83,85000
Khorugh,71.55,37.49,30000
```

### Using Real Elevation Data

```python
# If you have DEM (Digital Elevation Model) data
import rasterio

def calculate_terrain_risk_from_dem(self, dem_path):
    with rasterio.open(dem_path) as dem:
        # Extract elevation for each school location
        coords = [(point.x, point.y) for point in self.education_facilities.geometry]
        elevations = [x[0] for x in dem.sample(coords)]
        
        # Convert to risk scores
        scaler = MinMaxScaler()
        terrain_risk = scaler.fit_transform(np.array(elevations).reshape(-1, 1)).flatten()
        
    return terrain_risk

# Replace the simulated terrain calculation
TajikistanRiskMapper.calculate_terrain_risk = calculate_terrain_risk_from_dem
```

## Integration with Existing Workflow

### Using with giga-spatial library

```python
# Integration with your existing giga-spatial workflow
from gigaspatial.handlers import AdminBoundaries
from tjk_risk_mapper import TajikistanRiskMapper

# Load admin boundaries (from your existing notebook)
admin2_data = AdminBoundaries.create(country_code="TJK", admin_level=2).to_geodataframe()

# Initialize risk mapper
mapper = TajikistanRiskMapper()
mapper.load_education_facilities()

# Calculate risks
mapper.calculate_composite_risk()

# Aggregate risk scores by administrative region
from gigaspatial.generators import GeometryBasedZonalViewGenerator

view_gen = GeometryBasedZonalViewGenerator(
    zone_data=admin2_data,
    zone_id_column="id",
    zone_data_crs=admin2_data.crs
)

# Map risk scores to administrative zones
risk_by_admin = view_gen.map_points(
    points=mapper.education_facilities,
    value_columns=['composite_risk'],
    aggregation='mean'
)
```

## Troubleshooting

### Common Issues

1. **File Access Problems**
   ```python
   # If you can't read the geojson files directly:
   import json
   with open('tajikistan.geojson', 'r') as f:
       data = json.load(f)
   # Then manually create GeoDataFrame
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Process data in chunks
   mapper = TajikistanRiskMapper()
   schools = mapper.load_education_facilities()
   
   # Sample subset for testing
   schools_sample = schools.sample(n=100)
   mapper.education_facilities = schools_sample
   ```

3. **Coordinate System Issues**
   ```python
   # Ensure consistent CRS
   schools = schools.to_crs('EPSG:4326')  # WGS84
   ```

### Performance Tips

1. **For Large Datasets**: Process administrative regions separately
2. **For Speed**: Use sample data during development
3. **For Accuracy**: Integrate real healthcare facility locations
4. **For Detail**: Add road network and elevation data

## Next Steps

1. **Validate Results**: Compare with known healthcare access challenges
2. **Add Real Data**: Integrate actual hospital locations and road networks
3. **Policy Applications**: Use results to prioritize healthcare infrastructure
4. **Monitoring**: Update analysis as new facilities are built
5. **Integration**: Connect with health outcome data for validation

## Advanced Features

### Time-based Analysis
```python
# Analyze risk changes over time
def temporal_risk_analysis(years=[2020, 2021, 2022, 2023]):
    for year in years:
        # Load year-specific data
        # Calculate risks
        # Compare changes
        pass
```

### Scenario Analysis
```python
# What-if scenarios for new hospitals
def scenario_analysis(proposed_hospital_locations):
    original_risk = mapper.calculate_composite_risk()
    
    # Add proposed hospitals
    mapper.hospitals = pd.concat([mapper.hospitals, proposed_hospitals])
    
    # Recalculate risk
    new_risk = mapper.calculate_composite_risk()
    
    # Analyze improvement
    improvement = original_risk - new_risk
    return improvement
```

## Contributing

To extend this analysis:

1. Add new risk factors to the `calculate_*_risk()` methods
2. Modify visualization functions for different map styles
3. Integrate additional data sources (weather, economic, etc.)
4. Add machine learning models for risk prediction

## Support

For questions about:
- **Geospatial analysis**: Refer to geopandas and matplotlib documentation
- **Risk methodology**: Review WHO healthcare access guidelines
- **Tajikistan-specific data**: Check with local healthcare authorities
- **Technical implementation**: Review the code comments and examples