"""
High-Resolution Spatial Grid Module

Implements:
1. H3 Hexagonal Grid over India
2. Inverse Distance Weighting (IDW) interpolation
<<<<<<< Updated upstream
3. Smooth heatmap generation

This ensures hotspots "go over the area" naturally rather than
just placing a single point on district centers.
=======
3. Diverging color scale for Gi* Z-scores
>>>>>>> Stashed changes
"""

import numpy as np
import pandas as pd
<<<<<<< Updated upstream
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

# Try importing H3 for hexagonal grids
=======
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

>>>>>>> Stashed changes
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
<<<<<<< Updated upstream
    print("Warning: h3 not installed. Using fallback grid generation.")


# India bounding box (approximate)
INDIA_BOUNDS = {
    'min_lat': 6.5,
    'max_lat': 35.5,
    'min_lon': 68.0,
    'max_lon': 97.5
}

# Default H3 resolution (6 = ~36km hexagons, 5 = ~100km, 4 = ~300km)
=======
    print("Warning: h3 not installed. Using fallback grid.")


INDIA_BOUNDS = {
    'min_lat': 6.5, 'max_lat': 35.5,
    'min_lon': 68.0, 'max_lon': 97.5
}

>>>>>>> Stashed changes
DEFAULT_H3_RESOLUTION = 4


@dataclass
class HexCell:
<<<<<<< Updated upstream
    """Hexagonal grid cell"""
=======
>>>>>>> Stashed changes
    h3_index: str
    center_lat: float
    center_lon: float
    intensity: float
    z_score: float
    classification: str
    boundary: List[Tuple[float, float]]


class HexGridGenerator:
<<<<<<< Updated upstream
    """
    Generates H3 hexagonal grid over India
    
    Uses Uber's H3 hierarchical hexagonal grid system for
    uniform area coverage and smooth interpolation.
    """
    
    def __init__(self, resolution: int = DEFAULT_H3_RESOLUTION):
        """
        Initialize grid generator
        
        Args:
            resolution: H3 resolution (0-15, lower = larger hexagons)
        """
        self.resolution = resolution
        self.hex_cells = {}
        
    def generate_india_grid(self) -> List[str]:
        """
        Generate H3 hexagon indices covering India
        
        Returns:
            List of H3 index strings
        """
        if not H3_AVAILABLE:
            return self._fallback_grid()
        
        # Create polygon covering India
        india_polygon = [
            (INDIA_BOUNDS['min_lat'], INDIA_BOUNDS['min_lon']),
            (INDIA_BOUNDS['min_lat'], INDIA_BOUNDS['max_lon']),
            (INDIA_BOUNDS['max_lat'], INDIA_BOUNDS['max_lon']),
            (INDIA_BOUNDS['max_lat'], INDIA_BOUNDS['min_lon']),
        ]
        
        # Convert to GeoJSON format
        geojson_polygon = {
            'type': 'Polygon',
            'coordinates': [[
                [lon, lat] for lat, lon in india_polygon + [india_polygon[0]]
            ]]
        }
        
        try:
            # Get all hexagons covering the polygon
            hexagons = h3.polygon_to_cells(
                h3.LatLngPoly(india_polygon),
                self.resolution
            )
            return list(hexagons)
        except Exception as e:
            print(f"H3 polygon fill error: {e}")
            return self._generate_grid_points()
    
    def _generate_grid_points(self) -> List[str]:
        """Generate grid by sampling points within bounds"""
        hexagons = set()
        
        # Sample points across India
        lat_step = 0.5 if self.resolution >= 5 else 1.0
        lon_step = 0.5 if self.resolution >= 5 else 1.0
=======
    def __init__(self, resolution: int = DEFAULT_H3_RESOLUTION):
        self.resolution = resolution
        
    def generate_india_grid(self) -> List[str]:
        if not H3_AVAILABLE:
            return []
        
        hexagons = set()
        lat_step = 1.0 if self.resolution >= 5 else 2.0
        lon_step = 1.0 if self.resolution >= 5 else 2.0
>>>>>>> Stashed changes
        
        lat = INDIA_BOUNDS['min_lat']
        while lat <= INDIA_BOUNDS['max_lat']:
            lon = INDIA_BOUNDS['min_lon']
            while lon <= INDIA_BOUNDS['max_lon']:
                try:
                    h3_index = h3.latlng_to_cell(lat, lon, self.resolution)
                    hexagons.add(h3_index)
                except:
                    pass
                lon += lon_step
            lat += lat_step
        
        return list(hexagons)
    
<<<<<<< Updated upstream
    def _fallback_grid(self) -> List[str]:
        """Fallback grid when H3 not available"""
        # Return empty - will use district-level data instead
        return []
    
    def get_hex_center(self, h3_index: str) -> Tuple[float, float]:
        """Get center coordinates of a hexagon"""
        if H3_AVAILABLE:
            lat, lon = h3.cell_to_latlng(h3_index)
            return (lat, lon)
        return (0, 0)
    
    def get_hex_boundary(self, h3_index: str) -> List[Tuple[float, float]]:
        """Get boundary coordinates of a hexagon"""
        if H3_AVAILABLE:
            boundary = h3.cell_to_boundary(h3_index)
            return [(lat, lon) for lat, lon in boundary]
        return []
    
    def to_geojson(self, hexagons: List[HexCell]) -> Dict:
        """
        Convert hexagon cells to GeoJSON for Leaflet rendering
        
        Returns GeoJSON FeatureCollection with hexagon polygons
        """
        features = []
        
        for hex_cell in hexagons:
            # Create polygon from boundary
            if hex_cell.boundary:
                coordinates = [[
                    [lon, lat] for lat, lon in hex_cell.boundary
                ]]
                # Close the polygon
=======
    def get_hex_center(self, h3_index: str) -> Tuple[float, float]:
        if H3_AVAILABLE:
            return h3.cell_to_latlng(h3_index)
        return (0, 0)
    
    def get_hex_boundary(self, h3_index: str) -> List[Tuple[float, float]]:
        if H3_AVAILABLE:
            return list(h3.cell_to_boundary(h3_index))
        return []
    
    def to_geojson(self, hexagons: List[HexCell]) -> Dict:
        features = []
        for hex_cell in hexagons:
            if hex_cell.boundary:
                coordinates = [[[lon, lat] for lat, lon in hex_cell.boundary]]
>>>>>>> Stashed changes
                if coordinates[0][0] != coordinates[0][-1]:
                    coordinates[0].append(coordinates[0][0])
                
                features.append({
                    'type': 'Feature',
                    'properties': {
                        'h3_index': hex_cell.h3_index,
                        'intensity': hex_cell.intensity,
                        'z_score': hex_cell.z_score,
                        'classification': hex_cell.classification,
<<<<<<< Updated upstream
                        'center_lat': hex_cell.center_lat,
                        'center_lon': hex_cell.center_lon
                    },
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': coordinates
                    }
                })
        
        return {
            'type': 'FeatureCollection',
            'features': features
        }


class IDWInterpolator:
    """
    Inverse Distance Weighting (IDW) Interpolation
    
    Interpolates intensity values from district points to grid cells
    for smooth heatmap visualization.
    
    Formula: Z(x) = Σ(wi * Zi) / Σ(wi)
    Where wi = 1 / d(x, xi)^p
    """
    
    def __init__(self, power: float = 2.0, max_distance: float = 5.0):
        """
        Initialize IDW interpolator
        
        Args:
            power: Distance weighting power (2 = inverse square)
            max_distance: Maximum distance for influence (degrees)
        """
        self.power = power
        self.max_distance = max_distance
        
    def interpolate(
        self,
        target_points: List[Tuple[float, float]],
        source_points: List[Tuple[float, float]],
        source_values: List[float]
    ) -> List[float]:
        """
        Interpolate values at target points from source points
        
        Args:
            target_points: List of (lat, lon) to interpolate at
            source_points: List of (lat, lon) with known values
            source_values: Values at source points
            
        Returns:
            Interpolated values at target points
        """
        source_arr = np.array(source_points)
        values_arr = np.array(source_values)
        
=======
                        'color': get_color_hex(hex_cell.z_score)
                    },
                    'geometry': {'type': 'Polygon', 'coordinates': coordinates}
                })
        
        return {'type': 'FeatureCollection', 'features': features}


class IDWInterpolator:
    def __init__(self, power: float = 2.0, max_distance: float = 5.0):
        self.power = power
        self.max_distance = max_distance
        
    def interpolate(self, target_points, source_points, source_values):
        source_arr = np.array(source_points)
        values_arr = np.array(source_values)
>>>>>>> Stashed changes
        interpolated = []
        
        for target in target_points:
            target_arr = np.array(target)
<<<<<<< Updated upstream
            
            # Calculate distances to all source points
            distances = np.sqrt(np.sum((source_arr - target_arr) ** 2, axis=1))
            
            # Filter by max distance
            mask = distances < self.max_distance
            
            if not np.any(mask):
                # No nearby points, use mean
=======
            distances = np.sqrt(np.sum((source_arr - target_arr) ** 2, axis=1))
            mask = distances < self.max_distance
            
            if not np.any(mask):
>>>>>>> Stashed changes
                interpolated.append(np.mean(values_arr))
                continue
            
            distances_filtered = distances[mask]
            values_filtered = values_arr[mask]
            
<<<<<<< Updated upstream
            # Handle exact matches
=======
>>>>>>> Stashed changes
            if np.any(distances_filtered == 0):
                idx = np.where(distances_filtered == 0)[0][0]
                interpolated.append(values_filtered[idx])
                continue
            
<<<<<<< Updated upstream
            # Calculate weights
            weights = 1.0 / (distances_filtered ** self.power)
            
            # Weighted average
=======
            weights = 1.0 / (distances_filtered ** self.power)
>>>>>>> Stashed changes
            value = np.sum(weights * values_filtered) / np.sum(weights)
            interpolated.append(value)
        
        return interpolated
    
<<<<<<< Updated upstream
    def interpolate_grid(
        self,
        grid: HexGridGenerator,
        district_data: pd.DataFrame,
        value_column: str = 'scaled_intensity'
    ) -> List[HexCell]:
        """
        Interpolate district data onto hexagonal grid
        
        Args:
            grid: HexGridGenerator instance
            district_data: DataFrame with district points and values
            value_column: Column to interpolate
            
        Returns:
            List of HexCell objects with interpolated values
        """
        # Generate grid
        h3_indices = grid.generate_india_grid()
        
        if not h3_indices:
            # Fallback to district-level cells
            return self._district_fallback(district_data, value_column)
        
        # Get source points and values
        source_points = list(zip(district_data['lat'], district_data['lon']))
        source_values = district_data[value_column].tolist()
        
        # Get target points (hex centers)
        target_points = [grid.get_hex_center(h) for h in h3_indices]
        
        # Interpolate
        interpolated_values = self.interpolate(target_points, source_points, source_values)
        
        # Create HexCell objects
        cells = []
        for i, h3_index in enumerate(h3_indices):
            center = target_points[i]
            intensity = interpolated_values[i]
            
            # Calculate approximate z-score
            mean_val = np.mean(interpolated_values)
            std_val = np.std(interpolated_values)
            z_score = (intensity - mean_val) / (std_val + 1e-10)
            
            # Classify
            classification = self._classify_zscore(z_score)
            
=======
    def interpolate_grid(self, grid, district_data, value_column='scaled_intensity'):
        h3_indices = grid.generate_india_grid()
        
        if not h3_indices:
            return self._district_fallback(district_data, value_column)
        
        source_points = list(zip(district_data['lat'], district_data['lon']))
        source_values = district_data[value_column].tolist()
        target_points = [grid.get_hex_center(h) for h in h3_indices]
        
        interpolated_values = self.interpolate(target_points, source_points, source_values)
        
        cells = []
        mean_val = np.mean(interpolated_values)
        std_val = np.std(interpolated_values)
        
        for i, h3_index in enumerate(h3_indices):
            center = target_points[i]
            intensity = interpolated_values[i]
            z_score = (intensity - mean_val) / (std_val + 1e-10)
            
>>>>>>> Stashed changes
            cells.append(HexCell(
                h3_index=h3_index,
                center_lat=center[0],
                center_lon=center[1],
                intensity=float(intensity),
                z_score=float(z_score),
<<<<<<< Updated upstream
                classification=classification,
=======
                classification=classify_zscore(z_score),
>>>>>>> Stashed changes
                boundary=grid.get_hex_boundary(h3_index)
            ))
        
        return cells
    
<<<<<<< Updated upstream
    def _classify_zscore(self, z: float) -> str:
        """Classify based on Z-score thresholds"""
        if z > 2.58:
            return 'Significant Hotspot'  # 99% confidence
        elif z > 1.96:
            return 'Emerging Trend'       # 95% confidence
        elif z < -2.58:
            return 'Cold Spot'            # 99% confidence (exclusion zone)
        elif z < -1.96:
            return 'Declining'            # 95% confidence
        else:
            return 'In-Sync'              # Baseline
    
    def _district_fallback(
        self,
        district_data: pd.DataFrame,
        value_column: str
    ) -> List[HexCell]:
        """Fallback when H3 not available - use district polygons"""
        cells = []
        
=======
    def _district_fallback(self, district_data, value_column):
        cells = []
>>>>>>> Stashed changes
        mean_val = district_data[value_column].mean()
        std_val = district_data[value_column].std()
        
        for _, row in district_data.iterrows():
            intensity = row[value_column]
            z_score = (intensity - mean_val) / (std_val + 1e-10)
<<<<<<< Updated upstream
            classification = self._classify_zscore(z_score)
            
            # Create approximate hexagon around district center
            lat, lon = row['lat'], row['lon']
            offset = 0.5  # ~50km
=======
            lat, lon = row['lat'], row['lon']
            offset = 0.5
>>>>>>> Stashed changes
            
            boundary = [
                (lat + offset, lon),
                (lat + offset/2, lon + offset),
                (lat - offset/2, lon + offset),
                (lat - offset, lon),
                (lat - offset/2, lon - offset),
                (lat + offset/2, lon - offset),
            ]
            
            cells.append(HexCell(
<<<<<<< Updated upstream
                h3_index=f"{row.get('district_code', 'D0000')}",
=======
                h3_index=row.get('district_code', 'D0000'),
>>>>>>> Stashed changes
                center_lat=lat,
                center_lon=lon,
                intensity=float(intensity),
                z_score=float(z_score),
<<<<<<< Updated upstream
                classification=classification,
=======
                classification=classify_zscore(z_score),
>>>>>>> Stashed changes
                boundary=boundary
            ))
        
        return cells


<<<<<<< Updated upstream
def get_diverging_color(z_score: float, opacity: float = 0.7) -> str:
    """
    Get color from diverging palette based on Z-score
    
    Palette:
    - Crimson Red (Z > 2.58): Significant Growth Hotspot
    - Orange (1.96 < Z < 2.58): Emerging Trend
    - Soft Yellow/White (baseline): In-Sync
    - Deep Blue (Z < -2.58): Cold Spot / Digital Exclusion Zone
    
    Returns RGBA string for Leaflet
    """
    if z_score > 2.58:
        # Crimson Red - Significant hotspot (99% confidence)
        return f"rgba(220, 20, 60, {opacity})"
    elif z_score > 1.96:
        # Orange - Emerging trend
        return f"rgba(255, 140, 0, {opacity})"
    elif z_score > 0:
        # Light Orange to Yellow gradient
        t = z_score / 1.96
        r = int(255 - 50 * (1 - t))
        g = int(200 + 55 * (1 - t))
        return f"rgba({r}, {g}, 100, {opacity})"
    elif z_score > -1.96:
        # Light Blue to Yellow gradient
        t = abs(z_score) / 1.96
        r = int(255 - 100 * t)
        g = int(255 - 55 * t)
        b = int(150 + 105 * t)
        return f"rgba({r}, {g}, {b}, {opacity})"
    elif z_score > -2.58:
        # Blue - Declining
        return f"rgba(65, 105, 225, {opacity})"
    else:
        # Deep Blue - Cold spot / Exclusion zone
        return f"rgba(0, 0, 139, {opacity})"


def get_color_hex(z_score: float) -> str:
    """Get hex color code for Z-score"""
=======
def classify_zscore(z: float) -> str:
    if z > 2.58:
        return 'Significant Hotspot'
    elif z > 1.96:
        return 'Emerging Trend'
    elif z < -2.58:
        return 'Cold Spot'
    elif z < -1.96:
        return 'Declining'
    return 'In-Sync'


def get_color_hex(z_score: float) -> str:
>>>>>>> Stashed changes
    if z_score > 2.58:
        return '#DC143C'  # Crimson
    elif z_score > 1.96:
        return '#FF8C00'  # Dark Orange
    elif z_score > 0:
        return '#FFD700'  # Gold
    elif z_score > -1.96:
        return '#87CEEB'  # Sky Blue
    elif z_score > -2.58:
        return '#4169E1'  # Royal Blue
<<<<<<< Updated upstream
    else:
        return '#00008B'  # Dark Blue


def create_legend_data() -> List[Dict]:
    """Create legend data for the color scale"""
    return [
        {'label': 'Significant Hotspot (Z > 2.58)', 'color': '#DC143C', 'description': '99% confidence growth'},
        {'label': 'Emerging Trend (Z > 1.96)', 'color': '#FF8C00', 'description': '95% confidence growth'},
        {'label': 'In-Sync (Baseline)', 'color': '#FFD700', 'description': 'Normal growth rate'},
        {'label': 'Declining (Z < -1.96)', 'color': '#4169E1', 'description': 'Below average'},
        {'label': 'Cold Spot (Z < -2.58)', 'color': '#00008B', 'description': 'Digital Exclusion Zone'},
=======
    return '#00008B'      # Dark Blue


def create_legend_data() -> List[Dict]:
    return [
        {'label': 'Significant Hotspot', 'color': '#DC143C', 'description': 'Z > 2.58 (99% confidence)'},
        {'label': 'Emerging Trend', 'color': '#FF8C00', 'description': 'Z > 1.96 (95% confidence)'},
        {'label': 'In-Sync', 'color': '#FFD700', 'description': 'Baseline growth'},
        {'label': 'Declining', 'color': '#4169E1', 'description': 'Z < -1.96'},
        {'label': 'Cold Spot', 'color': '#00008B', 'description': 'Z < -2.58 (Exclusion Zone)'},
>>>>>>> Stashed changes
    ]
