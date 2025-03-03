import osmnx as ox
from overturemaps import core
import warnings
warnings.filterwarnings("ignore")

col_names = {
            'osmid': 'building_id',
            'id': 'building_id',
            'building': 'building_type',
            'start_date': 'building_age',
            'building:levels': 'building_floor',
            'building:material': 'facade_material',
            'class': 'building_type', 
            'age' : 'building_age',
            'num_floors': 'building_floor'
        }

class BuildingDataDownloader:
    def __init__(self, source="osm"):
        """
        Initialize the downloader with a data source.
        :param source: 'osm' for OpenStreetMap, 'overture' for Overture Maps.
        """
        if source not in ["osm", "overture"]:
            raise ValueError("Source must be either 'osm' or 'overture'.")
        self.source = source

    def _is_bbox(self, input_value):
        return (
            isinstance(input_value, (list, tuple))
            and len(input_value) == 4
            and all(isinstance(coord, (int, float)) for coord in input_value)
        )

    def _is_city_name(self, input_value):
        return isinstance(input_value, str)

    def _get_city_boundary(self, city_name):
        return ox.geocode_to_gdf(city_name)

    def _download_osm_building(self, city_name):
        """Download OpenStreetMap building data for a city."""
        tags = {"building": True}
        print("Retrieving data from OpenStreetMap...")
        gdf = ox.features_from_place(city_name, tags)

        # Keep only valid polygon geometries
        gdf = gdf[gdf.geometry.apply(lambda x: x.geom_type in ['Polygon', 'MultiPolygon'])]
        gdf.reset_index(drop=False, inplace=True)

        # Select relevant columns
        selected_columns = ['osmid', 'building', 'start_date', 'building:levels', 'building:material', 'geometry']
        gdf_clean = gdf[[col for col in selected_columns if col in gdf.columns]].copy()
        gdf_clean = gdf_clean.rename(columns=col_names)
        gdf_clean = gdf_clean.set_crs("EPSG:4326")
        
        print(f"{len(gdf_clean)} buildings retrieved from OpenStreetMap...")

        return gdf_clean

    def _download_osm_building_box(self, bbox):
        """Download OpenStreetMap building data within a bounding box."""
        tags = {"building": True}
        print("Retrieving data from OpenStreetMap...")
        
        # For versions of OSMnx >= 2.0.0
        # gdf = ox.features_from_bbox(tuple(bbox), tags) 
        
        # For versions of OSMnx < 2.0.0
        bbox = [bbox[3], bbox[1], bbox[2], bbox[0]]
        gdf = ox.features_from_bbox(bbox=bbox, tags = tags) # N, S, E, W

        # Keep only valid polygon geometries
        gdf = gdf[gdf.geometry.apply(lambda x: x.geom_type in ['Polygon', 'MultiPolygon'])]
        gdf.reset_index(drop=False, inplace=True)

        # Select relevant columns
        selected_columns = ['osmid', 'building', 'start_date', 'building:levels', 'building:material', 'geometry']
        gdf_clean = gdf[[col for col in selected_columns if col in gdf.columns]].copy()
        gdf_clean = gdf_clean.rename(columns=col_names)
        gdf_clean = gdf_clean.set_crs("EPSG:4326")
        
        print(f"{len(gdf_clean)} buildings retrieved from OpenStreetMap...")

        return gdf_clean

    def _download_overture_building(self, city_name):
        """Download building data from Overture Maps for a given city."""
        boundary_gdf = self._get_city_boundary(city_name)
        bbox = boundary_gdf.total_bounds.tolist()
        print("Retrieving data from Overture...")
        gdf = core.geodataframe("building", bbox=bbox)
        gdf = gdf[gdf.geometry.apply(lambda x: x.intersects(boundary_gdf.unary_union))]
        
        selected_columns = ['id', 'class', 'age', 'num_floors', 
                            'facade_material', 'geometry']
        gdf_clean = gdf[[col for col in selected_columns if col in gdf.columns]].copy()
        gdf_clean = gdf_clean.rename(columns=col_names)
        gdf_clean = gdf_clean.set_crs("EPSG:4326")

        print(f"{len(gdf_clean)} buildings retrived from Overture...")
        return gdf_clean

    def _download_overture_building_box(self, bbox):
        """Download building data from Overture Maps within a bounding box."""
        print("Retrieving data from Overture...")
        gdf = core.geodataframe("building", bbox=bbox)
        selected_columns = ['id', 'class', 'age', 'num_floors', 
                            'facade_material', 'geometry']
        gdf_clean = gdf[[col for col in selected_columns if col in gdf.columns]].copy()
        gdf_clean = gdf_clean.rename(columns=col_names)
        gdf_clean = gdf_clean.set_crs("EPSG:4326")
        print(f"{len(gdf_clean)} buildings retrived from Overture...")
        
        return gdf_clean

    def download_buildings(self, input_value):
        if self._is_city_name(input_value):
            if self.source == "osm":
                return self._download_osm_building(input_value)
            elif self.source == "overture":
                return self._download_overture_building(input_value)

        elif self._is_bbox(input_value):
            if self.source == "osm":
                return self._download_osm_building_box(input_value)
            elif self.source == "overture":
                return self._download_overture_building_box(input_value)

        else:
            raise ValueError("Invalid input: Provide either a city name (string) or a bounding box (list of four numerical values).")