import pandas as pd
import geopandas as gpd
import numpy as np
import fiona
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings("ignore")

def project_gdf(gdf):
    """
    Project the input GeoDataFrame into a local UTM coordinate system
    based on the representative longitude of its geometries.
    """
    mean_longitude = gdf["geometry"].representative_point().x.mean()
    utm_zone = int(np.floor((mean_longitude + 180) / 6) + 1)
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    return gdf.to_crs(utm_crs)

def fill_and_expand(gdf):
    """
    Expand MultiLineString and MultiPolygon geometries into Polygons
    so overlay operations can be performed uniformly.
    """
    gdf2 = gdf.copy()

    for i, geom in enumerate(gdf.geometry):
        # Remove simple line strings
        if geom.geom_type == 'LineString':
            gdf2 = gdf2.drop(i)

        # Attempt to convert MultiLineString into a polygon
        elif geom.geom_type == 'MultiLineString':
            linestring = gdf.loc[[i], :]
            linestring_exploded = linestring.explode(index_parts=True)

            list_of_minx = list(linestring_exploded.bounds.minx)
            min_ind = list_of_minx.index(min(list_of_minx))

            holes = []
            for k, subgeom in enumerate(linestring_exploded.geometry):
                if (k != min_ind) and (subgeom.geom_type != 'LineString'):
                    holes.append(list(subgeom.coords))

            try:
                polygon_geom = Polygon(
                    list(linestring_exploded.geometry.iloc[min_ind].coords),
                    holes=holes
                )
                polygon = gpd.GeoDataFrame(
                    index=[0],
                    data=[linestring],
                    crs=linestring.crs,
                    geometry=[polygon_geom]
                )
                gdf2 = gpd.GeoDataFrame(
                    pd.concat([gdf2, polygon], ignore_index=True),
                    crs=gdf2.crs
                )
            except ValueError:
                # Could not form polygon
                gdf2 = gdf2.drop(i)

    # Remove any remaining MultiLineStrings
    for i, geom in enumerate(gdf2.geometry):
        if geom.geom_type == 'MultiLineString':
            gdf2 = gdf2.drop(i)

    # Explode MultiPolygons
    gdf2 = gdf2.explode(index_parts=True)
    return gdf2


def preprocess_building_geometry(gdf_buildings, minimum_area=20):
    """
    Convert all geometry types to Polygons, remove invalid ones,
    project to local coordinates, and filter out footprints below
    the specified minimum area (in m²).
    """
    # 1) Expand MultiPolygons, remove line strings
    building_polygons = fill_and_expand(gdf_buildings)

    # 2) Remove invalid polygons
    building_polygons = building_polygons[building_polygons.geometry.is_valid]

    # 3) Count total
    total_buildings = len(building_polygons)
    print(f"Total number of buildings in dataset: {total_buildings}.")

    # 4) Project locally
    building_proj = project_gdf(building_polygons)

    # 5) Compute building footprint area
    building_proj['area'] = building_proj.geometry.area

    # 6) Filter out footprints below minimum_area
    filtered_gdf = building_proj[building_proj['area'] >= minimum_area]
    removed_count = total_buildings - len(filtered_gdf)
    print(f"Removed {removed_count} buildings with area < {minimum_area} sqm.")
    print(f"Remaining buildings: {len(filtered_gdf)}.")

    filtered_gdf.index = range(len(filtered_gdf))
    return filtered_gdf


def get_level(row):
    """
    Helper to parse 'building_floor' from strings that might contain 
    plus signs, minus signs, or comma-separated lists.
    """
    if pd.isna(row):
        return np.nan
    try:
        if isinstance(row, (int, float)):
            return float(row)
        elif '+' in row:
            return eval(row)
        elif '-' in row and row.strip('-').isdigit():
            # Negative numbers like '-1', '-2'
            return float(row)
        elif '-' in row:
            return float(row.split('-')[-1])
        elif ',' in row:
            return max([float(i) for i in row.split(',')])
        else:
            return float(row)
    except (ValueError, TypeError):
        print(f"Unexpected value for building_floor: {row}")
        return np.nan


def preprocess_osm_building_attributes(gdf_buildings):
    """
    Parse building_floor, remove negative/zero floors, keep 
    floors > 0 or NaN.
    """
    gdf_copy = gdf_buildings.copy()
    gdf_copy['building_floor'] = gdf_copy['building_floor'].apply(get_level)

    # Keep positive levels or NaNs
    gdf_copy = gdf_copy[
        (gdf_copy['building_floor'] > 0) |
        (gdf_copy['building_floor'].isna())
    ]
    removed_count = len(gdf_buildings) - len(gdf_copy)
    print(f"Removed {removed_count} buildings with invalid 'building_floor'.")
    return gdf_copy


def preprocess_external_building_attributes(gdf_buildings):
    """
    For external data, remove any underground buildings if flagged.
    Assumes a column named 'is_underground' that is True/False.
    """
    gdf_copy = gdf_buildings.copy()
    if 'is_underground' in gdf_copy.columns:
        # Filter out is_underground == True
        gdf_copy = gdf_copy[gdf_copy['is_underground'] == False]
        removed_count = len(gdf_buildings) - len(gdf_copy)
        print(f"Removed {removed_count} underground buildings (external).")
    return gdf_copy


def merge_external_to_osm_footprints(osm_buildings, external_buildings):
    """
    Merge external footprints into OSM footprints,
    supplementing OSM data where external data has more complete attributes.
    """
    
    required_columns = [
        'building_id', 'building_type', 'building_age',
        'building_floor', 'facade_material', 'building_area', 'area', 'geometry'
    ]

    # Check for additional columns not in required_columns
    osm_additional_columns = set(osm_buildings.columns) - set(required_columns)
    external_additional_columns = set(external_buildings.columns) - set(required_columns)

    if osm_additional_columns:
        print(f"Error: OSM buildings have additional columns not in required_columns: {osm_additional_columns}")
    if external_additional_columns:
        print(f"Error: External buildings have additional columns not in required_columns: {external_additional_columns}")
    
    for col in required_columns:
        if col not in osm_buildings.columns:
            osm_buildings[col] = np.nan
        if col not in external_buildings.columns:
            external_buildings[col] = np.nan
    
    # Convert 'yes' to NaN for building column
    osm_buildings.loc[osm_buildings['building_type'] == 'yes', 'building_type'] = np.nan

    # Compute difference
    buildings_diff_external = external_buildings.overlay(osm_buildings, how='difference')
    buildings_diff_external['external_difference_area'] = buildings_diff_external.geometry.area
    buildings_diff_external['external_proportion_not_covered'] = round(
        buildings_diff_external['external_difference_area'] / buildings_diff_external['area'] * 100, 3
    )

    # External-only (100% not covered by OSM)
    external_only_buildings = buildings_diff_external.copy()
    external_only_buildings = external_only_buildings[
        external_only_buildings['external_proportion_not_covered'] == 100
    ]

    # Overlap
    overlapping_buildings = osm_buildings.overlay(external_buildings, how='intersection')
    overlapping_buildings['overlap_area'] = overlapping_buildings.geometry.area
    overlapping_buildings['overlap_proportion'] = round(
        overlapping_buildings['overlap_area'] / overlapping_buildings['geometry'].area * 100, 3
    )

    # Identify footprints with high overlap
    supplement_buildings = overlapping_buildings[overlapping_buildings['overlap_proportion'] >= 90]

    # Merge attribute data
    supplemented_count = 0
    for index, row in supplement_buildings.iterrows():
        osm_index = osm_buildings[osm_buildings['building_id'] == row['building_id_1']].index
        if not osm_index.empty:
            # building_floor
            if pd.isna(osm_buildings.loc[osm_index, 'building_floor'].values[0]) and pd.notna(row['building_floor_2']):
                osm_buildings.loc[osm_index, 'building_floor'] = row['building_floor_2']
                supplemented_count += 1

            # building
            if pd.isna(osm_buildings.loc[osm_index, 'building_type'].values[0]) and pd.notna(row['building_type_2']):
                osm_buildings.loc[osm_index, 'building_type'] = row['building_type_2']
                supplemented_count += 1

            # building:material
            if pd.isna(osm_buildings.loc[osm_index, 'facade_material'].values[0]) and pd.notna(row['facade_material_2']):
                osm_buildings.loc[osm_index, 'facade_material'] = row['facade_material_2']
                supplemented_count += 1

    print(f"{len(supplement_buildings)} buildings have >=90% intersection.")
    print(f"{supplemented_count} attributes from external data supplemented OSM data.")

    # Tag data source
    osm_buildings['source'] = 'osm'
    external_only_buildings['source'] = 'external'

    # Ensure both GeoDataFrames have the required columns
    select_columns = [
        'building_id', 'source', 'building_type', 'building_age',
        'building_floor', 'facade_material', 'area', 'geometry'
    ]
    selected_osm = osm_buildings[select_columns]
    selected_external = external_only_buildings[select_columns]
    
    merged_gdf = gpd.GeoDataFrame(
        pd.concat([selected_osm, selected_external], ignore_index=True)
    )
    merged_gdf = merged_gdf.to_crs('epsg:4326')
    return merged_gdf


def harmonize_buildings(gdf_osm, gdf_external, filter_by_area=20, filter_underground=True):
    """
    Main pipeline function to harmonize OSM and external building footprints.

    Usage:
        ham_gdf = harmonize_buildings(gdf_osm, gdf_external, filter_by_area=20, filter_underground=True)

    Args:
        gdf_osm (GeoDataFrame): Raw OSM buildings data.
        gdf_external (GeoDataFrame): Raw external buildings data (e.g., from government or other sources).
        filter_by_area (int): Minimum area (m²) threshold to retain a building.
        filter_underground (bool): Whether to remove buildings marked as 'is_underground' == True in external data.

    Returns:
        gpd.GeoDataFrame: Harmonized building footprints combining OSM + external data.
    """
    # 1) Preprocess geometry
    print("Preprocessing OSM building geometry...")
    gdf_osm_pre = preprocess_building_geometry(gdf_osm, minimum_area=filter_by_area)

    print("Preprocessing external building geometry...")
    gdf_external_pre = preprocess_building_geometry(gdf_external, minimum_area=filter_by_area)

    # 2) Preprocess attributes
    print("Preprocessing OSM building attributes...")
    gdf_osm_attr = preprocess_osm_building_attributes(gdf_osm_pre)

    if filter_underground:
        gdf_external_attr = preprocess_external_building_attributes(gdf_external_pre)
    else:
        gdf_external_attr = gdf_external_pre

    # 3) Merge footprints
    ham_gdf = merge_external_to_osm_footprints(gdf_osm_attr, gdf_external_attr)

    print("Harmonization complete!")
    return ham_gdf
