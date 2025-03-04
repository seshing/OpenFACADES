import geopandas as gpd
import os
from shapely.geometry import Point, LineString
import pandas as pd
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import numpy as np
from tqdm import tqdm

def project_gdf(gdf):
    """Project GeoDataFrame to a local UTM coordinate system based on longitude."""
    mean_longitude = gdf["geometry"].representative_point().x.mean()
    utm_zone = int(np.floor((mean_longitude + 180) / 6) + 1)
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    return gdf.to_crs(utm_crs)

def process_point(row, gdf, buffer, distance_between_points):
    """Process a single observation point, computing its field-of-view (FoV)."""
    fov_metrics = []
    view_point = row['geometry']
    view_point_buffer = view_point.buffer(buffer)
    target_buildings = gdf[gdf.intersects(view_point_buffer)]

    def generate_points_along_geometry(gdf, distance):
        points_with_id = []
        for _, row in gdf.iterrows():
            geometry = row['geometry']
            building_id = str(row['building_id'])
            if geometry.geom_type == 'Polygon':
                exterior = geometry.exterior
            elif geometry.geom_type == 'LineString':
                exterior = geometry
            else:
                continue
            length = exterior.length
            current_distance = 0
            while current_distance <= length:
                point = exterior.interpolate(current_distance)
                points_with_id.append((point, building_id))
                current_distance += distance
        return points_with_id

    def create_view_lines(view_point, target_points_with_id):
        return [(LineString([view_point, point]), building_id) for point, building_id in target_points_with_id]

    def remove_intersecting_lines(lines, gdf):
        return [(line, building_id) for line, building_id in lines if not gdf.intersects(line).any()]

    def calculate_angle(view_point, target_point):
        delta_x = target_point.x - view_point.x
        delta_y = target_point.y - view_point.y
        angle_radians = math.atan2(delta_x, delta_y)
        return (math.degrees(angle_radians) + 360) % 360

    target_points = generate_points_along_geometry(target_buildings, distance_between_points)
    unobstructed_lines = remove_intersecting_lines(create_view_lines(view_point, target_points), target_buildings)
    
    # Compute the shortest distance for each building from the unobstructed lines
    building_distances = {}
    for line, building_id in unobstructed_lines:
        dist = line.length
        if building_id in building_distances:
            building_distances[building_id] = min(building_distances[building_id], dist)
        else:
            building_distances[building_id] = dist

    building_lines = {}
    for line, building_id in unobstructed_lines:
        angle = calculate_angle(Point(line.coords[0]), Point(line.coords[1]))
        building_lines.setdefault(building_id, []).append(angle)

    for building_id, angles in building_lines.items():
        angles.sort()
        largest_fov, angle_pair = 0, (0, 0)
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                fov = min((angles[i] - angles[j]) % 360, (angles[j] - angles[i]) % 360)
                if fov > largest_fov:
                    largest_fov = fov
                    angle_pair = (angles[i], angles[j])

        total_fov = largest_fov if largest_fov < 180 else 360 - largest_fov
        shortest_distance = building_distances[building_id]

        fov_metrics.append({
            "pid": row['pid'],
            "lat": row['lat'],
            "lng": row['lng'],
            "compass_angle": row['compass_angle'],
            "building_id": building_id,
            "left_angle_geo": min(angle_pair) if max(angle_pair) - min(angle_pair) < 180 else max(angle_pair),
            "right_angle_geo": max(angle_pair) if max(angle_pair) - min(angle_pair) < 180 else min(angle_pair),
            "aov_geo": total_fov,
            "distance": shortest_distance
        })

    return fov_metrics

def process_batch(batch, gdf, buffer, distance_between_points):
    """Process a batch of observation points."""
    results = []
    for _, row in batch.iterrows():
        results.extend(process_point(row, gdf, buffer, distance_between_points))
    return results

def geo_aov_calculator(img_metadata, json_dir, buffer, distance_between_points, out_dir, batch_size=10, max_workers=8):
    """
    Calculate the geometric field of view (FoV) for observation points relative to buildings.
    Now with multiprocessing support and progress reporting using tqdm.
    """
    
    # Load building footprint data and determine city CRS
    gdf_buildings = gpd.read_file(json_dir)
    gdf_buildings = project_gdf(gdf_buildings)  # Determine CRS automatically
    gdf_buildings['building_id'] = gdf_buildings['building_id'].astype(str)
    city_crs = gdf_buildings.crs  # Extract the determined CRS

    # Convert observation points to GeoDataFrame and project to the same CRS
    df_points = pd.read_csv(img_metadata)
    df_points.rename(columns={'id': 'pid', 'computed_compass_angle': 'compass_angle'}, inplace=True)
    gdf_points = gpd.GeoDataFrame(df_points, geometry=gpd.points_from_xy(df_points.lng, df_points.lat), crs="EPSG:4326")
    gdf_points = gdf_points.to_crs(city_crs)

    # Check for existing output and skip processed building IDs
    processed_pairs = set()
    if os.path.exists(out_dir) and os.path.getsize(out_dir) > 0:
        try:
            existing_data = pd.read_csv(out_dir)
            processed_pairs = set(zip(existing_data['pid'], existing_data['building_id']))
            print(f"Found {len(processed_pairs)} existing point-building pairs to skip.")
        except Exception as e:
            print(f"Error reading existing output file: {e}")
            print("Will process all point-building pairs.")

    total_points = len(gdf_points)

    # Open CSV for writing results
    with open(out_dir, 'a', newline='') as f:
        fieldnames = ["pid", "lat", "lng", "compass_angle", "building_id", "left_angle_geo", "right_angle_geo", "aov_geo", "distance"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if file is empty
        f.seek(0, 2)
        if f.tell() == 0:
            writer.writeheader()

        # Process in parallel with a progress bar
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            batches = [gdf_points.iloc[i:i + batch_size] for i in range(0, total_points, batch_size)]
            future_to_batch = {executor.submit(process_batch, batch, gdf_buildings, buffer, distance_between_points): batch for batch in batches}

            for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Processing batches"):
                for metric in future.result():
                    # Skip if this point-building pair has already been processed
                    if (metric['pid'], metric['building_id']) not in processed_pairs:
                        writer.writerow(metric)

    print("Geometric FoV calculation completed.")
    