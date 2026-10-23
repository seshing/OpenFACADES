"""Retrieve different views of one OSM building using the existing panorama pipeline."""
import argparse
import json
import os
from pathlib import Path
import re
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd

from run import (download_and_merge_buildings, prepare_mly_metadata,
                 download_selected_images, run_object_detection,
                 select_suitable_building_image)
from openfacades.streetview import geo_aov_calculator


RESULT_COLUMNS = ['image_name', 'pid', 'building_id', 'osm_id', 'lat', 'lng',
                  'view_angle', 'visible_sides', 'frontality', 'aov_geo', 'distance']


def normalize_id(building_id):
    value = str(building_id).strip().lower()
    if value.isascii() and value.isdigit():
        value = 'way/' + value
    match = re.fullmatch(r'(way|relation)/([1-9][0-9]*)', value)
    if not match:
        raise ValueError('Use a positive OSM ID, way/123 or relation/123.')
    return value


def get_target(building_id, data_dir):
    """Fetch a complete OSM object and let OSMnx assemble its polygon geometry."""
    import requests
    import osmnx as ox
    kind, number = building_id.split('/')
    response = requests.get(
        f'https://api.openstreetmap.org/api/0.6/{kind}/{number}/full',
        headers={'User-Agent': 'OpenFACADES building search'}, timeout=60)
    response.raise_for_status()
    xml_path = Path(data_dir) / 'target.osm'
    xml_path.write_bytes(response.content)
    buildings = ox.features_from_xml(xml_path, tags={'building': True})
    matches = buildings.loc[buildings.index.isin([(kind, int(number))])]
    if matches.empty or str(matches.iloc[0].get('building', '')).lower() in ('', 'no', 'nan'):
        raise ValueError('OSM object is not a building with a polygon footprint.')
    geometry = matches.geometry.iloc[0]
    if geometry.geom_type not in ('Polygon', 'MultiPolygon') or geometry.is_empty or not geometry.is_valid:
        raise ValueError('OSM building has no valid polygon footprint.')
    # Existing crop filenames contain building_id, so keep '/' out of the internal ID.
    return gpd.GeoDataFrame({'building_id': [building_id.replace('/', '_')],
                             'osm_id': [building_id], 'geometry': [geometry]}, crs=buildings.crs).to_crs(4326)


def search_boundaries(target):
    local = target.to_crs(target.estimate_utm_crs())
    geometry = local.geometry.iloc[0]
    boundaries = [gpd.GeoSeries([geometry.buffer(radius)], crs=local.crs).to_crs(4326)
                  for radius in (30, 100)]
    return local, boundaries[0], boundaries[1]


def filter_metadata(metadata, local_target):
    data = pd.read_csv(metadata, dtype={'id': str})
    required = ['id', 'lat', 'lng', 'computed_compass_angle']
    if not set(required).issubset(data):
        raise ValueError('Mapillary metadata is missing image IDs, coordinates or headings.')
    for column in required[1:]:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna(subset=required)
    data = data[data.lat.between(-90, 90) & data.lng.between(-180, 180)
                & data.computed_compass_angle.between(0, 360)]
    points = gpd.GeoSeries(gpd.points_from_xy(data.lng, data.lat), index=data.index,
                          crs=4326).to_crs(local_target.crs)
    geometry = local_target.geometry.iloc[0]
    keep = (points.distance(geometry) <= 30 + 1e-6) & ~points.intersects(geometry)
    data = data.loc[keep].drop_duplicates('id').copy()
    if 'quality_score' in data:
        data['quality_score'] = pd.to_numeric(data.quality_score, errors='coerce').fillna(0)
    else:
        data['quality_score'] = 0.0
    data.to_csv(metadata, index=False)
    return data


def restore_target(footprints, target, local_target):
    """Keep the exact target, removing only overlapping geometry from neighbours."""
    buildings = gpd.read_file(footprints).to_crs(local_target.crs)
    geometry = local_target.geometry.iloc[0]
    buildings = buildings[buildings.intersects(geometry.buffer(100))].copy()
    buildings['geometry'] = buildings.geometry.difference(geometry)
    buildings = buildings[~buildings.geometry.is_empty].explode(index_parts=False)
    # Ignore numerical projection slivers smaller than 0.01 square metres.
    buildings = buildings[(buildings.geom_type == 'Polygon') & (buildings.area > .01)].to_crs(4326)
    # Neighbour IDs are only internal obstruction labels; OSM numeric IDs lack types here.
    buildings['building_id'] = ['context_' + str(i) for i in range(len(buildings))]
    buildings = gpd.GeoDataFrame(pd.concat([buildings, target], ignore_index=True), crs=4326)
    buildings.to_file(footprints, driver='GeoJSON')


def select_views(data, limit, final=False):
    """Greedily add uncovered visible facades, breaking ties by quality and angle."""
    remaining = data.drop_duplicates('pid').reset_index(drop=True).copy()
    columns = (['blur_score', 'building_ratio'] if final else [])
    columns += ['frontality', 'quality_score', 'aov_geo', 'distance', 'pid']
    columns = [column for column in columns if column in remaining]
    remaining = remaining.sort_values(columns,
        ascending=[column in ('distance', 'pid') for column in columns], kind='stable')
    chosen, covered = [], set()
    while len(chosen) < limit and not remaining.empty:
        sides = remaining.visible_sides.apply(lambda value: set(json.loads(value)))
        gains = sides.apply(lambda value: len(value - covered))
        index = gains.idxmax()
        chosen.append(remaining.loc[index])
        covered.update(sides.loc[index])
        remaining = remaining.drop(index)
    return pd.DataFrame(chosen, columns=data.columns).reset_index(drop=True)


def run_building(building_id, mly_api_key, max_images=5):
    building_id = normalize_id(building_id)
    if not isinstance(max_images, int) or isinstance(max_images, bool) or not 1 <= max_images <= 20:
        raise ValueError('max_images must be between 1 and 20 (the panorama download limit).')
    if not mly_api_key:
        raise ValueError('Set the local MAPILLARY_API_KEY environment variable first.')
    base = Path('output') / ('building_' + building_id.replace('/', '_'))
    base.mkdir(parents=True, exist_ok=True)
    # Fresh run directories prevent stale AoV, detections or classifiers being reused.
    output = Path(tempfile.mkdtemp(prefix='run_', dir=base))
    data_dir, img_dir = output / '01_data', output / '02_img'
    data_dir.mkdir()
    img_dir.mkdir()
    (img_dir / 'individual_building').mkdir()
    pd.DataFrame(columns=['image_name', 'pid', 'building_id']).to_csv(img_dir / 'individual_building.csv', index=False)
    result_path = img_dir / 'individual_building_select.csv'
    pd.DataFrame(columns=RESULT_COLUMNS).to_csv(result_path, index=False)

    def finish(status):
        result = pd.read_csv(result_path, dtype={'pid': str, 'building_id': str})
        if result.empty:
            pd.DataFrame(columns=RESULT_COLUMNS).to_csv(result_path, index=False)
        summary = {'osm_id': building_id, 'status': status, 'selected_images': len(result),
                   'max_images': max_images, 'download_limit': 20}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'{status}: {len(result)} image(s). Results: {result_path}')
        return result_path

    stage = 'target'
    try:
        target = get_target(building_id, data_dir)
        target.to_file(data_dir / 'target.geojson', driver='GeoJSON')
        local, pano_boundary, context_boundary = search_boundaries(target)
        stage = 'metadata'
        metadata_path = prepare_mly_metadata(pano_boundary.total_bounds.tolist(), str(data_dir), mly_api_key)
        metadata = filter_metadata(metadata_path, local)
        if metadata.empty:
            return finish('no_panoramas')
        stage = 'footprints'
        footprint_path = data_dir / 'footprint.geojson'
        download_and_merge_buildings(context_boundary.total_bounds.tolist(), str(data_dir), overwrite=True)
        restore_target(footprint_path, target, local)
        target_id = target.building_id.iloc[0]
        aov_path = data_dir / 'aov.csv'
        stage = 'visibility'
        geo_aov_calculator(str(metadata_path), str(footprint_path), buffer=30,
                           distance_between_points=2, out_dir=str(aov_path), batch_size=20,
                           max_workers=4, target_building_id=target_id)
        aov = pd.read_csv(aov_path, dtype={'pid': str, 'building_id': str})
        aov = aov[(aov.building_id == target_id) & (aov.aov_geo > 10) & (aov.aov_geo < 130)]
        if aov.empty:
            return finish('no_visible_views')
        aov = aov.merge(metadata[['id', 'quality_score']], left_on='pid', right_on='id', validate='many_to_one')
        candidates = select_views(aov, 20)
        selected_metadata = metadata[metadata.id.isin(candidates.pid)]
        selected_path = data_dir / 'pids_urls_select.csv'
        selected_metadata.to_csv(selected_path, index=False)
        # Preserve facade information separately from the existing detector CSV contract.
        candidates.to_csv(data_dir / 'candidate_views.csv', index=False)
        candidates.to_csv(aov_path, index=False)
        stage = 'download'
        download_selected_images(str(img_dir), str(selected_path), mly_api_key)
        from openfacades.detection.pano2pers import _find_image_file
        available = candidates.pid.apply(lambda pid: _find_image_file(pid, str(img_dir / 'mly_svi')) is not None)
        if not available.any():
            raise RuntimeError('No candidate panoramas were downloaded; check the download service and API key.')
        candidates = candidates[available].copy()
        candidates.to_csv(aov_path, index=False)
        stage = 'detection'
        run_object_detection(str(aov_path), str(img_dir), max_fov=135)
        crops = pd.read_csv(img_dir / 'individual_building.csv', dtype={'pid': str, 'building_id': str})
        crops = crops[crops.building_id == target_id]
        if crops.empty:
            return finish('no_detections')
        stage = 'quality'
        select_suitable_building_image(str(img_dir))
        qualified = pd.read_csv(result_path, dtype={'pid': str, 'building_id': str})
        qualified = qualified[qualified.building_id == target_id]
        if qualified.empty:
            return finish('no_quality_views')
        qualified = qualified.merge(candidates, on=['pid', 'building_id'], validate='many_to_one')
        from openfacades.detection.filter import calculate_blur_score
        qualified['blur_score'] = qualified.image_name.apply(
            lambda name: calculate_blur_score(str(img_dir / 'individual_building' / name)))
        ratios = pd.read_csv(img_dir / 'pixel_ratios.csv').set_index('filename_key')
        qualified['building_ratio'] = qualified.image_name.apply(
            lambda name: ratios.get('Building', pd.Series(dtype=float)).get(Path(name).stem, 0))
        result = select_views(qualified, max_images, final=True)
        result['osm_id'] = building_id
        result = result.drop(columns=[column for column in result if column.startswith('Unnamed:')])
        result.to_csv(result_path, index=False)
        return finish('complete')
    except Exception as error:
        # Do not record request exceptions: a third-party error may contain a token URL.
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(result_path, index=False)
        (output / 'summary.json').write_text(json.dumps({'osm_id': building_id, 'status': 'error',
            'stage': stage, 'error_type': type(error).__name__}) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--building-id', required=True, help='OSM way/123, relation/123 or a numeric way ID')
    parser.add_argument('--max-images', type=int, default=5, help='Maximum returned images (1–20; default 5)')
    args = parser.parse_args()
    try:
        run_building(args.building_id, os.environ.get('MAPILLARY_API_KEY'), args.max_images)
    except Exception as error:
        # Keep third-party request URLs, including potential tokens, out of CLI logs.
        parser.exit(1, f'Building search failed ({type(error).__name__}). Check local configuration and services.\n')


if __name__ == '__main__':
    main()
