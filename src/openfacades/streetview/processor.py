import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def bbox_to_geojson(bbox):
    """
    Convert a bounding box to a GeoDataFrame in GeoJSON format.

    Parameters:
    bbox (tuple): A tuple of (minx, miny, maxx, maxy) representing the bounding box.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the bounding box as a GeoJSON.
    """
    minx, miny, maxx, maxy = bbox
    geom = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs="EPSG:4326")
    return gdf

def process_metadata(input_file, 
                     output_file, 
                     chunk_size = 500, 
                     selected_columns = ['id', 'computed_compass_angle', 'quality_score', 'computed_geometry', 'url', 'captured_at']):
    chunks = pd.read_csv(input_file, chunksize=chunk_size)

    for i, chunk in enumerate(chunks):
        print(f'Processing chunk {i+1}')
        chunk_clean = chunk[selected_columns]
        chunk_clean[['lng','lat']] = chunk_clean['computed_geometry'].apply(lambda x: pd.Series(eval(x)['coordinates']) if pd.notnull(x) else pd.Series([None, None]))
        chunk_clean.drop('computed_geometry', axis=1, inplace=True)
        
        # Drop rows where 'lat' or 'lng' is NaN
        chunk_clean.dropna(subset=['lng','lat'], inplace=True)
        
        # Filter rows based on 'quality_score'
        chunk_clean = chunk_clean[(chunk_clean['quality_score'] == 0) | (chunk_clean['quality_score'] >= 0.2)]
        
        if i == 0:
            chunk_clean.to_csv(output_file, index=False, mode='w')
        else:
            chunk_clean.to_csv(output_file, index=False, mode='a', header=False)
        print(f'Finished processing chunk {i+1}')
        
        
def filter_img_by_aov(aov_dir, 
                      metadata_dir,
                      max_aov=130, 
                      min_aov=10, 
                      max_image_per_building=6, 
                      method='by_aov'):
    df_aov = pd.read_csv(aov_dir)
    data = df_aov.copy()

    data = data[(data['aov_geo'] >= min_aov) & (data['aov_geo'] <= max_aov)]

    if method == 'random':
        img_select = data.groupby('building_id', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_image_per_building))
        )
    elif method == 'by_aov':
        img_select = data.sort_values('aov_geo', ascending=False)
        img_select = img_select.groupby('building_id').head(max_image_per_building)

    img_select.reset_index(drop=True, inplace=True)
    img_pid_select = img_select['pid'].to_list()
    metadata = pd.read_csv(metadata_dir)
    metadata_select = metadata[metadata['id'].isin(img_pid_select)]
    print(f"Selected {len(metadata_select)} rows from {len(metadata)} data points")

    return metadata_select

