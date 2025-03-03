import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

plt.rcParams['font.size'] = 14  # Increase base font size

def project_gdf(gdf):
    """Project GeoDataFrame to a local UTM coordinate system based on longitude."""
    mean_longitude = gdf["geometry"].representative_point().x.mean()
    utm_zone = int(np.floor((mean_longitude + 180) / 6) + 1)
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    return gdf.to_crs(utm_crs)


def visualize_building(csv_path, geojson_path, img_dir):
    """
    Visualize a randomly selected building with its footprint and images.
    North is oriented to the top of the map.
    
    Args:
        csv_path (str): Path to the CSV file containing building data
        geojson_path (str): Path to the GeoJSON file containing footprint data
        img_dir (str): Directory containing the building images
    """
    # Load data
    df = pd.read_csv(csv_path)
    gdf = gpd.read_file(geojson_path)
    gdf = project_gdf(gdf)

    # 1. Sample one building id
    unique_buildings = df['building_id'].unique()
    selected_building_id = random.choice(unique_buildings)

    # Get all rows for the selected building
    building_rows = df[df['building_id'] == selected_building_id]
    # selected_image_id = building_rows['pid'].values
    image_paths = building_rows['image_name'].drop_duplicates().values

    # Limit to at most 8 images
    image_paths = image_paths[:8]
    # print(image_paths)

    # Create a figure with a proper grid layout
    fig = plt.figure(figsize=(32, 10))

    # Create two separate areas - left for map, right for images
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5])

    # Left subplot for the map
    ax_map = fig.add_subplot(gs[0])
    gdf.plot(ax=ax_map, alpha=0.5, edgecolor='white', linewidth=0.5, color='#2079B4')
    selected_footprint = gdf[gdf['building_id'] == selected_building_id]
    if not selected_footprint.empty:
        selected_footprint.plot(ax=ax_map, color='#D1342B', alpha=1.0, edgecolor='none')
    
    # Ensure map is oriented with North at the top
    ax_map.set_aspect('equal')  # Preserves the correct aspect ratio
    
    # Add North arrow
    x, y, arrow_length = 0.05, 0.95, 0.05
    ax_map.annotate('N', xy=(x, y), xytext=(x, y-arrow_length*2),
                    arrowprops=dict(facecolor='black', width=2, headwidth=8, headlength=10),
                    ha='center', va='bottom', xycoords='axes fraction', fontsize=18)
    
    ax_map.set_title(f"Building ID: {selected_building_id}", fontsize=20, fontweight='bold')
    ax_map.axis('off')

    # Right area for images - create a sub-gridspec
    num_images = len(image_paths)
    if num_images == 0:
        ax_img = fig.add_subplot(gs[1])
        ax_img.text(0.5, 0.5, "No images found", ha='center', va='center', fontsize=20)
        ax_img.axis('off')
    else:
        # Calculate grid dimensions (maximum 2x4 grid)
        cols = min(4, num_images)
        rows = min(2, (num_images + cols - 1) // cols)
        
        gs_right = GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs[1], wspace=0.1, hspace=0.1)
        
        for i, img_path in enumerate(image_paths):
            if i < rows * cols:  # Safety check
                ax = fig.add_subplot(gs_right[i // cols, i % cols])
                try:
                    full_path = os.path.join(img_dir, img_path)
                    img = plt.imread(full_path)
                    ax.imshow(img)
                except:
                    try:
                        base_dir = os.path.dirname(csv_path)
                        full_path = os.path.join(base_dir, img_path)
                        img = plt.imread(full_path)
                        ax.imshow(img)
                    except:
                        ax.text(0.5, 0.5, f"Failed to load\n{os.path.basename(img_path)}", 
                            ha='center', va='center', transform=ax.transAxes, fontsize=16)
                # Get the corresponding pid based on the image name to avoid mismatches
                try:
                    pid = building_rows[building_rows['image_name'] == img_path]['pid'].iloc[0]
                except IndexError:
                    pid = 'unknown'
                ax.set_title(f"PID: {pid}", fontsize=16)
                ax.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    
    return fig