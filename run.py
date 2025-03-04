import os
from pathlib import Path
from openfacades.footprint import BuildingDataDownloader, harmonize_buildings
from zensvi.download import MLYDownloader
from openfacades.streetview import bbox_to_geojson, process_metadata
from openfacades.streetview import geo_aov_calculator, filter_img_by_aov
from openfacades.detection import GroundingDinoDetector
import argparse

def parse_bbox(bbox_str: str) -> list[float]:
    trimmed = bbox_str.strip().strip("[]")

    parts = [x.strip() for x in trimmed.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have exactly 4 values.")
    return [float(x) for x in parts]

def download_and_merge_buildings(bbox, output_path):
    """
    Download OSM and Overture building footprints for the given bbox,
    harmonize them, and save as GeoJSON.
    """
    out_geojson = os.path.join(output_path, "footprint.geojson")
    
    if os.path.exists(out_geojson):
        print(f"Building footprints file already exists, skipping: {out_geojson}")
        return
    
    print("=== Downloading OSM Buildings ===")
    downloader_osm = BuildingDataDownloader(source="osm")
    osm_data = downloader_osm.download_buildings(bbox)

    print("=== Downloading Overture Buildings ===")
    downloader_overture = BuildingDataDownloader(source="overture")
    overture_data = downloader_overture.download_buildings(bbox)

    print("=== Harmonizing Building Footprints ===")
    ham_gdf = harmonize_buildings(
        gdf_osm=osm_data,
        gdf_external=overture_data,  # or other external data with matching columns
        filter_by_area=20,
        filter_underground=True
    )
    
    ham_gdf.to_file(out_geojson, driver="GeoJSON")
    print(f"Saved harmonized footprints to: {out_geojson}")


def prepare_mly_metadata(bbox, output_path, mly_api_key):
    """
    Generate a bounding-box GeoJSON, then download Mapillary metadata
    (pano metadata only) into 'pids_urls.csv'.
    """
    print("=== Preparing Bounding Box GeoJSON ===")
    bbox_gdf = bbox_to_geojson(bbox)
    bbox_path = os.path.join(output_path, "bbox.geojson")
    bbox_gdf.to_file(bbox_path, driver="GeoJSON")
    kwarg = {"image_type": "pano"}
    
    print("=== Downloading MLY Metadata ===")
    mly_downloader = MLYDownloader(mly_api_key)
    mly_downloader.download_svi(
        output_path,  # output directory
        input_shp_file=os.path.join(output_path, 'bbox.geojson'),  # path to the input shapefile containing the location information
        input_place_name="",  # name of the location to download
        resolution=2048,  # resolution of the image
        batch_size=1000,  # batch size for downloading images
        start_date=None,  # start date for downloading images (YYYY-MM-DD)
        end_date=None,  # end date for downloading images (YYYY-MM-DD)
        metadata_only=True,  # if True, only metadata is downloaded
        use_cache=True,  # if True, the cache is used
        additional_fields=["camera_type", "captured_at",
                           "computed_compass_angle", "computed_geometry",
                           "geometry", "quality_score"],  # Additional fields to fetch from the API. Defaults to ["all"].
        **kwarg
    )

    # Clean the raw metadata
    raw_csv = os.path.join(output_path, "pids_urls.csv")
    cleaned_csv = os.path.join(output_path, "pids_urls_clean.csv")
    print(f"=== Cleaning MLY Metadata ===\nFrom: {raw_csv}\nTo:   {cleaned_csv}")
    process_metadata(raw_csv, cleaned_csv)

    return cleaned_csv


def compute_and_filter_aov(cleaned_metadata, output_path):
    """
    Compute the AoV for each panorama and filter by constraints,
    then produce a CSV of selected panos ('pids_urls_select.csv').
    """
    aov_csv_path = os.path.join(output_path, "aov.csv")
    if not os.path.exists(aov_csv_path):
        print("=== Computing AoV ===")
        geo_aov_calculator(
            img_metadata=cleaned_metadata,
            json_dir=os.path.join(output_path, "footprint.geojson"),
            buffer=30,
            distance_between_points=2,
            out_dir=aov_csv_path,
            batch_size=20,
            max_workers=4
        )
    else:
        print("AoV CSV already exists, skipping: ", aov_csv_path)

    print("=== Filtering AoV (by max/min) ===")
    out_select = os.path.join(output_path, "pids_urls_select.csv")
    filter_img_by_aov(
        aov_csv_path,
        cleaned_metadata,
        max_aov=130,
        min_aov=10,
        max_image_per_building=5,
        method='by_aov'
    ).to_csv(out_select, index=False)

    return aov_csv_path, out_select


def download_selected_images(output_path, path_pid, mly_api_key):
    """
    Download selected panorama images (i.e., .png files) from the 'pids_urls_select.csv'.
    """
    print("=== Downloading Selected Panoramic Images ===")
    mly_downloader = MLYDownloader(mly_api_key)
    mly_downloader.download_svi(
        output_path,
        path_pid=Path(path_pid),    # e.g. pids_urls_select.csv
        update_pids=False,
        resolution=2048,
        batch_size=500,
        use_cache=True,
        image_type="pano"  # from your kwarg usage
    )


def run_object_detection(aov_csv_path, output_path):
    """
    Run GroundingDINO object detection on the downloaded 360 images,
    referencing the AoV CSV.
    """
    print("=== Initializing GroundingDinoDetector ===")
    detector = GroundingDinoDetector(
        config_path="config/GroundingDINO_SwinT_OGC.py",
        weights_url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    )

    print("=== Running Detection ===")
    detector.run_detection(
        geo_aov_csv=aov_csv_path,
        images_folder=os.path.join(output_path, "mly_svi"),
        output_folder=output_path,
        max_aov=130,
        min_aov=10,
        box_threshold=0.35,
        adjust_angle=0.05,
        save_annotated_images=True,
        text_prompt="building"
    )
    

def select_suitable_building_image(output_path):
    """
    Filter building images based on specific features
    """
    from openfacades.detection import filter_img_by_features
    import pandas as pd
    
    # Get suitable keys from feature filtering
    suitable_key = filter_img_by_features(
        os.path.join(output_path, 'individual_building'),
        output_path,
        min_file_size=20000,  # Bytes
        min_building_ratio=0.15,
        max_wall_ratio=0.3,
        min_blur=30
    )
    
    df = pd.read_csv(os.path.join(output_path, 'individual_building.csv'))
    bd_clean_path = os.path.join(output_path, 'individual_building_select.csv')
    df[df['image_name'].apply(lambda x: x.split('.')[0] in suitable_key)].to_csv(bd_clean_path)
    


def main():
    # === Configuration / Parameters ===
    parser = argparse.ArgumentParser(description="Run the OpenFACADES pipeline...")
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        required=True,
        help="Bounding box coordinates [left, bottom, right, top]"
    )
    parser.add_argument("--api_key", type=str, required=True, help="Mapillary API key")

    args = parser.parse_args()
    
    bbox = args.bbox
    mly_api_key = args.api_key
    
    output_path = os.path.join(os.getcwd(), "output")
    output_data_path = os.path.join(output_path, '01_data')
    output_img_path = os.path.join(output_path, '02_img')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_data_path, exist_ok=True)
    os.makedirs(output_img_path, exist_ok=True)

    # === 1) Download + Merge Building Footprints ===
    download_and_merge_buildings(bbox, output_data_path)

    # === 2) Prepare MLY Metadata ===
    cleaned_csv = prepare_mly_metadata(bbox, output_data_path, mly_api_key)

    # === 3) Compute + Filter AoV ===
    aov_csv_path, out_select = compute_and_filter_aov(cleaned_csv, output_data_path)

    # === 4) Download Panoramic Images (selected) ===
    download_selected_images(output_img_path, out_select, mly_api_key)

    # === 5) Run GroundingDINO Detection ===
    run_object_detection(aov_csv_path, output_img_path)
    
    # === 6) Filter Building Image by Features ===
    select_suitable_building_image(output_img_path)


if __name__ == "__main__":
    main()