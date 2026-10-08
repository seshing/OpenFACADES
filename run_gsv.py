"""Process local GSV panoramas using OpenFACADES."""
import argparse
import math
from pathlib import Path


def run_gsv(images, metadata, output="output", footprints=None):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from openfacades.streetview import geo_aov_calculator
    from run import download_and_merge_buildings, run_object_detection, select_suitable_building_image

    images = Path(images)
    if not images.is_dir():
        raise ValueError("The image directory does not exist.")
    data = pd.read_csv(metadata, dtype={"pid": str})
    required = ["pid", "lat", "lng", "heading"]
    if not set(required).issubset(data.columns):
        raise ValueError("GSV metadata needs pid, lat, lng and heading (in radians).")
    data = data[required].copy()
    if data.empty or data.isna().any().any():
        raise ValueError("GSV metadata must contain complete image IDs and coordinates.")
    for column in ["lat", "lng", "heading"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    data["degree"] = 360 * data["heading"] / (2 * math.pi)
    if not (data.lat.between(-90, 90).all() and data.lng.between(-180, 180).all()
            and data.degree.between(0, 360).all()):
        raise ValueError("Invalid coordinates or heading values in GSV metadata.")

    output = Path(output)
    data_dir = output / "01_data"
    img_dir = output / "02_img"
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    cleaned = data.rename(columns={"pid": "id", "degree": "computed_compass_angle"})
    cleaned_path = data_dir / "pids_urls_clean.csv"
    cleaned.to_csv(cleaned_path, index=False)

    # Surround the metadata points with a rectangle, expanded by 100 metres.
    points = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lng, data.lat), crs="EPSG:4326")
    local = points.to_crs(points.estimate_utm_crs())
    left, bottom, right, top = local.total_bounds
    boundary = gpd.GeoSeries([box(left - 100, bottom - 100, right + 100, top + 100)], crs=local.crs)
    boundary = boundary.to_crs("EPSG:4326")
    boundary.to_file(data_dir / "bbox.geojson", driver="GeoJSON")

    footprint_path = data_dir / "footprint.geojson"
    if footprints:
        buildings = gpd.read_file(footprints)
        if "building_id" not in buildings.columns:
            buildings = buildings.rename(columns={"osmid": "building_id"})
        if "building_id" not in buildings.columns:
            raise ValueError("Building footprints need building_id or osmid.")
        buildings.to_crs("EPSG:4326").to_file(footprint_path, driver="GeoJSON")
    else:
        download_and_merge_buildings(boundary.total_bounds.tolist(), str(data_dir), overwrite=True)

    aov_path = data_dir / "aov.csv"
    aov_path.unlink(missing_ok=True)
    geo_aov_calculator(str(cleaned_path), str(footprint_path), buffer=30,
                       distance_between_points=2, out_dir=str(aov_path),
                       batch_size=20, max_workers=4)
    if pd.read_csv(aov_path).empty:
        print("No visible buildings found for these metadata points.")
        return
    run_object_detection(str(aov_path), str(img_dir), images_folder=str(images))
    if pd.read_csv(img_dir / "individual_building.csv").empty:
        print("No building images detected.")
        return
    select_suitable_building_image(str(img_dir))


def main():
    parser = argparse.ArgumentParser(description="Process local GSV panoramas; no image downloading.")
    parser.add_argument("--images", required=True, help="Directory containing <pid>.jpg/.jpeg/.png panoramas")
    parser.add_argument("--metadata", required=True, help="CSV containing pid, lat, lng, heading (radians)")
    parser.add_argument("--footprints", help="Optional local building footprints; otherwise download OSM + Overture")
    args = parser.parse_args()
    run_gsv(args.images, args.metadata, footprints=args.footprints)


if __name__ == "__main__":
    main()
