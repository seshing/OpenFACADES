import os
import cv2
import ast
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Optional, List, Dict, Any

def xyz2lonlat(xyz: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates (x, y, z) to longitude/latitude in radians.

    Args:
        xyz (np.ndarray): Array of shape (..., 3) representing x, y, z coordinates.

    Returns:
        np.ndarray: Array of shape (..., 2) with [longitude, latitude] in radians.
    """
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = np.arctan2(x, z)
    lat = np.arcsin(y)
    return np.concatenate([lon, lat], axis=-1)

def lonlat2XY(lonlat: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert longitude/latitude in radians to 2D equirectangular coordinates.

    Args:
        lonlat (np.ndarray): Array of shape (..., 2) with [longitude, latitude].
        shape (tuple): Shape of the original equirectangular image (height, width, channels).

    Returns:
        np.ndarray: Array of shape (..., 2) with [X, Y] pixel coordinates.
    """
    height, width = shape[0], shape[1]
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (width - 1)
    Y = (lonlat[..., 1:] / np.pi + 0.5) * (height - 1)
    return np.concatenate([X, Y], axis=-1)

def get_perspective(
    original_image: np.ndarray,
    FOV: float,
    THETA: float,
    PHI: float,
    height: float,
    width: float
) -> Image.Image:
    """
    Perform perspective transformation on an equirectangular image,
    simulating a camera pointed at (THETA, PHI) with a given FOV.
    This section of code is adapted from Equirec2Perspec: https://github.com/fuenwang/Equirec2Perspec

    Args:
        original_image (np.ndarray): The original image in RGB format.
        FOV (float): Field of View in degrees.
        THETA (float): Left/right angle in degrees.
        PHI (float): Up/down angle in degrees.
        height (float): The height of the output image in pixels.
        width (float): The width of the output image in pixels.

    Returns:
        Image.Image: A PIL Image object representing the perspective-cropped image.
    """
    width_int = max(1, int(round(width)))
    height_int = max(1, int(round(height)))

    f = 0.5 * width_int / math.tan(0.5 * math.radians(FOV))
    cx = (width_int - 1) / 2.0
    cy = (height_int - 1) / 2.0

    K = np.array([
        [f,   0, cx],
        [0,   f, cy],
        [0,   0,  1],
    ], dtype=np.float32)
    K_inv = np.linalg.inv(K)

    x_coords = np.arange(width_int, dtype=np.float32)
    y_coords = np.arange(height_int, dtype=np.float32)
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    z = np.ones_like(x_mesh)

    xyz = np.stack([x_mesh, y_mesh, z], axis=-1) @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    R1, _ = cv2.Rodrigues(y_axis * math.radians(THETA))
    R2, _ = cv2.Rodrigues((R1 @ x_axis) * math.radians(PHI))
    R = R2 @ R1

    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz)
    XY = lonlat2XY(lonlat, shape=original_image.shape).astype(np.float32)

    persp = cv2.remap(
        original_image,
        XY[..., 0],
        XY[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )
    return Image.fromarray(persp)

def calculate_dynamic_height(
    box_height: float,
    height: float,
    y_center: float,
    min_scale: float = 1.0,
    max_scale: float = 1.1
) -> float:
    
    """
    Dynamically adjusts the height based on the tangent of the upper portion
    of the bounding box angle.
    """
    
    box_height_degree_upper = (0.5 - y_center + (box_height / 2)) * 180
    tan_value = math.tan(math.radians(box_height_degree_upper))
    tan_normalized = tan_value / (1 + tan_value)
    height_scale = min_scale + (max_scale - min_scale) * tan_normalized
    return box_height * height * height_scale

def _find_image_file(pid: str, image_dir: str) -> Optional[str]:
    """
    Helper to locate the correct image file (.jpeg or .png) for a given PID.
    Searches recursively in subfolders. Returns None if not found.
    """
    for root, _, files in os.walk(image_dir):
        for name in files:
            if name == f"{pid}.jpeg" or name == f"{pid}.png":
                return os.path.join(root, name)
    return None

def save_bounding_boxes_as_images(
    all_results_path: str,
    image_dir: str,
    out_dir: str
) -> pd.DataFrame:
    
    """
    1) Reads bounding box metadata from `all_results_path` CSV.
    2) Locates and loads each image from `image_dir`.
    3) For each bounding box, applies perspective cropping.
    4) Saves the cropped image in `out_dir`.
    5) Returns a DataFrame summarizing the results.

    Args:
        all_results_path (str): Path to CSV containing columns:
            - 'pid': unique image identifier
            - 'building_id': building identifier
            - 'boxes': stringified list [x_center, y_center, w, h, ...]
        image_dir (str): Directory containing input images in .jpeg or .png.
        out_dir (str): Output directory to store subfolder of cropped images.

    Returns:
        pd.DataFrame: Summary DataFrame with columns:
            - 'image_name'
            - 'pid'
            - 'building_id'
    """

    all_results = pd.read_csv(all_results_path)
    building_dir = os.path.join(out_dir, "individual_building")
    os.makedirs(building_dir, exist_ok=True)

    all_data: List[Dict[str, Any]] = []
    pid_to_imgurl = {}
    
    for pid in all_results['pid'].unique():
        img_path = _find_image_file(pid, image_dir)
        if img_path is not None:
            pid_to_imgurl[pid] = img_path

    for _, row in tqdm(all_results.iterrows(), total=len(all_results), desc="Cropping building images"):
        pid = row["pid"]
        building_id = row["building_id"]
        boxes = ast.literal_eval(row["boxes"])  # e.g. [cx, cy, w, h, confidence]
        img_path = pid_to_imgurl.get(pid, None)
        if img_path is None:
            print(f"Image not found for PID: {pid}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_height, img_width, _ = img.shape

        x_center, y_center, box_width, box_height = map(float, boxes[:4])

        theta = int((x_center - 0.5) * 360)  
        phi = int((0.5 - y_center) * 180)    
        FOV = int(box_width * 360)

        dyn_height = calculate_dynamic_height(
            box_height=box_height,
            height=img_height,
            y_center=y_center
        )
        out_width = box_width * img_width

        # Perspective crop
        cropped_image = get_perspective(
            original_image=img,
            FOV=FOV,
            THETA=theta,
            PHI=phi,
            height=dyn_height,
            width=out_width
        )

        out_filename = f"pid_{pid}_bdid_{building_id}.png"
        building_url = os.path.join(building_dir, out_filename)
        cropped_image.save(building_url)

        all_data.append({
            "image_name": out_filename,
            "pid": pid,
            "building_id": building_id,
        })

        pd.DataFrame(all_data).to_csv(os.path.join(out_dir, "individual_building.csv"), index=False)