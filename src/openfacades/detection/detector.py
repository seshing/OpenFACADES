import os
import cv2
import csv
import io
import math
import warnings
import random
import urllib.request

import pandas as pd
from PIL import Image
from tqdm import tqdm

# GroundingDINO
from groundingdino.util.inference import load_model, load_image, predict
from .pano2pers import save_bounding_boxes_as_images

# Suppress unhelpful warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class GroundingDinoDetector:
    """
    A class to handle loading GroundingDINO model, downloading weights (if needed),
    and detecting bounding boxes on images based on textual prompts.
    """

    def __init__(
        self,
        config_path: str = "config/GroundingDINO_SwinT_OGC.py",
        weights_url: str = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        weights_dir: str = "weights",
        weights_file: str = "groundingdino_swint_ogc.pth"
    ):
        """
        Initialize the detector, ensuring weights are downloaded
        and the model is loaded.

        Args:
            config_path (str): Path to the GroundingDINO config.py file.
            weights_url (str): URL to download model weights if they don't exist locally.
            weights_dir (str, optional): Local directory to store downloaded weights. Defaults to "weights".
            weights_file (str, optional): Name of the weights file. Defaults to "groundingdino_swint_ogc.pth".
        """
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.current_dir, config_path)
        self.weights_dir = os.path.join(self.current_dir, weights_dir)
        self.weights_file = os.path.join(self.weights_dir, weights_file)
        self.weights_url = weights_url

        # Ensure weights are downloaded
        self._download_weights()

        # Load the model
        self.model = load_model(
            self.config_path,
            self.weights_file
        )

    def _download_weights(self):
        """Download GroundingDINO weights if they don't exist locally."""
        if not os.path.exists(self.weights_file):
            os.makedirs(self.weights_dir, exist_ok=True)
            print("Downloading GroundingDINO weights...")
            urllib.request.urlretrieve(self.weights_url, self.weights_file)
            print(f"Weights downloaded to {self.weights_file}")
        else:
            # print("Weights file already exists.")
            pass

    @staticmethod
    def generate_random_color():
        """Generate a random BGR color for bounding boxes."""
        return [random.randint(0, 255) for _ in range(3)]

    @staticmethod
    def filter_boxes_based_on_conditions(results, cropped_image):
        """
        Filter boxes to keep the one that contains the center of cropped_image
        and has the highest confidence if multiple boxes exist.

        Returns: (filtered_boxes, filtered_logits)
        """
        img_width = cropped_image.shape[1]
        img_height = cropped_image.shape[0]
        central_x = img_width / 2

        results['boxes'] = [
            [
                (box[0] - 0.5 * box[2]) * img_width,
                (box[1] - 0.5 * box[3]) * img_height,
                (box[0] + 0.5 * box[2]) * img_width,
                (box[1] + 0.5 * box[3]) * img_height
            ]
            for box in results['boxes']
        ]

        keep_indices = [
            i for i, box in enumerate(results['boxes'])
            if box[0] <= central_x <= box[2]
        ]

        if len(keep_indices) >= 2:
            filtered_logits = [results['logits'][i] for i in keep_indices]
            max_logit_index = filtered_logits.index(max(filtered_logits))
            keep_indices = [keep_indices[max_logit_index]]

        filtered_boxes = [results['boxes'][i] for i in keep_indices]
        filtered_logits = [results['logits'][i] for i in keep_indices]

        return filtered_boxes, filtered_logits

    def run_detection(
        self,
        geo_aov_csv: str,
        images_folder: str,
        output_folder: str,
        max_aov=130, 
        min_aov=10,
        box_threshold: float = 0.35,
        adjust_angle: float = 0.0,
        text_threshold: float = 0.3,
        save_annotated_images: bool = True,
        text_prompt: str = "building"
    ):
        """
        Main method to detect buildings (or any text prompt) in images using
        bounding boxes from GroundingDINO, filtered by AoV geometry from CSV.

        Args:
            geo_aov_csv (str): Path to CSV containing fields:
            pid, building_id, left_angle_image, right_angle_image, ...
            images_folder (str): Folder with {pid}.png images.
            output_folder (str): Folder to save results (annotated images, CSV).
            max_aov (int, optional): Maximum angle of view to filter. Default=130
            min_aov (int, optional): Minimum angle of view to filter. Default=10
            box_threshold (float, optional): Box threshold for GroundingDINO. Default=0.35
            adjust_angle (float, optional): Extra fraction of image width to expand crop. Default=0.0
            save_annotated_images (bool, optional): Whether to save annotated images. Default=True
            text_prompt (str, optional): Detection prompt text. Default="building"

        Returns:
            None. Saves 'detected_buildings.csv' and annotated images (if opted).
        """

        # Load the CSV with AoV geometry
        geo_aov = pd.read_csv(geo_aov_csv)
        geo_aov['pid'] = geo_aov['pid'].astype(str)
        geo_aov['left_angle_image'] = (geo_aov['left_angle_geo'] - geo_aov['compass_angle'] + 180) % 360
        geo_aov['right_angle_image'] = (geo_aov['right_angle_geo'] - geo_aov['compass_angle'] + 180) % 360
        geo_aov = geo_aov[(geo_aov['aov_geo'] > min_aov) & (geo_aov['aov_geo'] < max_aov)]  # Filter AoV based on min and max values
        
        # Prepare sets of PIDs
        pid_all = self._check_id(images_folder)
        
        file_path = os.path.join(output_folder, "building_bbox.csv")
        file_exists = os.path.isfile(file_path)
        already_id = set()
        if file_exists:
            already_id = set(pd.read_csv(file_path)['pid'].astype(str).unique())
        remaining_pid = pid_all - already_id
        print(f"Found {len(pid_all)} images total in {images_folder}, {len(remaining_pid)} remain to process.")
        
        # Create a DataFrame of 'pid' and 'image_url' based on 'remaining_pid'
        img_data = []
        for pid in remaining_pid:
            img_path = next((os.path.join(root, name) for root, _, files in os.walk(images_folder) for name in files if name == f"{pid}.png"), None)
            if img_path is not None and os.path.exists(img_path):
                img_data.append({'pid': pid, 'image_url': img_path})
                
        img_df = pd.DataFrame(img_data)
        
        # Main detection loop
        for pid in tqdm(remaining_pid, desc="Processing images", dynamic_ncols=True, leave=False):
            img_row = img_df[img_df['pid'] == pid]
            if img_row.empty:
                continue
            img_path = img_row.iloc[0]['image_url']
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"No image found for PID: {pid}")
                continue

            img_geo_aov = geo_aov[geo_aov['pid'] == pid]

            for _, row in img_geo_aov.iterrows():
                building_id = row['building_id']
                left_bd = row.get('left_angle_image', float("nan"))
                right_bd = row.get('right_angle_image', float("nan"))
                if math.isnan(left_bd) or math.isnan(right_bd):
                    continue

                img_height, img_width = img.shape[:2]
                color = self.generate_random_color()

                # Convert angles to pixel positions
                x_left = max(int(left_bd / 360 * img_width - adjust_angle * img_width), 0)
                x_right = min(int(right_bd / 360 * img_width + adjust_angle * img_width), img_width)

                # Crop the relevant portion of the image
                if x_right <= x_left:
                    # print(f"Invalid crop for building: {building_id}")
                    continue

                cropped_image = img[:, x_left:x_right]
                pil_image = Image.fromarray(cropped_image)

                # Convert to in-memory bytes for GroundingDINO
                image_io = io.BytesIO()
                pil_image.save(image_io, format='PNG')
                image_io.seek(0)
                _, gdi_image = load_image(image_io)

                # Predict boxes
                
                try:
                    boxes, logits, phrases = predict(
                        model=self.model,
                        image=gdi_image,
                        caption=text_prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold
                    )
                except Exception as e:
                    print(f"Error processing image {img_path} and building id {building_id}: {e}")
                    continue

                results = {
                    'boxes': boxes.tolist(),
                    'logits': logits.tolist(),
                    'phrases': phrases
                }

                # Filter boxes
                filtered_boxes, filtered_logits = self.filter_boxes_based_on_conditions(results, cropped_image)
                if not filtered_boxes:
                    continue

                # Take the highest-confidence box
                x1, y1, x2, y2 = map(int, [
                    filtered_boxes[0][0] + x_left,
                    filtered_boxes[0][1],
                    filtered_boxes[0][2] + x_left,
                    filtered_boxes[0][3]
                ])
                conf = float(filtered_logits[0])

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text = f"ID: {building_id}, Conf: {conf:.2f}"
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Convert to normalized coords
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                normalized_box = [
                    cx / img_width,
                    cy / img_height,
                    w / img_width,
                    h / img_height,
                    conf
                ]

                # Write to CSV
                with open(file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(["pid", "building_id", "boxes"])
                        file_exists = True  # Update the flag after writing the header
                    writer.writerow([pid, building_id, normalized_box])

            # Save annotated image if requested
            if save_annotated_images:
                os.makedirs(output_folder, exist_ok=True)
                annotated_dir = os.path.join(output_folder, "annotated_img")
                if save_annotated_images:
                    os.makedirs(annotated_dir, exist_ok=True)
                
                save_path = os.path.join(annotated_dir, f"{pid}.png")
                cv2.imwrite(save_path, img)
                
        save_bounding_boxes_as_images(file_path, 
                                      images_folder,
                                      output_folder)

        print(f"Processing complete. Results saved in {output_folder}.")

    @staticmethod
    def _check_id(folder):
        """
        Retrieve set of IDs from files in the specified folder, i.e., the
        file's stem without extension. Skips hidden files.
        """
        if not os.path.exists(folder):
            return set()
        return {
            os.path.splitext(name)[0]
            for root, _, files in os.walk(folder)
            for name in files
            if not name.startswith('.') and os.path.isfile(os.path.join(root, name))
        }