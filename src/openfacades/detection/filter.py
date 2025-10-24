from zensvi.cv import Segmenter, ClassifierPlaces365
import os
import pandas as pd
from PIL import Image
import pandas as pd
import cv2
import numpy as np

def get_image_sizes(directory):
    data = []
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            try:
                with Image.open(item_path) as img:
                    width, height = img.size
                    file_size = os.path.getsize(item_path)
                    data.append({
                        'image_name': item,
                        'width': width,
                        'height': height,
                        'file_size': file_size
                    })
            except Exception as e:
                print(f"Error processing {item_path}: {e}")
    
    return pd.DataFrame(data)
   
def calculate_blur_score(image_path):
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return 0
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        blur_score = np.var(laplacian)
        return blur_score
    except Exception as e:
        print(f"Error calculating blur score for {image_path}: {e}")
        return 0

def check_blur_in_directory(directory):
    """
    Process all images in a directory and check if they are blurry.
    
    Args:
        directory: Path to directory containing images
        threshold: Blur score threshold (lower values indicate blurrier images)
        
    Returns:
        DataFrame: Contains filenames, blur scores, and blurry status
    """
    results = []
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            try:
                blur_score = calculate_blur_score(item_path)
                results.append({
                    'image_name': item,
                    'blur_score': blur_score
                })
            except Exception as e:
                print(f"Error processing {item_path}: {e}")
    
    return pd.DataFrame(results)

def segmantation(building_path, output_path): 
    dataset = "mapillary"
    task = "semantic"
    segmenter = Segmenter(dataset=dataset, task=task)
    pixel_ratio_save_format = "csv"
    csv_format = "wide" # "long" or "wide"
    segmenter.segment(building_path,
                    dir_summary_output = output_path,
                    batch_size=2,
                    save_format = pixel_ratio_save_format,
                    csv_format = csv_format)


def places365(building_path, output_path):
    pixel_ratio_save_format = "csv"
    csv_format = "wide" # "long" or "wide"
    classifier = ClassifierPlaces365()
    classifier.classify(
        building_path,
        batch_size = 32,
        save_format = pixel_ratio_save_format,
        csv_format = csv_format,
        dir_summary_output=output_path
    )
    

def filter_img_by_features(building_path, output_path, min_file_size=20000, min_building_ratio=0.15, max_wall_ratio=0.3, min_blur=30):
    """
    Filter images based on:
    1. Image file size (>= min_file_size)
    2. Building pixel ratio (>= min_building_ratio)
    3. Wall pixel ratio (<= max_wall_ratio)
    4. Environment type (outdoor only)
    5. Blur score (<= max_blur)
    
    Args:
        building_path: Directory with building images
        output_path: Directory with segmentation and classification results
        min_file_size: Minimum file size in bytes (default: 20KB)
        min_building_ratio: Minimum building ratio (default: 0.15)
        max_wall_ratio: Maximum wall ratio (default: 0.3)
        max_blur: Maximum blur score (default: 30)
        
    Returns:
        Set of building IDs that meet all criteria
    """
    
    if not os.path.exists(os.path.join(output_path, 'results.csv')):
        print("Running Places365 classification...")
        places365(building_path, output_path)
    
    places_df = pd.read_csv(
    os.path.join(output_path, 'results.csv'),
    usecols=['filename_key', 'environment_type']
    )
    outdoor_filtered = places_df[places_df['environment_type'] == 'outdoor']
    outdoor_filtered_keys = set(outdoor_filtered['filename_key'])
    
    if not os.path.exists(os.path.join(output_path, 'pixel_ratios.csv')):
        print("Running segmentation...")
        segmantation(building_path, output_path)

    segmentation_df = pd.read_csv(os.path.join(output_path, 'pixel_ratios.csv'))
    building_filtered = segmentation_df[segmentation_df['Building'] >= min_building_ratio]
    building_wall_filtered = building_filtered[building_filtered['Wall'] <= max_wall_ratio]
    segmentation_filtered_keys = set(building_wall_filtered['filename_key'])
    

    # Get image sizes and file information
    print("Getting image size...")
    all_sort = get_image_sizes(building_path)
    all_sort['filename_key'] = all_sort['image_name'].apply(lambda x: os.path.splitext(x)[0])
    filter_size_all = all_sort[all_sort['file_size'] >= min_file_size]
    filter_size_all.reset_index(drop=True, inplace=True)
    size_filtered_keys = set(filter_size_all['filename_key'])
    
    print("Evaluating bluriness...")
    blur_scores = check_blur_in_directory(building_path)
    blur_scores['filename_key'] = blur_scores['image_name'].apply(lambda x: os.path.splitext(x)[0])
    blur_filtered = blur_scores[blur_scores['blur_score'] > min_blur]
    blur_filtered_keys = set(blur_filtered['filename_key'])
    
    # Combine all filters
    final_filename_keys = size_filtered_keys.intersection(
        segmentation_filtered_keys, 
        outdoor_filtered_keys,
        blur_filtered_keys
    )
    print(f"{len(final_filename_keys)} images passed all filters from {len(all_sort)} total images.")
    
    return final_filename_keys