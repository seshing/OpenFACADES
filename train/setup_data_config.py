import json

# =============================================================================
# CONFIGURATION - Update these paths for your setup
# =============================================================================

# Path to your training images directory
img_dir = "data/img"

# Path to your training annotations (JSONL format)
annotations = "data/jsonl/train.jsonl"

# Number of times to repeat the dataset during training
repeat_time = 1

def get_length(annotations):
    with open(annotations, 'r') as f:
        length = sum(1 for _ in f)
    return length

data = {
    "osm_caption": {
        "root": img_dir,
        "annotation": annotations,
        "data_augment": False,
        "repeat_time": repeat_time,
        "length": get_length(annotations)
    }
    
}

# Output path for the training configuration JSON
output_path = "shell/data/internvl_finetune_building.json"

print(f"Creating training data configuration...")
print(f"Image directory: {img_dir}")
print(f"Annotations file: {annotations}")
print(f"Dataset length: {get_length(annotations)}")
print(f"Output configuration: {output_path}")

with open(output_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print("Training data configuration created successfully!")