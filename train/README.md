
# Fine-tuning InternVL for Building Detection

This guide walks you through the process of fine-tuning InternVL3 models for building facade detection and analysis. We use **InternVL3-1B** as the example throughout this guide, but the process works for other model sizes (2B, 8B) as well.

## Prerequisites

- GPU resources available for training
- Conda or Python environment management
- Git for cloning repositories

## Complete Setup

Follow these steps to set up everything from scratch:

### Step 1: Clone Repositories and Setup Environment

```bash
# Clone OpenFACADES repository
git clone https://github.com/seshing/OpenFACADES.git
cd OpenFACADES

# Clone InternVL repository
git clone https://github.com/OpenGVLab/InternVL.git

# Create and activate conda environment
conda create -n internvl python=3.9
conda activate internvl

# Install dependencies
pip install -r train/requirements.txt
```

### Step 2: Download Pre-trained Model

```bash
# Download InternVL3-1B model
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-1B --local-dir InternVL/internvl_chat/pretrain/InternVL3-1B
```

## Training Steps

### Step 3: Prepare Training Data and Configuration

Navigate to the InternVL directory and run the training setup scripts:

```bash
cd InternVL/internvl_chat

# Download sample training data from Hugging Face
python3 ../../train/get_train_data.py

# Setup data configuration
python3 ../../train/setup_data_config.py

# Configure training parameters and generate training script
python3 ../../train/setup_training_config.py
```

**What these scripts do:**
- `get_train_data.py`: Downloads training images and annotations from `seshing/openfacades-dataset`
- `setup_data_config.py`: Creates data configuration JSON for InternVL training
- `setup_training_config.py`: Generates customized training script with your parameters

**Using your own data:** 
- Place images in `InternVL/internvl_chat/data/img/` directory
- Create JSONL annotations in `InternVL/internvl_chat/data/jsonl/train.jsonl` following [InternVL format](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html)

### Step 4: Run Fine-tuning

Execute the training process:

```bash
GPUS=1 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl3.0/2nd_finetune/internvl3_1b_dynamic_res_2nd_finetune_full_building.sh
```

**Training Configuration:**
- `GPUS=1`: Number of GPUs to use (adjust based on your hardware)  
- `PER_DEVICE_BATCH_SIZE=1`: Batch size per GPU device (increase if you have more GPU memory)
- Uses InternVL3-1B model as example

### Step 5: Monitor Training

- Training logs will be saved to the specified output directory
- Monitor GPU usage and training loss
- The process will save checkpoints periodically

## Configuration Files

- `get_train_data.py`: Downloads and extracts training data from Hugging Face repository
- `setup_data_config.py`: Creates data configuration JSON for training
- `setup_training_config.py`: Generates customized training script with your parameters
- Generated shell script: Ready-to-use training execution script

## Directory Structure

```
OpenFACADES/
├── train/                          # Training scripts
│   ├── get_train_data.py
│   ├── setup_data_config.py
│   ├── setup_training_config.py
│   └── requirements.txt
└── InternVL/
    └── internvl_chat/
        ├── pretrain/
        │   └── InternVL3-1B/           # Downloaded pre-trained model
        ├── data/
        │   ├── img/                    # Training images
        │   └── jsonl/
        │       └── train.jsonl         # Training annotations
        └── shell/
            ├── data/
            │   └── internvl_finetune_building.json  # Generated data config
            └── internvl3.0/2nd_finetune/
                └── internvl3_1b_*_building.sh      # Generated training script
```

## Notes

- Ensure your training data follows the InternVL chat format specification
- Adjust batch size and GPU count based on your hardware capacity
- Training time depends on dataset size and selected hyperparameters
- Monitor GPU memory usage and adjust batch size if needed
- The fine-tuned model will be saved to `InternVL/internvl_chat/finetuned/InternVL3-1B-finetuned/`