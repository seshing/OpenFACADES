
# Fine-tuning InternVL for Building Detection

This guide walks you through the process of fine-tuning InternVL models for building facade detection and analysis.

## Prerequisites

- InternVL model installed and configured
- Training data in the correct format (see [InternVL Chat Data Format](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html))
- GPU resources available for training

## Setup

The training scripts use relative paths from your InternVL installation directory. Make sure you have:

- **InternVL installation**: Download and set up InternVL3 model (see [InternVL3 Fine-tuning Guide](https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html))
- **Training data**: Organize your images and JSONL annotations in the `data/` directory
- **Working directory**: Run scripts from your InternVL installation root

## Training Steps

### Step 1: Prepare Training Data

You have two options for training data:

**Option A: Use your own data**
- Organize your images in `data/img/` directory
- Create JSONL annotations in `data/jsonl/train.jsonl` following [InternVL format](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html)

**Option B: Download sample training data**

```bash
python get_train_data.py
```

**What this does:**
- Downloads training images from Hugging Face `seshing/openfacades-dataset`
- Downloads training annotations (`jsonl/train.jsonl`)
- Automatically extracts and organizes the data
- Creates the proper directory structure for training

### Step 2: Setup Training Data Configuration

Configure your training data paths and parameters:

```bash
python setup_data_config.py
```

**What this does:**
- Creates the data configuration JSON file for InternVL training
- Specifies image directory and annotation file paths
- Calculates dataset length for training optimization

**Before running:** Update the paths in `setup_data_config.py`:
- `img_dir`: Path to your training images (default: `data/img`)
- `annotations`: Path to your JSONL annotation file (default: `data/jsonl/train.jsonl`)
- Verify your data structure matches the expected format

### Step 3: Configure Training Parameters

Customize the training parameters and generate the training script:

```bash
python setup_training_config.py
```

**What this does:**
- Modifies the InternVL training shell script with your parameters
- Sets training epochs, learning rate, and model paths
- Configures output directory and backbone freezing options
- Generates a customized training script

**Configuration options:**
- `model_size`: InternVL model size ('1', '2', or '8')
- `adjust_train_epochs`: Number of training epochs
- `adjust_learning_rate`: Learning rate for fine-tuning
- `adjust_freeze_backbone`: Whether to freeze the backbone during training

### Step 4: Run Fine-tuning

Execute the training process using the generated script:

```bash
cd InternVL/internvl_chat
GPUS=5 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl3.0/2nd_finetune/internvl3_2b_dynamic_res_2nd_finetune_full_building.sh
```

**Training Configuration:**
- `GPUS`: Number of GPUs to use (adjust based on your hardware)  
- `PER_DEVICE_BATCH_SIZE`: Batch size per GPU device
- The exact script name depends on your model size configuration from Step 2

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
InternVL/
├── models/
│   ├── InternVL3-2B/                # Pre-trained model
│   └── InternVL3-2B-finetuned/      # Output directory for fine-tuned model
├── internvl_chat/
│   └── shell/
│       ├── data/
│       │   └── internvl_finetune_building.json  # Generated data config
│       └── internvl3.0/2nd_finetune/
│           └── internvl3_*_building.sh          # Generated training script
└── data/
    ├── img/                         # Your training images
    └── jsonl/
        └── train.jsonl              # Your training annotations
```

## Notes

- Ensure your training data follows the InternVL chat format specification
- Adjust batch size and GPU count based on your hardware capacity
- Training time depends on dataset size and selected hyperparameters
- Monitor GPU memory usage and adjust batch size if needed
- The fine-tuned model will be saved to `InternVL/models/InternVL3-{model_size}B-finetuned/`