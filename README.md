<p align="center">
  <img src="./figs/logo.png" alt="text" width="400">
  </p>

<!-- <p align="center">
  Tutorial preview: <a href="https://colab.research.google.com/drive/1hFKdHCBz9KJTDwCwWsMXOHH5wwOHmnyv?usp=sharing">[Google Colab]</a>
  </p> -->

# OpenFACADES
An Open Framework for Architectural Caption and Attribute Data Enrichment via Street View Imagery

## Overview
OpenFACADES is an open-source framework designed to enrich building profiles with objective attributes and semantic descriptors by leveraging multimodal crowdsourced data and large vision-language models. It provides tools for integrating diverse datasets, automating building facade detection, and generating detailed annotations at scale.

<p align="center">
<img src="./figs/overview.jpg" alt="overview" width="900">
 </p>

## Tutorial preview
**1. Retrieving building image data**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hFKdHCBz9KJTDwCwWsMXOHH5wwOHmnyv?usp=sharing)

**2. Building image labeling and captioning using VLM**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M01pYey72DiLQlOW6hnkb-J2OUlASnNC?usp=sharing)

## What can our method do?

1. **Integrating multimodal crowdsourced data**: acquire building data and street view imagery from crowdsourced platforms for selected areas, and conduct isovist analysis to integrate them.

2. **Retrieving building image data**: perform object detection to identify target buildings in panoramic images and reproject them back to a holistic perspective view, with image filtering functions to select high-quality building images.
![detect](./figs/detect_example.png)

<p align="center"> 
  <img src="./figs/dashboard.gif" alt="vlm" width="700">
  <br>
  <em>retrieving building image</em>
</p>

3. **Establishing dataset and multimodal models**: apply state-of-the-art multimodal large language models to annotate building images with multiple attributes, including building type, surface material, number of floors, and building age, and provide detailed descriptive captions.
<p align="center"> 
  <img src="./figs/labeling.gif" alt="vlm" width="700">
  <br>
  <em>(a) building attributes labeling</em>
</p>

<p align="center"> 
  <img src="./figs/captioning.gif" alt="vlm" width="700">
  <br>
  <em>(b) image captioning</em>
</p>


## To Do List
- [x] Release code for building data harmonization.
- [x] Release code for integrating building and street view imagery data.
- [x] Develop Google Colab tutorial for retriving building image data.
- [x] Release training code for fine-tuning InternVL models.
- [x] Release training data.
- [x] Develop Google Colab tutorial for building labeling and captioning.
- [x] Release fine-tuned model (1B, 2B).
- [ ] Integrate more SVI platforms into the framework.
- [ ]  Expand criteria for building image selection.

## Installation

To install OpenFACADES, follow these steps:

1. Clone the repository:
  ```bash
  git clone https://github.com/seshing/OpenFACADES.git
  ```

2. Install the package and required dependencies:
  ```bash
  conda create -n openfacades
  conda activate openfacades
  
  pip install -e OpenFACADES/.
  pip install -r OpenFACADES/requirements.txt
  ```
*Note:* The package used `pytorch` and `torchvision`, you may need to install them separately. Please refer to the [official website](https://pytorch.org/get-started/locally/) for installation instructions.


## Quick start

To acquire individual building images (Steps 1 & 2 above) for an area, you can simply run:
  ```bash
  python OpenFACADES/run.py \
    --bbox=[left,bottom,right,top] \
    --api_key='YOUR_MAPILLARY_API_KEY'
  ```
*Note: please check Mapillary has panoramic images available for the selected area.* 

**Example bbox:**  <br />
```[8.552,47.372,8.554,47.376]```: an area in Zurich, Switzerland;<br />
```[-81.382,28.540,-81.376,28.543]```: an area in Orlando, the US;<br />
```[-70.660,-33.442,-70.655,-33.437]```: an area in Santiago, Chile;<br />
```[-73.578,45.497,-73.569,45.502]```: an area in Montreal, Canada;<br />
```[37.618,55.758,37.628,55.763]```: an area in Moscow, Russia;<br />
```[25.273,54.684,25.283,54.687]```: an area in Vilnius, Lithuania.

**Output paths:**  <br />
building footprint: ```output/01_data/footprint.geojson```;  <br />
detected building images: ```output/02_img/individual_building```;  <br />
building image ids after filtering: ```output/02_img/individual_building_select.csv```.

## Model Training

To finetune InternVL models for building facade analysis and captioning, see our detailed training guide:

📖 [Fine-tuning Guide](train/README.md) — Instructions for training InternVL models on building data

🗂️ [OpenFACADES Training Dataset](https://huggingface.co/datasets/seshing/openfacades-dataset) — Training data on Hugging Face

## Use case
1. Liang, X., Cheng, S., Biljecki, F. (2025, June). Decoding Characteristics of Building Facades Using Street ViewImagery and Vision-Language Model. In 19th International Conference on Computational Urban Planning & Urban Management, CUPUM 2025. <br />
https://osf.io/abyqh/files/osfstorage/685400519a7097303ec89a95

2. Liang, X., Chang, J.H., Gao, S., Zhao, T. and Biljecki, F., 2024. Evaluating human perception of building exteriors using street view imagery. Building and Environment, 263, p.111875. <br />
https://doi.org/10.1016/j.buildenv.2024.111875

3. Lei, B., Liang, X. and Biljecki, F., 2024. Integrating human perception in 3D city models and urban digital twins. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 10, pp.211-218. <br /> https://isprs-annals.copernicus.org/articles/X-4-W5-2024/211/2024/isprs-annals-X-4-W5-2024-211-2024.html


## Citation
Please cite the following paper if you use `OpenFACADES` in a scientific publication:

```bibtex
@article{liang2025openfacades,
        title = {OpenFACADES: An open framework for architectural caption and attribute data enrichment via street view imagery},
        author = {Liang, Xiucheng and Xie, Jinheng and Zhao, Tianhong and Stouffs, Rudi and Biljecki, Filip},
        year = 2025,
        journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
        volume = {230},
        pages = {918--942},
        issn = {0924-2716}
        }
```

## Acknowledgement
We acknowledge the contributors of OpenStreetMap, Mapillary and other platforms for providing valuable open data resources and code that support street-level imagery research and applications.
