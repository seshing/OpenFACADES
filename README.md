<p align="center">
  <img src="./figs/logo.png" alt="text" width="400">
  </p>

<p align="center">
  Preview: <a href="https://colab.research.google.com/drive/1hFKdHCBz9KJTDwCwWsMXOHH5wwOHmnyv?usp=sharing">[Google Colab]</a>
  </p>

# OpenFACADES
An Open Framework for Architectural Caption and Attribute Data Enrichment via Street View Imagery

## Overview
<p align="center">
<img src="./figs/workflow.jpg" alt="workflow" width="900">
 </p>
 
## To Do List
- [x] Release code for building data harmonization.
- [x] Release code for integrating building and street view imagery data.
- [x] Develop Google Colab tutorial (Preview).
- [ ] Release training data and global building dataset.
- [ ] Release installation guideline.
- [ ] Release fine-tuned model (1B, 2B).

## Installation

To install OpenFACADES, follow these steps:

1. Clone the repository:
  ```bash
  git clone https://github.com/seshing/OpenFACADES.git
  ```

2. Install the package and required dependencies:
  ```bash
  pip install -e OpenFACADES/.
  pip install -r OpenFACADES/requirements.txt
  ```

## What can our method do?

1. **Integrating multimodal crowdsourced data**: acquire building data from crowdsourced platforms and street view imagery for selected areas, and conduct isovist analysis to integrate them.

2. **Establishing building image data**: perform object detection to identify target buildings in panoramic images and reproject them back to a holistic perspective view, with image filtering functions to select high-quality building images.
![detect](./figs/detect_example.png)

3. **Developing datasets and multimodal models**: apply state-of-the-art multimodal large language models to annotate building images with multiple attributes, including building type, surface material, number of floors, and building age, and provide detailed descriptive captions.
<p align="center">
<img src="./figs/labeling.gif" alt="vlm" width="700">
 </p>

## Quick start

To acquire individual building images (Steps 1 & 2 above) for an area, you can simply run:
  ```bash
  python OpenFACADES/run.py \
    --bbox=[left,bottom,right,top] \
    --api_key='YOUR_MAPILLARY_API_KEY'
  ```
*Note: please check Mapillary has panoramic images available for the selected area.*

## Acknowledgement
We acknowledge the contributors of OpenStreetMap, Mapillary and other platforms for providing valuable open data resources and code that support street-level imagery research and applications.
