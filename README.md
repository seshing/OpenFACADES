<p align="center">
  <img src="./figs/logo.png" alt="text" width="300">
  </p>

# OpenFACADES
An Open Framework for Architectural Caption and Attribute Data Enrichment via Street View Imagery

## Overview
![text](./figs/workflow.jpg)

## To Do List
- [x] Release code for building data harmonization.
- [x] Release code for integrating building and street view imagery data.
- [x] Develop Google Colab tutorial (Preview).
- [ ] Release training data and global building dataset.
- [ ] Release installation guideline.
- [ ] Release fine-tuned model (1B, 2B).

## What can our method do?

1. **Integrating multimodal crowdsourced data**: acquire building data from crowdsourced platforms and street view imagery for selected areas, and conduct isovist analysis to integrate them.

2. **Establishing building image data**: perform object detection to identify target buildings in panoramic images and reproject them back to a holistic perspective view, with image filtering functions to select high-quality building images.
![detect](./figs/detect_example.png)

3. **Developing datasets and multimodal models**: apply state-of-the-art multimodal large language models to annotate building images with multiple attributes, including building type, surface material, number of floors, and building age, and provide detailed descriptive captions.
![vlm](./figs/labeling.gif)

## Acknowledgement
We acknowledge the contributors of OpenStreetMap, Mapillary and other platforms for providing valuable open data resources and code that support street-level imagery research and applications.
