# Aerial Object Classification and Detection System

This project implements a Deep Learning solution for classifying and detecting aerial objects, specifically distinguishing between **Birds** and **Drones**.

## Project Overview
The system utilizes multiple deep learning approaches:
- **Custom CNN**: A tailored convolutional neural network architecture.
- **Transfer Learning**: Pre-trained models including ResNet50 and VGG16.
- **YOLOv8**: State-of-the-art object detection.
- **Web Application**: A Flask-based interface for real-time inference on images, videos, and GIFs.

## Features
- Multi-model ensemble for robust classification.
- Support for various file formats (JPG, PNG, GIF, MP4, etc.).
- Comprehensive CUDA diagnostic support for GPU acceleration.
- Automated result logging and visualization.

## Project Structure
- `app.py`: Flask web application for inference.
- `Bird_vs_Drone_Detection.ipynb`: Jupyter notebook for model training and analysis.
- `Dataset/`: Training and testing data.
- `Charts/`: Performance visualizations.
- `requirements.txt`: Project dependencies.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Run the Web App
```bash
python app.py
```

### Training
Open and run the `Bird_vs_Drone_Detection.ipynb` notebook to retrain models or perform analysis.

## Frameworks Used
- PyTorch
- OpenCV
- Flask
- YOLOv8
- Streamlit (for additional dashboards)
