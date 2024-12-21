# Vital Sign Extraction from Video Using Text Detection and Recognition

## Overview

This project implements a pipeline for extracting vital signs (ECG, SpO2, BP) from a video feed using text detection and recognition techniques. The system processes video frames to detect text regions, recognize the text, and extract vital signs. The extracted data is then saved to a CSV file for further analysis. The project leverages TensorFlow for deep learning and OpenCV for video processing.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)
- [Limitations](#limitations)
- [Proposed Improvements](#proposed-improvements)
- [License](#license)

---

## Features

- **Text Detection**: Detects regions of text within video frames using a convolutional neural network (CNN).
- **Text Recognition**: Recognizes text within the detected regions using a Bidirectional GRU-based neural network.
- **Vital Sign Extraction**: Parses the recognized text to extract vital signs like ECG, SpO2, and BP using regular expressions.
- **CSV Export**: Saves the extracted data into a CSV file for further analysis.

---

## Installation

To run this project on your local machine, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/NipunGuleria/vital-sign-extraction.git
cd vital-sign-extraction
```

### 2. Create a virtual environment:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare your video file:
Ensure your video is in the correct format (e.g., `.mp4`), and place it in the root directory of the project.

### 2. Run the pipeline:

```bash
python process_video.py --video_path path/to/video.mp4 --output_csv_path output_vital_signs.csv
```

This will process the video and save the extracted vital signs data into `output_vital_signs.csv`.

### 3. Customizing Models:
- **Text Detection Model**: This model detects the regions in the video frames where text appears.
- **Text Recognition Model**: This model reads the detected text regions and recognizes the text content.

Both models are defined in the `models.py` file. You can customize the models or integrate pre-trained models for better performance.

---
## Model Details

### 1. **Text Detection Model**:
- **Architecture**: Convolutional layers with max pooling followed by a fully connected layer.
- **Output**: A set of bounding boxes (x, y, w, h) for detected text regions in the video frames.

### 2. **Text Recognition Model**:
- **Architecture**: A sequence model with convolutional layers followed by Bidirectional GRU layers for text recognition.
- **Output**: A sequence of characters predicted from the detected text regions.

### 3. **Vital Sign Extraction**:
- After recognizing the text, the system uses regular expressions to extract specific vital signs (ECG, SpO2, BP) from the recognized text.

---

## Performance Metrics

The models' performance can be evaluated using the following metrics:

- **Accuracy**: Percentage of correctly detected and recognized vital signs.
- **Precision**: Ratio of true positive detections to all detected vital signs.
- **Recall**: Ratio of true positive detections to all actual vital signs.
- **F1-Score**: Harmonic mean of precision and recall.

**Note**: Performance metrics will be provided once the models are trained on a labeled dataset.

---

## Limitations

- **Model Training**: The models need to be trained on a large and diverse dataset of video frames with annotated text. Performance will depend on the quality of the training data.
- **Text Quality**: The system may struggle with low-quality, distorted, or noisy text.
- **Bounding Box Detection**: The accuracy of text recognition depends on the precision of the bounding boxes provided by the detection model.

---

## Proposed Improvements

- **Better Text Detection**: Use more advanced models like YOLO or Faster R-CNN for more precise text region detection.
- **Text Recognition Enhancement**: Implement CRNN (Convolutional Recurrent Neural Networks) for improved text recognition accuracy.
- **Stabilized Video Feed Interpretation**: Integrate temporal models such as Optical Flow or tracking algorithms to improve text consistency across frames.
- **Data Augmentation**: Apply augmentation techniques to improve model robustness to different video conditions.

---

## Acknowledgements

- TensorFlow: An open-source machine learning framework.
- OpenCV: A powerful library for computer vision tasks.

---

Feel free to modify the repository link, video file paths, and model training details as required. This `README.md` will provide users with a complete understanding of how to use, set up, and improve the system.
