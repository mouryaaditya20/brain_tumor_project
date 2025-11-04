# Brain Tumor Classification using CNN

A deep learning project that classifies brain MRI scans into four categories: *Glioma, **Meningioma, **Pituitary Tumor, and **No Tumor* using Convolutional Neural Networks (CNN).

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🎯 Project Overview

This project implements a CNN-based classification model to detect and classify brain tumors from MRI images. The model achieves *96.34% validation accuracy* after 10 epochs of training.

### Key Features
- Multi-class classification (4 tumor types)
- Custom CNN architecture with dropout regularization
- Data augmentation and preprocessing
- Model training with visualization
- Single image prediction capability

## 📁 Dataset Structure




### Dataset Statistics
- *Training Images*: 5,712 images
- *Testing Images*: 1,311 images
- *Image Size*: 150x150 pixels
- *Classes*: 4 (glioma, meningioma, notumor, pituitary)

## 🏗 Model Architecture

python
Sequential Model:
- Input Layer: (150, 150, 3)
- Conv2D (32 filters) → ReLU → MaxPooling
- Conv2D (64 filters) → ReLU → MaxPooling
- Conv2D (128 filters) → ReLU → MaxPooling
- Flatten
- Dense (128 units) → ReLU
- Dropout (0.5)
- Dense (4 units) → Softmax


### Training Configuration
- *Optimizer*: Adam
- *Loss Function*: Categorical Crossentropy
- *Metrics*: Accuracy
- *Batch Size*: 32
- *Epochs*: 10
- *Image Augmentation*: Rescaling (1./255)

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Steps

1. *Clone the repository*
bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification


2. *Create a virtual environment* (recommended)
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. *Install dependencies*
bash
pip install -r requirements.txt


## 💻 Usage

### Training the Model

1. *Open the Jupyter notebook*
bash
jupyter notebook test.ipynb


2. *Run all cells* to:
   - Load and preprocess the dataset
   - Build and compile the model
   - Train the model
   - Save the trained model
   - Visualize training metrics

### Making Predictions

python
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("model/brain_tumor_classifier.h5")

# Define class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load and preprocess image
img_path = "path/to/your/image.jpg"
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

print(f"Predicted Tumor Type: {predicted_class}")


## 📊 Results

### Final Training Metrics (Epoch 10)
- *Training Accuracy*: 97.93%
- *Validation Accuracy*: 96.34%
- *Training Loss*: 0.0568
- *Validation Loss*: 0.1089

### Training Progress
The model shows excellent convergence with minimal overfitting:
- Consistent improvement across all epochs
- Validation accuracy closely follows training accuracy
- Achieved 96% validation accuracy

## 📦 Requirements


numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
jupyter>=1.0.0
ipykernel>=6.25.0
python-dateutil>=2.8.0


## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 📝 Notes

- The model is saved in HDF5 format (.h5)
- Images are automatically resized to 150x150 pixels
- The model expects RGB images (3 channels)
- Normalization is applied (pixel values divided by 255)

## 🔮 Future Improvements

- [ ] Implement data augmentation (rotation, zoom, flip)
- [ ] Try transfer learning with pre-trained models (VGG16, ResNet)
- [ ] Add class activation maps for interpretability
- [ ] Create a web interface for predictions
- [ ] Implement ensemble methods
- [ ] Add confidence scores to predictions


---

*⭐ If you find this project helpful, please consider giving it a star!*
