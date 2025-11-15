---
title: "COVID-19 and Lung Disease Classification from Chest X-Rays"
excerpt: "A deep learning computer vision project using ResNet50, VGG16, and GoogLeNet to classify chest X-ray images into four categories: COVID-19, Normal, Lung Opacity, and Viral Pneumonia, with multiple loss functions and data augmentation."
collection: portfolio
date: 2023-12-15
permalink: /portfolio/lung-disease-classification
---

## Overview

This project develops deep learning classifiers for automated diagnosis of lung diseases from chest X-ray images. Using the COVID-19 Radiography Database, the system classifies X-rays into four categories: COVID-19 positive, Normal, Lung Opacity (Non-COVID lung infection), and Viral Pneumonia.

The project implements and compares three state-of-the-art convolutional neural network architectures (ResNet50, VGG16, GoogLeNet) with multiple loss functions to address class imbalance and optimize classification performance for medical diagnosis.

**GitHub Repository**: [skapoor2024/lung-disease-classification](https://github.com/skapoor2024/lung-disease-classification)

## Key Features

###  Multi-Class Disease Classification
- **COVID-19 Detection**: Identification of COVID-19 positive cases from chest X-rays
- **Normal vs. Abnormal**: Distinction between healthy and diseased lungs
- **Lung Opacity Classification**: Detection of non-COVID lung infections
- **Viral Pneumonia Diagnosis**: Identification of viral pneumonia cases

###  Deep Learning Architectures
- **ResNet50**: Residual networks with skip connections for deep feature learning
- **VGG16**: Classic architecture with deep convolutional layers
- **GoogLeNet (Inception)**: Multi-scale feature extraction with inception modules
- **Transfer Learning**: Pre-trained ImageNet weights with fine-tuning

###  Advanced Loss Functions
- **Cross Entropy**: Standard classification loss baseline
- **Weighted Cross Entropy**: Class-balanced loss for imbalanced dataset
- **Focal Loss**: Down-weights easy examples to focus on hard cases
- **Learning Rate Scheduling**: Adaptive learning rate for optimal convergence

## Technical Implementation

### System Architecture

The project implements a comprehensive training pipeline with three CNN architectures:

1. **ResNet50**
   - 50-layer deep residual network
   - Skip connections prevent vanishing gradients
   - Dense layer customization for 4-class output
   - Best performer across all loss functions

2. **VGG16**
   - 16-layer architecture with small 3×3 filters
   - Deep convolutional feature extraction
   - Dense classifier head modification
   - Data augmentation integration

3. **GoogLeNet (Inception v1)**
   - Inception modules for multi-scale features
   - Efficient architecture with reduced parameters
   - Auxiliary classifiers for gradient flow
   - Competitive accuracy with lower complexity

### Key Technologies

- **Framework**: PyTorch with torchvision models
- **Dataset**: COVID-19 Radiography Database (21,165 X-ray images)
  - 3,616 COVID-19 positive
  - 10,192 Normal
  - 6,012 Lung Opacity
  - 1,345 Viral Pneumonia
- **Data Augmentation**: Rotation, flipping, scaling for generalization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Evaluation**: Accuracy, precision, recall, F1-score per class

### Training Pipeline

**Data Preprocessing**:
```python
- Image resizing to 224×224 pixels
- Normalization using ImageNet statistics
- Train-validation-test split (70-15-15)
- Augmentation: rotation, horizontal flip, brightness
```

**Loss Functions**:
- **Standard CE**: L = -Σ y_i log(ŷ_i)
- **Weighted CE**: L = -Σ w_i y_i log(ŷ_i) (class weights for imbalance)
- **Focal Loss**: L = -Σ α(1-ŷ_i)^γ y_i log(ŷ_i) (focus on hard examples)

**Notebooks Organization**:
- `resnet50/`: ResNet50 experiments with all loss functions
- `vgg16/`: VGG16 training with schedulers and augmentation
- `googlenet/`: GoogLeNet variants with focal and weighted loss
- `split_data.ipynb`: Dataset splitting and organization
- `generate_augmented_dataset.ipynb`: Data augmentation pipeline

## Experimental Results

### Performance Summary

Comprehensive evaluation across three architectures and multiple loss configurations:

**Best Overall Model**:
- **Architecture**: ResNet50
- **Configuration**: Dense layers + Learning rate scheduler + Focal loss
- **Validation Accuracy**: ~95%+ across all classes
- **Key Strength**: Best balance between COVID-19 detection and overall accuracy

**Architecture Comparison**:
1. **ResNet50**: Best overall performance, superior gradient flow through residual connections
2. **VGG16**: Strong performance with augmentation, deeper feature learning
3. **GoogLeNet**: Efficient computation with competitive accuracy

**Key Findings**:
1. Focal loss improved performance on underrepresented classes (Viral Pneumonia)
2. Weighted cross-entropy effectively addressed class imbalance
3. Learning rate scheduling crucial for convergence stability
4. Data augmentation significantly improved generalization
5. ResNet50's skip connections provided best feature learning

## Technical Highlights

### Class Imbalance Handling
- Weighted loss functions compensate for dataset imbalance
- Focal loss focuses training on hard-to-classify examples
- Data augmentation increases minority class representation
- Per-class performance metrics for balanced evaluation

### Transfer Learning Strategy
- Pre-trained ImageNet weights initialization
- Frozen early layers preserve general features
- Fine-tuned classifier heads for medical imaging
- Gradual unfreezing for domain adaptation

### Model Optimization
- Learning rate scheduling with StepLR or ReduceLROnPlateau
- Early stopping to prevent overfitting
- Batch normalization for training stability
- Dropout regularization in dense layers

### Key Contributions

- Comprehensive comparison of CNN architectures for medical imaging classification
- Systematic evaluation of loss functions for imbalanced medical datasets
- Transfer learning effectiveness validation for chest X-ray analysis
- Production-ready model with 95%+ accuracy for clinical deployment consideration

---

**Dataset**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

**Keywords**: Deep Learning, Medical Imaging, COVID-19 Detection, Chest X-Ray Classification, ResNet50, VGG16, GoogLeNet, Transfer Learning, Focal Loss
