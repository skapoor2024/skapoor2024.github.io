---
title: "Medical Image Segmentation Using Deep Learning"
excerpt: "A comprehensive medical image segmentation project implementing U-Net, ResU-Net, and TransUNet architectures for COVID-19 lung infection segmentation and cardiac ventricle segmentation using multiple loss functions."
collection: portfolio
date: 2023-10-20
permalink: /portfolio/medical-image-segmentation
---

## Overview

This project implements advanced deep learning architectures for medical image segmentation on two critical tasks: COVID-19 lung infection detection from chest X-rays and cardiac ventricle segmentation from MRI scans. The system employs multiple state-of-the-art architectures including U-Net, Residual U-Net (ResU-Net), and Transformer U-Net (TransUNet) with various loss functions optimized for medical imaging.

Using the COVID-19 Radiography Database and ACDC (Automated Cardiac Diagnosis Challenge) dataset, the project performs pixel-level segmentation to identify infected lung regions and delineate left and right ventricles for clinical diagnosis support.

**GitHub Repository**: [skapoor2024/DL_MIA_ass_3](https://github.com/skapoor2024/DL_MIA_ass_3)

## Key Features

###  Medical Segmentation Tasks
- **COVID-19 Lung Segmentation**: Pixel-wise infection region identification in chest X-rays
- **Cardiac Ventricle Segmentation**: Left and right ventricle boundary detection in cardiac MRI
- **Multi-Dataset Training**: COVID-19 Radiography Database and ACDC cardiac imaging
- **Clinical Metrics**: Dice coefficient, IoU, and Hausdorff distance evaluation

### ️ Deep Architecture Variants
- **U-Net**: Classic encoder-decoder with skip connections for spatial preservation
- **ResU-Net**: Residual blocks for deeper networks and gradient flow
- **TransU-Net**: Vision Transformer encoder with U-Net decoder for global context
- **Custom Implementations**: Built from scratch with PyTorch for full control

###  Specialized Loss Functions
- **Binary Cross-Entropy**: Standard pixel-wise classification loss
- **Soft Dice Loss**: Direct optimization of Dice coefficient for overlap maximization
- **Focal Loss**: Addresses class imbalance by focusing on hard examples
- **Combined Losses**: Hybrid approaches for balanced optimization

## Technical Implementation

### System Architecture

The project implements three segmentation architectures with different inductive biases:

1. **U-Net**
   - Encoder-decoder architecture with skip connections
   - Double convolution blocks with batch normalization
   - MaxPooling downsampling and transposed convolution upsampling
   - Crop-and-concatenate for feature fusion

2. **ResU-Net**
   - Residual blocks replace standard convolutions
   - Identity mappings enable deeper networks
   - Improved gradient flow for training stability
   - Better feature reuse across network depth

3. **TransU-Net**
   - Vision Transformer (ViT) encoder for global receptive field
   - Patch embedding with positional encoding
   - Multi-head self-attention for long-range dependencies
   - U-Net decoder for spatial resolution recovery

### Key Technologies

- **Framework**: PyTorch with custom architecture implementations
- **Datasets**: 
  - COVID-19 Radiography Database (lung infection masks)
  - ACDC Dataset (cardiac MRI with ventricle annotations)
- **Metrics**: Dice Score, IoU, Hausdorff Distance, Pixel Accuracy
- **Augmentation**: Random rotation, flipping, intensity transforms
- **Optimization**: Adam optimizer with learning rate scheduling

### Model Components

**Core Modules** (from `model.py`, `resunet.py`, `transunet.py`, `vit.py`):
```python
- DoubleConv: Dual convolution with BatchNorm and ReLU
- DownSample: MaxPooling for spatial reduction
- UpSample: Transposed convolution for upsampling
- CropAndConcat: Feature map fusion from encoder
- ResidualBlock: Skip connections for ResU-Net
- ViTEncoder: Transformer encoder for TransU-Net
- MultiHeadAttention: Self-attention mechanism
```

**Loss Functions**:
- Soft Dice Loss: `L = 1 - (2|X∩Y|)/(|X|+|Y|)`
- Focal Loss: `L = -α(1-p)^γ log(p)` with γ=2
- BCE + Dice: Hybrid for stable training

## Experimental Results

### Performance Summary

Comprehensive evaluation across architectures and loss functions:

**Task 1: COVID-19 Lung Segmentation**
- **Best Model**: U-Net + Soft Dice Loss
- **Dice Score**: ~0.92 on validation set
- **Configuration**: BCE baseline → Focal loss → Soft Dice (best)
- **Insight**: Soft Dice directly optimizes segmentation overlap

**Task 2: Cardiac Ventricle Segmentation (Single)**
- **Best Model**: TransU-Net
- **Dice Score**: ~0.89 for ventricle delineation
- **Advantage**: Global attention captures cardiac shape context
- **Performance**: Superior boundary localization

**Task 3: Multi-Class Cardiac Segmentation**
- **Best Model**: ResU-Net
- **Classes**: Background, Left Ventricle, Right Ventricle
- **Dice Score**: ~0.87 average across classes
- **Strength**: Residual connections handle complex boundaries

**Key Findings**:
1. Soft Dice loss outperformed BCE and Focal loss for segmentation
2. TransU-Net excelled on tasks requiring global shape understanding
3. ResU-Net provided best balance for multi-class segmentation
4. Batch normalization crucial for training stability
5. Hausdorff distance validated boundary accuracy

### Visualizations

![U-Net Training Curves](https://raw.githubusercontent.com/skapoor2024/DL_MIA_ass_3/main/assignment_3_send/asgn_runs_graph/model_1_runs.png)
*U-Net training metrics showing convergence with BCE loss*

![Soft Dice Loss Performance](https://raw.githubusercontent.com/skapoor2024/DL_MIA_ass_3/main/assignment_3_send/asgn_runs_graph/model_3_soft_dice_runs.png)
*Improved segmentation performance with Soft Dice loss*

![TransU-Net Results](https://raw.githubusercontent.com/skapoor2024/DL_MIA_ass_3/main/assignment_3_send/asgn_runs_graph/model_q2_transunet_runs.png)
*TransU-Net training curves demonstrating transformer effectiveness*

![Multi-Class Cardiac Segmentation](https://raw.githubusercontent.com/skapoor2024/DL_MIA_ass_3/main/assignment_3_send/asgn_runs_graph/model_q3_resunet.png)
*ResU-Net performance on multi-class ventricle segmentation*

## Technical Highlights

### Architecture Design
- Custom PyTorch implementations from scratch
- Modular design with reusable components
- Batch normalization for training stability
- Reflection padding to preserve boundary information

### Loss Function Engineering
- Soft Dice loss for direct metric optimization
- Focal loss with γ=2 for hard example mining
- Hybrid losses for balanced training dynamics
- Per-class weighting for multi-class scenarios

### Evaluation Metrics
- Dice coefficient for overlap quantification
- Hausdorff distance for boundary accuracy
- IoU (Jaccard Index) for region similarity
- Pixel-wise accuracy for overall performance

### Key Contributions

- Comprehensive comparison of U-Net variants for medical segmentation
- Systematic loss function evaluation for medical imaging tasks
- TransU-Net implementation demonstrating transformer effectiveness
- Production-ready architectures with detailed training analysis

---

**Documentation**: [Technical Report (PDF)](https://github.com/skapoor2024/DL_MIA_ass_3/blob/main/assignment_3_send/assignment_3_readme.pdf)

**Keywords**: Medical Image Segmentation, U-Net, ResU-Net, TransU-Net, COVID-19, Cardiac MRI, Soft Dice Loss, Focal Loss, Deep Learning
