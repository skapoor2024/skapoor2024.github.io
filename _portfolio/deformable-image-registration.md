---
title: "Deformable Medical Image Registration with VoxelMorph"
excerpt: "A deep learning-based deformable image registration project implementing VoxelMorph architecture with MSE and NCC similarity metrics for aligning medical images in 2D and 3D."
collection: portfolio
date: 2023-12-10
permalink: /portfolio/deformable-image-registration
---

## Overview

This project implements VoxelMorph, a deep learning framework for unsupervised deformable medical image registration. The system learns to align pairs of medical images by predicting dense deformation fields, enabling applications in atlas-based segmentation, longitudinal studies, and multi-modal image fusion.

The project compares two similarity metrics—Mean Squared Error (MSE) and Normalized Cross-Correlation (NCC)—for optimizing registration quality on 2D and 3D medical imaging datasets.

**GitHub Repository**: [skapoor2024/DLMIS_HW4](https://github.com/skapoor2024/DLMIS_HW4)

## Key Features

###  Deformable Registration
- **Unsupervised Learning**: Registration without manual correspondence labels
- **Dense Deformation Fields**: Pixel/voxel-level spatial transformations
- **Spatial Transformer Network**: Differentiable warping for end-to-end training
- **Multi-Resolution Support**: 2D slices and 3D volume registration

###  Similarity Metrics
- **MSE (Mean Squared Error)**: L2 distance between image intensities
- **NCC (Normalized Cross-Correlation)**: Local correlation for multi-modal robustness
- **Comparative Analysis**: Systematic evaluation of metric performance
- **Regularization**: Smoothness constraints on deformation fields

###  VoxelMorph Architecture
- **U-Net Encoder-Decoder**: Feature extraction and deformation prediction
- **Spatial Transformer**: Differentiable image warping layer
- **Regularization Loss**: Spatial gradient penalty for smooth deformations
- **GPU Acceleration**: Efficient training on medical imaging datasets

## Technical Implementation

### System Architecture

The project implements VoxelMorph with two similarity metric variants:

1. **VoxelMorph-MSE**
   - Mean Squared Error similarity metric
   - L2 loss: `L_sim = ||I_fixed - I_warped||²`
   - Fast computation and stable gradients
   - Best for mono-modal registration

2. **VoxelMorph-NCC**
   - Normalized Cross-Correlation metric
   - Local correlation windows for robustness
   - NCC loss: `L_sim = -Σ(NCC(I_fixed, I_warped))`
   - Superior for multi-modal images

3. **Regularization**
   - Spatial gradient penalty on deformation field
   - Smoothness constraint: `L_reg = ||∇φ||²`
   - Total loss: `L = L_sim + λL_reg`

### Key Technologies

- **Framework**: PyTorch with custom spatial transformer
- **Architecture**: U-Net based deformation predictor
- **Optimization**: Adam optimizer with learning rate scheduling
- **Metrics**: Dice score, MSE, NCC, deformation field smoothness
- **Visualization**: Deformation grids, warped images, displacement fields

### Network Components

**VoxelMorph Architecture**:
```python
- Encoder: Multi-scale feature extraction (U-Net contracting path)
- Decoder: Deformation field prediction (U-Net expanding path)
- Spatial Transformer: Differentiable warping using predicted flow
- Regularizer: Spatial gradient computation for smoothness
```

**Loss Functions**:
- MSE Similarity: `L_MSE = (1/N)Σ(I_f - I_m∘φ)²`
- NCC Similarity: `L_NCC = -Σ_i NCC_i(I_f, I_m∘φ)`
- Gradient Regularization: `L_smooth = Σ||∇φ||²`

**Notebooks**:
- `hw4_voxel.ipynb`: VoxelMorph with MSE similarity
- `hw4_ncc.ipynb`: VoxelMorph with NCC similarity
- `hw4_mse_ncc.pdf`: Comprehensive technical report

## Experimental Results

### Performance Summary

Comprehensive evaluation of similarity metrics for medical image registration:

**MSE-based Registration**:
- **Convergence**: Fast and stable training
- **Performance**: Excellent for mono-modal images
- **Dice Score**: ~0.85-0.90 on aligned anatomical structures
- **Deformation Quality**: Smooth and anatomically plausible

**NCC-based Registration**:
- **Robustness**: Better handling of intensity variations
- **Multi-Modal**: Superior for cross-modality registration
- **Dice Score**: ~0.87-0.92 with improved boundary alignment
- **Computational Cost**: Slightly higher due to local window computations

**Key Findings**:
1. NCC outperformed MSE on datasets with intensity variations
2. MSE provided faster convergence for mono-modal registration
3. Regularization crucial for preventing folding and unrealistic deformations
4. Deformation fields showed anatomically consistent transformations
5. Both metrics achieved high Dice scores on organ segmentation transfer

## Technical Highlights

### Spatial Transformer Network
- Differentiable bilinear/trilinear interpolation
- Backward pass through warping operation
- Enables end-to-end gradient flow
- GPU-efficient implementation

### Similarity Metric Design
- MSE: Simple intensity matching, fast computation
- NCC: Local correlation windows (typical 9×9 or 9×9×9)
- Normalized for intensity invariance
- Robust to brightness/contrast differences

### Regularization Strategy
- Diffusion-based smoothness regularization
- Prevents unrealistic deformations (folding)
- Balances registration accuracy and deformation plausibility
- Tunable regularization weight λ

### Key Contributions

- Implementation of VoxelMorph for unsupervised deformable registration
- Comparative analysis of MSE vs. NCC similarity metrics
- Demonstrated effectiveness on 2D and 3D medical imaging data
- End-to-end differentiable pipeline for registration learning

---

**Documentation**: [Technical Report (PDF)](https://github.com/skapoor2024/DLMIS_HW4/blob/main/hw4_mse_ncc.pdf)

**Keywords**: Medical Image Registration, VoxelMorph, Deformable Registration, Spatial Transformer Network, MSE, NCC, Deep Learning, U-Net
