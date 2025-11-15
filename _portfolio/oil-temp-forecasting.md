---
title: "Oil Temperature Forecasting for Electric Load Prediction"
excerpt: "A time series forecasting project comparing adaptive filtering algorithms (LMS, KLMS, QKLMS) to predict electric consumption from transformer oil temperature data using the ETTm2 dataset."
collection: portfolio
date: 2023-04-05
permalink: /portfolio/oil-temp-forecasting
---

## Overview

This project predicts future electric consumption using oil temperature data from power transformers. With rising energy prices, accurate load forecasting has become critical for grid management. Since oil temperature strongly correlates with electrical load, it provides valuable insights for consumption prediction.

The project compares linear (LMS) and non-linear (KLMS, QKLMS) adaptive filters trained on the ETTm2 time series dataset, evaluating performance using both Mean Square Error (MSE) and Maximum Correlation Criterion (MCC) cost functions.

**GitHub Repository**: [skapoor2024/oil_temp_forecasting](https://github.com/skapoor2024/oil_temp_forecasting)

## Key Features

###  Adaptive Filtering Algorithms
- **LMS (Linear Mean Squares)**: Traditional linear filtering for baseline predictions
- **KLMS (Kernel LMS)**: Non-linear filtering using Gaussian kernels to capture complex patterns
- **QKLMS (Quantized KLMS)**: Memory-efficient variant that reduces storage by 60-80%
- **Adaptive Learning**: Online weight updates for real-time consumption forecasting

###  Dual Cost Functions
- **MSE (Mean Squared Error)**: Standard error minimization for prediction accuracy
- **MCC (Maximum Correlation Criterion)**: Information-theoretic approach using error distribution
- **Performance Gain**: MCC outperforms MSE by ~25% across all models
- **Robustness**: MCC provides better outlier handling through kernel smoothing

###  Multi-Step Forecasting
- **Single-Step Prediction**: Next hour consumption forecasting
- **Extended Forecasting**: Up to 10 steps ahead with error analysis
- **Trajectory Prediction**: Continuous forecasting with stability evaluation
- **Generalization Testing**: Performance validation on unseen test data

## Technical Implementation

### System Architecture

The project implements three adaptive filtering approaches:

1. **LMS Filter**
   - Linear weight updates with gradient descent
   - Optimal parameters: η=0.003 (MSE), η=0.0007 (MCC)
   - 8 weights for temporal pattern capture
   - Fast convergence with low computational cost

2. **KLMS Filter**
   - Gaussian kernel: ```K(x₁,x₂) = exp(-||x₁-x₂||²/h²)```
   - Optimal parameters: η=0.109, h=0.922 (MSE)
   - Stores all training samples as centers
   - Superior accuracy for complex patterns

3. **QKLMS Filter**
   - Vector quantization with threshold ε
   - Optimal parameters: h=0.4, η=0.9, ε=0.2 (MSE)
   - Stores only ~20-40% of training samples
   - Balances accuracy and memory efficiency

### Key Technologies

- **Python**: NumPy, SciPy, Pandas for numerical computing
- **Dataset**: ETTm2 (17,420 hourly samples, 2 years)
- **Visualization**: Matplotlib for plots and heatmaps
- **Notebooks**: Jupyter for experiments and analysis
- **Implementation**: Modular `algo_func.py` with reusable functions

### Algorithm Details

**Cost Functions**:
```python
MSE:  L = (1/N)Σ(y - ŷ)²
MCC:  L = -E[exp(-e²/2σ²)]
```

**Core Functions** (`algo_func.py`):
- `lms_mse()`, `lms_mcc()`: Linear filtering
- `KLMS_mse_2()`, `KLMS_mcc_2()`: Kernel filtering
- `QKLMS_mse_2()`, `QKLMS_mcc_2()`: Quantized kernel filtering
- `learn_curve()`: Learning curve generation
- `plot3d()`: Parameter sweep visualization

## Experimental Results

### Performance Summary

The project evaluated all models on the ETTm2 dataset (10,000 train / 1,000 validation / 6,000 test samples):

**Best Models**:
- **LMS-MCC**: Outperformed LMS-MSE on test set, revealed weight periodicity
- **KLMS-MCC**: 35% improvement over LMS-MSE baseline, best overall accuracy
- **QKLMS-MCC**: 32% improvement with 60-80% memory reduction

**Key Findings**:
1. MCC consistently outperforms MSE by ~25% across all algorithms
2. KLMS shows superior multi-step prediction and generalization
3. QKLMS achieves near-KLMS accuracy with fraction of memory
4. Weight tracks reveal temporal periodicity in consumption patterns

### Visualizations

![Test Set Predictions](https://raw.githubusercontent.com/skapoor2024/oil_temp_forecasting/main/figures/test_set_predictions.png)
*Prediction comparison showing KLMS-MCC superiority on test data*

![Weight Tracks](https://raw.githubusercontent.com/skapoor2024/oil_temp_forecasting/main/figures/lms_mcc_wt_tracks.png)
*LMS weight evolution revealing convergence patterns and periodicity*

![KLMS Predictions](https://raw.githubusercontent.com/skapoor2024/oil_temp_forecasting/main/figures/pred_klms.png)
*KLMS predictions demonstrating close alignment with actual values*

## Technical Highlights

### Parameter Optimization
- Grid search over learning rates and kernel sizes
- 3D heatmaps for optimal configuration identification
- Separate tuning for MSE and MCC cost functions
- Validation set used for hyperparameter selection

### Multi-Step Forecasting
- Extended predictions up to 10 steps ahead
- Error accumulation analysis for stability assessment
- KLMS maintains accuracy while LMS diverges
- Histogram analysis of prediction error distributions

### Memory Efficiency
- QKLMS reduces memory by 60-80% vs. KLMS
- Threshold parameter ε controls quality-memory tradeoff
- Maintains prediction accuracy with fewer centers

## Project Impact

This research demonstrates practical advances in electric load forecasting with real-world applications:

- **Energy Management**: Accurate predictions enable efficient grid balancing and resource allocation
- **Cost Reduction**: Better forecasting helps utilities respond to fluctuating energy prices
- **Algorithm Innovation**: MCC outperforms traditional MSE by 25% across all models
- **Practical Deployment**: QKLMS enables deployment on memory-constrained systems

### Key Contributions

- Comprehensive comparison of linear vs. non-linear adaptive filters for time series forecasting
- Empirical validation of MCC's superiority over MSE for prediction tasks
- Memory-efficient QKLMS achieving 80% storage reduction without accuracy loss

---

**Documentation**: [Technical Report (PDF)](https://github.com/skapoor2024/oil_temp_forecasting/blob/main/mlts_project_2.pdf)

**Keywords**: Time Series Forecasting, Adaptive Filtering, Kernel Methods, LMS, KLMS, QKLMS, Electric Load Prediction
