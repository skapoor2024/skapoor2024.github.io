---
title: "Speech Denoising Using Adaptive Filtering (APA Family)"
excerpt: "An adaptive filtering project for speech enhancement and interference cancellation using NLMS and APA algorithms to denoise speech signals corrupted by background noise."
collection: portfolio
date: 2023-03-15
permalink: /portfolio/speech-denoising-apa
---

## Overview

This project implements speech denoising and interference cancellation using adaptive filtering algorithms from the Affine Projection Algorithm (APA) family. The system removes background noise from corrupted speech signals through adaptive filter design, achieving significant noise reduction while preserving speech quality.

The project compares multiple adaptive algorithms including Normalized Least Mean Squares (NLMS) and Affine Projection Algorithms (APA-2), evaluating their performance using Echo Return Loss Enhancement (ERLE) and Normalized Mean Square Error (NMSE) metrics.

**GitHub Repository**: [skapoor2024/denoising_speech_APA](https://github.com/skapoor2024/denoising_speech_APA)

## Key Features

### ️ Adaptive Filtering Algorithms
- **NLMS (Normalized LMS)**: Normalized step-size for stable convergence
- **APA-2 (Affine Projection Algorithm)**: Parallel weight updates for faster convergence
- **RLS (Recursive Least Squares)**: Exponentially weighted least squares optimization
- **Filter Order Optimization**: Systematic evaluation of filter lengths from 2 to 30

###  Performance Metrics
- **ERLE (Echo Return Loss Enhancement)**: Measures noise suppression effectiveness
- **NMSE (Normalized Mean Square Error)**: Quantifies prediction accuracy
- **Frequency Response Analysis**: Filter characteristics in frequency domain
- **Spectrogram Comparison**: Visual assessment of denoising quality

###  Parameter Optimization
- **Step Size Tuning**: Grid search over learning rates for optimal convergence
- **Filter Order Analysis**: Performance vs. computational complexity tradeoff
- **Convergence Tracking**: Weight evolution and error monitoring
- **Contour Plots**: 2D visualization of parameter space

## Technical Implementation

### System Architecture

The project implements three adaptive filtering approaches:

1. **NLMS Filter**
   - Normalized step-size: μ/(X^T X)
   - Optimal parameters: Filter order=5, μ=0.003
   - Stable convergence for non-stationary signals
   - Best overall ERLE performance

2. **APA-2 Filter**
   - Affine projection with 2 parallel updates
   - Faster convergence than NLMS
   - Higher computational complexity
   - Effective for correlated input signals

3. **RLS Filter**
   - Exponential forgetting factor: λ=0.9999
   - Recursive covariance matrix updates
   - Fast convergence with higher complexity
   - Optimal for short data sequences

### Key Technologies

- **Python**: NumPy, SciPy for signal processing
- **Audio Processing**: WAV file I/O and manipulation
- **Visualization**: Matplotlib for spectrograms and plots
- **Notebook**: Jupyter for experimentation
- **Implementation**: Modular `mlts_func.py` with filter functions

### Algorithm Details

**NLMS Update Rule**:
```python
w(n+1) = w(n) + (μ * e(n) * x(n)) / (x^T(n) * x(n))
```

**Core Functions** (`mlts_func.py`):
- `nlms()`: Normalized LMS adaptive filter
- `lms()`: Standard LMS implementation
- `rls()`, `rls_2()`: Recursive Least Squares variants
- `err_wav()`: Normalized MSE computation
- `speech_norm()`: Audio signal normalization

## Experimental Results

### Performance Summary

The project evaluated all algorithms on real speech data corrupted with background noise:

**Best Configuration**:
- **Algorithm**: NLMS
- **Filter Order**: 5 taps
- **Step Size**: μ=0.003
- **ERLE**: Maximum enhancement achieved
- **Quality**: Clean speech recovery with minimal distortion

**Key Findings**:
1. NLMS with filter order 5 achieved optimal ERLE vs. complexity balance
2. Step size μ=0.003 provided best convergence-stability tradeoff
3. APA-2 showed faster initial convergence but similar steady-state performance
4. Filter orders beyond 5 showed diminishing returns

### Visualizations

![NLMS Best Performance](https://raw.githubusercontent.com/skapoor2024/denoising_speech_APA/main/best_nlms_5_0.003.png)
*NLMS denoising results with optimal parameters (filter order=5, μ=0.003)*

![ERLE vs Filter Order](https://raw.githubusercontent.com/skapoor2024/denoising_speech_APA/main/ERLE%20vs%20Filter%20order.png)
*ERLE performance across different filter orders showing optimal point at 5 taps*

![Spectrogram Comparison](https://raw.githubusercontent.com/skapoor2024/denoising_speech_APA/main/spec_5_nlms.png)
*Spectrogram showing noise reduction: original, noisy, and denoised speech*

![Parameter Contour](https://raw.githubusercontent.com/skapoor2024/denoising_speech_APA/main/contour.png)
*2D contour plot of ERLE vs. filter order and step size*

## Technical Highlights

### Filter Order Optimization
- Systematic evaluation from 2 to 30 taps
- ERLE vs. computational cost analysis
- Optimal point identified at 5 taps
- Diminishing returns beyond optimal order

### Step Size Analysis
- Grid search over learning rates [0.001, 0.01]
- Convergence speed vs. stability tradeoff
- μ=0.003 achieved best performance
- Stability verified through weight convergence

### Frequency Domain Analysis
- Filter frequency response visualization
- Desired vs. error signal comparison
- Noise spectrum characterization
- Clean speech spectrum recovery

### Key Contributions

- Comparative analysis of NLMS, APA, and RLS for speech denoising
- Optimal parameter selection through systematic grid search
- Filter order vs. performance tradeoff quantification
- Real audio signal validation with spectrogram analysis

---

**Documentation**: [Technical Report (PDF)](https://github.com/skapoor2024/denoising_speech_APA/blob/main/project_1.pdf)

**Keywords**: Speech Denoising, Adaptive Filtering, NLMS, APA, RLS, Interference Cancellation, ERLE, Audio Signal Processing
