---
title: "Sample Similarity Loss for Robust Language Identification in Indian Languages"
collection: publications
permalink: /publication/2021-06-01-icassp-language-identification
excerpt: 'Novel similarity-based loss functions (CCSL and WSSL) for improving language embedding robustness in multilingual speech processing using BiLSTM architectures with multi-scale attention mechanisms.'
date: 2021-06-01
venue: 'IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)'
paperurl: 'https://ieeexplore.ieee.org/document/9414878'
citation: 'Kapoor, S., et al. (2021). &quot;Sample Similarity Loss for Robust Language Identification in Indian Languages.&quot; <i>ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)</i>.'
---

## Abstract

This research introduces novel similarity-based loss functions for robust automatic language identification (LID) in Indian languages. We propose Cross-Centroid Similarity Loss (CCSL) and Within Sample Similarity Loss (WSSL) that enhance the discriminative power of language embeddings extracted from bidirectional LSTM networks. These loss functions address the challenges of speaker variability, channel noise, and acoustic diversity in real-world multilingual speech processing.

**GitHub Repository**: [skapoor2024/MANAS-LAB-LSTM](https://github.com/skapoor2024/MANAS-LAB-LSTM)

## Key Contributions

### Novel Loss Functions

#### Cross-Centroid Similarity Loss (CCSL)
- **Objective**: Maximize inter-language discrimination by pushing language centroids apart
- **Mechanism**: Computes cosine similarity between sample embeddings and language-specific centroids
- **Innovation**: Dynamically updates centroids during training for adaptive learning
- **Impact**: Ensures clear separation between language clusters in embedding space

**Implementation Details**:
```python
# CCSL penalizes similarity to centroids of other languages
err_c = sum(cosine_similarity(u, centroid[other_lang]) 
            for other_lang != current_lang)
err = cross_entropy_loss + λ₁ * err_c / (num_languages - 1)
```

#### Within Sample Similarity Loss (WSSL)
- **Objective**: Improve consistency of embeddings from multi-scale temporal contexts
- **Mechanism**: Enforces similarity between embeddings from different temporal resolutions
- **Innovation**: Multi-scale attention combines high-resolution (20 frames) and long-context (50 frames) features
- **Impact**: Captures both fine-grained phonetic details and broader prosodic patterns

**Implementation Details**:
```python
# WSSL ensures embeddings from different scales are consistent
u1 = model1(high_res_context)  # 20 frames, step size 5
u2 = model2(long_context)      # 50 frames, step size 10
wssl = cosine_similarity(u1, u2)
err = cross_entropy_loss + λ₂ * wssl
```

### Multi-Scale Attention Architecture

The proposed Multi-Scale Attention (MSA) network processes speech at two temporal resolutions:

1. **High-Resolution Path**:
   - Look-back window: 20 frames
   - Step size: 5 frames
   - Captures fine phonetic details and transitions

2. **Long-Context Path**:
   - Look-back window: 50 frames (sub-sampled by 3)
   - Step size: 10 frames
   - Captures prosodic and rhythmic language patterns

3. **Attention Fusion**:
   - Bottleneck features (u₁, u₂) from both paths
   - Attention mechanism learns optimal combination
   - Final language prediction from fused representation

## Technical Implementation

### Network Architecture

**BiLSTM Feature Extractor**:
```
Input (80-dim bottleneck features)
    ↓
LSTM Layer 1 (256 units, bidirectional) → 512-dim
    ↓
LSTM Layer 2 (64 units, bidirectional) → 128-dim
    ↓
Attention Mechanism (100 units)
    ↓
Context Vector (128-dim language embedding)
    ↓
Classifier (6-9 language outputs)
```

**Key Components**:
- **Bottleneck Features**: 80-dimensional features from pre-trained DNN
- **Bidirectional Processing**: Captures both past and future context
- **Temporal Attention**: Automatically weights relevant time segments
- **Embedding Normalization**: Mean-variance normalization per utterance

### Training Strategy

**Loss Function Formulation**:
```
Total Loss = L_CE + α * L_CCSL + β * L_WSSL

where:
  L_CE    = Cross-Entropy Loss (language classification)
  L_CCSL  = Cross-Centroid Similarity Loss
  L_WSSL  = Within Sample Similarity Loss
  α = 0.5, β = 0.25 (empirically determined)
```

**Training Specifications**:
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.01
- **Batch Processing**: Single utterance per iteration
- **Epochs**: 25-300 (depending on dataset size)
- **Data Augmentation**: DCASE noise addition (beach, bus, city, train)

### Language Coverage

The system was trained and evaluated on **9 Indian languages**:
1. Assamese (asm)
2. Bengali (ben)
3. Gujarati (guj)
4. Hindi (hin)
5. Kannada (kan)
6. Malayalam (mal)
7. Manipuri (man)
8. Odia (odi)
9. Telugu (tel)

## Experimental Results

### Dataset Specifications

**IIT Madras Indian Language Dataset**:
- Training: 6 hours per language
- Channel-perturbed versions with DCASE background noise
- Multiple speakers per language
- Diverse acoustic conditions

**Evaluation Metrics**:
- Classification accuracy
- Cavg (Average cost) metric for detection cost function
- Confusion matrices for error analysis

### Performance Achievements

**CCSL Benefits**:
- Improved inter-language separation by pushing centroids apart
- Reduced false positives from acoustically similar languages
- Better generalization to unseen speakers

**WSSL Benefits**:
- 25% improvement in robustness to temporal variations
- Consistent embeddings across different speech rates
- Enhanced performance on short utterances

**Multi-Scale Attention**:
- Combines complementary information from different temporal scales
- Adapts to both short and long utterance scenarios
- Robust to speaking rate variations

## Implementation Details

### Code Organization

**Repository Structure**:
```
MANAS-LAB-LSTM/
├── base_ccsl_wssl/          # Core implementation
│   ├── nn_def.py            # Neural network definitions
│   ├── base_csl_4_l2.py     # CCSL training script
│   ├── msa_wsl2.py          # MSA-WSSL training
│   ├── utils.py             # Data processing utilities
│   ├── audio2bottleneck.py  # Feature extraction
│   └── spec_bnf.py          # Spectrogram processing
├── cavg_code/               # Evaluation metrics
│   ├── cavg.py              # Cavg calculation
│   └── tf_cavg_2.py         # TensorFlow metrics
├── Final_term_project_*.pdf # Project report
├── slt_2021.pdf            # CCSL paper
└── wssl_v6.pdf             # WSSL paper
```

### Key Technologies

- **Framework**: PyTorch 1.x
- **Audio Processing**: Librosa, NumPy
- **Feature Extraction**: MFCC, bottleneck features from pre-trained DNN
- **Data Format**: CSV files with 80-dimensional features
- **Encoding**: UTF-16 for multilingual support

## Research Impact

### Scientific Contributions

1. **Theoretical Foundation**: Novel loss functions grounded in metric learning principles
2. **Architectural Innovation**: Multi-scale attention for temporal speech modeling
3. **Empirical Validation**: Comprehensive experiments on low-resource Indian languages
4. **Reproducibility**: Open-source implementation with detailed documentation

## Evaluation and Metrics

### Cavg Metric

The **Cost Average (Cavg)** metric balances:
- **Miss probability**: Failing to detect target language
- **False alarm probability**: Incorrectly detecting non-target language
- **Language priors**: Realistic deployment scenarios

Implementation available in `cavg_code/cavg.py` with TensorFlow support.

### Confusion Analysis

Detailed confusion matrices reveal:
- Challenges with closely-related languages (e.g., Bengali-Assamese)
- Impact of acoustic similarity (e.g., Dravidian language family)
- Performance across different utterance lengths

## Publications and Resources

**Related Papers**:
- CCSL methodology: `slt_2021.pdf` (SLT 2021 Conference)
- WSSL approach: `wssl_v6.pdf` (Version 6)
- Comprehensive report: `Final_term_project_shantanu_kapoor_160907430_2.pdf`

**Code Access**: Complete implementation available at [GitHub Repository](https://github.com/skapoor2024/MANAS-LAB-LSTM)
