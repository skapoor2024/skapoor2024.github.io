---
title: "CLIP-Deep-Driven FiLM: Text-Guided Multi-Organ Segmentation with SwinUNETR"
collection: publications
permalink: /publication/2023-08-01-text-guided-segmentation
excerpt: 'Novel CLIP-driven Feature-wise Linear Modulation (FiLM) framework integrating text embeddings with SwinUNETR for text-guided 3D medical image segmentation across multiple organ datasets.'
date: 2024-11-07
venue: 'Research Project - Dr. Gong Laboratory'
paperurl: 'https://github.com/skapoor2024/swinunetr_deep_film_MSD'
citation: 'Kapoor, S. (2024). &quot;CLIP-Deep-Driven FiLM: Text-Guided Multi-Organ Segmentation with SwinUNETR.&quot; <i>University of Florida, Dr. Gong Laboratory</i>.'
---

## Abstract

This research presents a novel CLIP-driven Feature-wise Linear Modulation (FiLM) architecture for text-guided 3D medical image segmentation. We introduce two variants—CLIP-Deep-Driven FiLM and CLIP-Deep-FiLM-2—that integrate CLIP text embeddings with SwinUNETR's transformer-based encoder-decoder structure. The framework enables natural language-driven organ segmentation by modulating image features at multiple decoder stages using text embeddings, achieving robust performance across diverse abdominal CT datasets.

**GitHub Repository**: [skapoor2024/swinunetr_deep_film_MSD](https://github.com/skapoor2024/swinunetr_deep_film_MSD)

**Research Institution**: University of Florida, Dr. Gong's Laboratory

## Key Innovations

### CLIP-Deep-Driven FiLM Architecture

#### FiLM-Based Conditional Modulation
- **Feature-wise Linear Modulation**: Applies affine transformations to image features using text-derived gamma and beta parameters
- **Deep Integration**: FiLM blocks inserted at multiple decoder stages for hierarchical text-image fusion
- **Xavier Initialization**: Optimized weight initialization for stable training convergence

**FiLM Block Implementation**:
```python
class FiLMBlock(nn.Module):
    def __init__(self, feature_dim, text_dim=512):
        super().__init__()
        self.gamma = nn.Linear(text_dim, feature_dim)  # Scaling parameter
        self.beta = nn.Linear(text_dim, feature_dim)   # Shift parameter
        nn.init.xavier_uniform_(self.gamma.weight)
        nn.init.xavier_uniform_(self.beta.weight)
    
    def forward(self, feature, text):
        # feature: (B, C, D, H, W)
        # text: (B, text_dim=512)
        gamma = self.gamma(text).unsqueeze(-1)  # (B, C, 1)
        beta = self.beta(text).unsqueeze(-1)     # (B, C, 1)
        modulated = gamma * feature_flat + beta
        return modulated.view(B, C, D, H, W)
```

#### CLIP Text Encoder Integration
- **Pre-trained CLIP**: Uses OpenAI's CLIP-ViT-base-patch32 for text encoding
- **Frozen Text Encoder**: Text encoder weights kept frozen to preserve pre-trained language understanding
- **Pooled Embeddings**: Extracts 512-dimensional pooled output from CLIP's text encoder
- **Precomputed Prompts**: Supports efficient caching of text embeddings for repeated organ descriptions

### Two Architectural Variants

#### Variant 1: CLIP-Deep-Driven FiLM
**Architecture**: FiLM modulation applied only at decoder stages

```
SwinViT Encoder → Hidden States (HS0-HS4)
                ↓
Encoder Pathway: Enc0 → Enc1 → Enc2 → Enc3 → Dec4
                ↓
Text Embeddings → FiLM Blocks at each decoder
                ↓
Decoder Pathway: Dec4 → [FiLM4] → Dec3 → [FiLM3] → Dec2 → [FiLM2] → Dec1 → [FiLM1] → Dec0
                ↓
Final Segmentation Logits
```

**Feature Dimensions**:
- Dec4 (bottleneck): 768 channels → FiLM Block 4
- Dec3: 384 channels → FiLM Block 3
- Dec2: 192 channels → FiLM Block 2
- Dec1: 96 channels → FiLM Block 1

#### Variant 2: CLIP-Deep-FiLM-2
**Architecture**: FiLM modulation at both encoder and decoder stages

```
SwinViT → HS0-HS4
        ↓
Enc1 → [FiLM0] → Enc2 → [FiLM1] → Enc3 → [FiLM2] → HS3 → [FiLM3]
        ↓
Dec4 → [FiLM4] → Dec3 → Dec2 → Dec1 → Dec0
```

**Key Differences**:
- Pre-residual normalization for improved gradient flow
- Residual connections around FiLM layers
- Encoder-side modulation for early text-image fusion
- More parameters but richer text-visual interaction

## Technical Implementation

### Text Prompt Processing

**Flexible Prompt Handling**:
1. **String Prompts**: "A Computed Tomography of abdomen organ"
2. **Batch Prompts**: List of organ-specific descriptions per sample
3. **Precomputed Embeddings**: Cached embeddings from pickle files for efficiency
4. **Default Fallback**: Generic CT description when no prompt provided

**Example Prompts**:
```python
prompts = [
    "Segment liver and kidneys from abdominal CT",
    "Identify pancreas and spleen regions",
    "Locate gallbladder in CT scan"
]
```

### Training Configuration

**Distributed Training**:
- **Framework**: PyTorch Distributed Data Parallel (DDP)
- **Multi-node Support**: SLURM-based cluster training
- **Command**:
  ```bash
  python -m torch.distributed.launch \
      --nproc_per_node=8 \
      --master_port=1234 \
      train_deep_film.py --dist True --uniform_sample
  ```

**Hyperparameters**:
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Linear warmup + cosine annealing (10 warmup epochs)
- **Batch Size**: 3-4 per GPU
- **ROI Size**: 96×96×96 voxels
- **Intensity Normalization**: CT window [-175, 250] HU → [0.0, 1.0]
- **Spatial Resolution**: 1.5mm × 1.5mm × 1.5mm

**Loss Functions**:
```python
Total Loss = BCE Loss + Dice Loss

# Multi-class binary cross-entropy
loss_bce = Multi_BCELoss(num_classes=NUM_CLASS)

# Soft Dice loss for handling class imbalance
loss_dice = DiceLoss(num_classes=NUM_CLASS)
```

### Dataset Configuration

**Multi-Dataset Training**:
The framework supports training on diverse medical imaging datasets:

1. **PAOT (Pan-Abdominal Organ Tracking)**: 10 organs, inner annotations
2. **BTCV (Beyond The Cranial Vault)**: Multi-organ abdominal segmentation
3. **KiTS (Kidney Tumor Segmentation)**: Kidney and tumor delineation
4. **AMOS 2022**: Abdominal multi-organ benchmark
5. **CT-ORG**: 12-organ abdominal CT dataset

**Dataset Organization**:
```
dataset/dataset_list/
├── PAOT_train.txt    # {image_path \t label_path}
├── PAOT_val.txt
├── PAOT_test.txt
├── PAOT_test2.txt    # Test without ground truth
└── ... (similar for each dataset)
```

### Evaluation Methodology

**Sliding Window Inference**:
- ROI size: 96×96×96 voxels
- Overlap strategy for full volume reconstruction
- Sigmoid activation for multi-label predictions

**Metrics**:
- **Dice Score**: Primary metric for segmentation quality
- **Per-Organ Analysis**: Individual organ performance tracking
- **Template-based Evaluation**: Organ-specific templates for different datasets

**Validation Script**:
```python
python model_val.py \
    --pretrain ./out/deep_film/epoch_160.pth \
    --model_type film \
    --file_name deep_film_btcv_e160.txt \
    --dataset_list btcv
```

## Experimental Results

### Qualitative Results

**Sample Predictions** (from repository):

1. **12-Organ CT Dataset**:
   - Accurate liver, kidney, spleen delineation
   - Precise pancreas and gallbladder boundaries
   - Clean organ separation without over-segmentation

2. **KiTS Dataset**:
   - Clear kidney cortex identification
   - Tumor region segmentation with well-defined boundaries
   - Robust performance on varying tumor sizes

3. **AMOS Dataset**:
   - Consistent multi-organ segmentation
   - Handles anatomical variations effectively
   - Maintains accuracy across different CT protocols

### Model Comparison

**Supported Baselines**:
- **SwinUNETR**: Pure vision-based transformer segmentation
- **Universal Model**: Multi-dataset training without text guidance
- **UNETR**: Original U-Net + Transformer architecture
- **FiLM Variants**: Deep-FiLM vs Deep-FiLM-2

### Training Efficiency

**Multi-Node Training**:
```bash
srun python train_deep_film_2_multinode.py \
    --dist True \
    --uniform_sample \
    --num_workers 4 \
    --log_name deep_film_2_multinode \
    --world_size $WORLD_SIZE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
```

**Performance Optimizations**:
- Gradient checkpointing for memory efficiency
- Mixed precision training support
- Efficient data loading with prefetching
- TensorBoard integration for monitoring

## Technical Highlights

### Code Organization

```
swinunetr_deep_film_MSD/
├── model/
│   ├── SwinUNETR_DEEP_FILM.py      # Main FiLM architecture
│   ├── SwinUNETR_DEEP_FILM_2.py    # Enhanced variant
│   ├── SwinUNETR_FLIM.py           # Base FiLM
│   └── Universal_model.py           # Baseline model
├── train_deep_film.py               # Single-node training
├── train_deep_film_2_multinode.py  # Distributed training
├── train_val_clip.py                # CLIP-based training
├── model_test.py                    # Evaluation script
├── model_val.py                     # Validation script
├── gg_tools.py                      # Utilities (Dice, losses)
├── dataset/                         # Dataset lists
├── optimizers/                      # Custom optimizers
└── utils/                           # Helper functions
```

### Key Technologies

- **Deep Learning**: PyTorch 1.x with DDP
- **Medical Imaging**: MONAI (Medical Open Network for AI)
- **Vision-Language**: Hugging Face Transformers (CLIP)
- **Compute**: Multi-GPU and multi-node training
- **Visualization**: TensorBoard, custom plotting tools

### Innovations Over Baseline

**Compared to Standard SwinUNETR**:
1. **Text Conditioning**: Natural language guidance for organ specification
2. **FiLM Modulation**: Learnable affine transformations from text
3. **Multi-Stage Integration**: Text features used throughout decoder
4. **Flexible Prompting**: Support for varied natural language descriptions

**Compared to Universal CLIP**:
1. **Deep FiLM**: Multiple modulation points vs single integration
2. **Encoder Modulation**: FiLM-2 adds encoder-side text fusion
3. **Architectural Flexibility**: Easy switching between variants
4. **Precomputed Caching**: Efficient handling of repeated prompts

## Implementation Details

### Memory Management

**GPU Memory Optimization**:
- Patch-based training with 96³ ROI
- Gradient checkpointing for deeper networks
- Efficient sliding window inference
- Mixed precision with automatic mixed precision (AMP)

### Data Preprocessing

**CT Scan Processing**:
```python
# Intensity windowing
a_min, a_max = -175, 250  # HU
b_min, b_max = 0.0, 1.0

# Spatial resampling
target_spacing = (1.5, 1.5, 1.5)  # mm

# Data augmentation
- Random flips
- Random rotations
- Intensity shifts
- Elastic deformations
```

### Label Transfer Utility

The repository includes `label_transfer.py` for:
- Converting between different organ label formats
- Harmonizing multi-dataset annotations
- Creating unified organ templates

## Research Contributions

### Scientific Novelty

1. **First CLIP-FiLM Integration for 3D Medical Imaging**: Pioneering use of vision-language models with deep FiLM for volumetric CT segmentation

2. **Hierarchical Text Modulation**: Multi-stage FiLM blocks enabling text guidance at different semantic levels

3. **Practical Framework**: Production-ready implementation with distributed training and evaluation scripts

4. **Multi-Dataset Generalization**: Robust performance across diverse organ datasets and imaging protocols

### Practical Impact

**Clinical Applications**:
- **Radiotherapy Planning**: Text-guided ROI delineation for treatment
- **Surgical Planning**: Organ localization for preoperative assessment
- **Disease Quantification**: Automated organ volume measurements
- **Multi-Reader Studies**: Standardized organ definitions via text prompts

**Research Applications**:
- **Dataset Curation**: Efficient annotation with text guidance
- **Model Interpretability**: Understanding text-image interactions
- **Transfer Learning**: Leveraging CLIP for medical domain adaptation

## Future Directions


### Technical Enhancements

1. **Attention Visualization**: Understanding which text tokens drive segmentation
2. **Prompt Optimization**: Learning optimal text descriptions for organs
3. **Uncertainty Quantification**: Confidence estimates for clinical deployment
4. **Real-Time Inference**: Optimization for interactive clinical use


## Acknowledgments

This work was conducted at the University of Florida under the supervision of Dr. Gong. The research leverages the Universal CLIP architecture and extends it with novel FiLM-based text-image fusion mechanisms.

---

*This research demonstrates expertise in vision-language models, 3D medical image segmentation, distributed deep learning, and production-ready system development. The work contributes both novel architectural innovations and practical tools for text-guided medical imaging analysis.*
- **Multi-Modal Extension**: Integration with other imaging modalities (MRI, PET)
- **Real-Time Processing**: Optimization for real-time clinical applications
- **Few-Shot Learning**: Adaptation to new organ types with minimal training data
- **Clinical Validation**: Prospective studies in clinical environments

## Industry Applications


### Research Contributions
- **Open Source Tools**: Public release of model and evaluation code
- **Benchmark Datasets**: Contribution to standardized evaluation protocols
- **Reproducible Research**: Comprehensive documentation for result reproduction

This research demonstrates the potential for natural language interfaces to make medical AI more accessible and intuitive, representing a significant step toward user-friendly medical imaging tools that can enhance clinical workflows and improve patient care.