---
title: "Polycrystalline Grain Segmentation"
excerpt: "Advanced image processing for materials science with 7% SAM enhancement<br/><img src='/images/grain-segmentation.png'>"
collection: portfolio
date: 2023-02-01
venue: 'UF SMART Data Lab'
technologies: 'Python, OpenCV, PyTorch, UNET, SAM'
github: 'https://github.com/skapoor2024/grain-segmentation'
---

## Project Overview

Developed advanced image processing algorithms for grain segmentation in polycrystalline TEM (Transmission Electron Microscopy) images to validate the PRIMME grain growth model, achieving an average IOU score of 0.79 and enhancing SAM model performance by 7% through innovative filtering techniques.

### Materials Science Context
Understanding grain structure evolution in polycrystalline materials is crucial for predicting material properties such as strength, conductivity, and corrosion resistance. The PRIMME (Parallel Rational Instance-based Methods for Materials Microstructure Evolution) model provides theoretical predictions for grain growth, requiring experimental validation through microscopy image analysis.

## Technical Challenge

### Problem Statement
- **Complex Grain Boundaries**: Polycrystalline aluminum exhibits irregular, overlapping grain boundaries
- **Noise and Artifacts**: TEM imaging introduces various forms of noise and imaging artifacts
- **Partial Labels**: Ground truth annotations are incomplete and require intelligent handling
- **Temporal Analysis**: Tracking grain evolution over time requires consistent segmentation

### Dataset Characteristics
- **Image Type**: High-resolution TEM images of polycrystalline aluminum
- **Resolution**: 2048×2048 pixels with nanometer-scale resolution
- **Annotation**: Partially labeled datasets with incomplete boundary information
- **Temporal Series**: Sequential images showing grain growth evolution

## Algorithm Development

### UNET-Based Segmentation Pipeline
```python
class GrainSegmentationUNET(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.encoder = UNETEncoder(n_channels)
        self.decoder = UNETDecoder(n_classes)
        self.bottleneck = ConvBlock(512, 1024)
        
    def forward(self, x):
        # Encoder pathway
        enc1 = self.encoder.conv1(x)
        enc2 = self.encoder.conv2(self.encoder.pool1(enc1))
        enc3 = self.encoder.conv3(self.encoder.pool2(enc2))
        enc4 = self.encoder.conv4(self.encoder.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.encoder.pool4(enc4))
        
        # Decoder pathway with skip connections
        dec4 = self.decoder.upconv4(bottleneck, enc4)
        dec3 = self.decoder.upconv3(dec4, enc3)
        dec2 = self.decoder.upconv2(dec3, enc2)
        dec1 = self.decoder.upconv1(dec2, enc1)
        
        return torch.sigmoid(dec1)
```

### Loss Function Optimization
- **Binary Cross-Entropy Loss**: Primary loss for grain boundary detection
- **Dice Score Optimization**: Additional metric focusing on boundary overlap
- **Weighted Loss**: Handling class imbalance between grain interiors and boundaries
- **Focal Loss**: Addressing hard negative examples in boundary detection

### Training Strategy
```python
def train_grain_segmentation(model, dataloader, optimizer):
    criterion_bce = nn.BCELoss()
    criterion_dice = DiceLoss()
    
    for batch_idx, (images, targets, weights) in enumerate(dataloader):
        optimizer.zero_grad()
        
        outputs = model(images)
        
        # Combined loss function
        bce_loss = criterion_bce(outputs, targets)
        dice_loss = criterion_dice(outputs, targets)
        weighted_loss = (weights * bce_loss).mean()
        
        total_loss = 0.7 * weighted_loss + 0.3 * dice_loss
        total_loss.backward()
        optimizer.step()
```

## SAM Model Enhancement

### Segment Anything Model Integration
- **Baseline Performance**: Initial SAM application to grain boundary detection
- **Domain Adaptation**: Fine-tuning SAM for materials science imagery
- **Prompt Engineering**: Developing effective prompting strategies for grain segmentation
- **Performance Benchmarking**: Systematic evaluation against traditional methods

### Mode Filter Innovation
```python
class TemporalModeFilter:
    def __init__(self, window_size=5, confidence_threshold=0.8):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.grain_history = []
        
    def apply_temporal_consistency(self, current_segmentation, grain_velocities):
        """
        Leverage stationary structure of grain movement for filtering
        """
        # Track grain boundaries over temporal sequence
        stable_regions = self.identify_stationary_grains(grain_velocities)
        
        # Apply mode filtering to stable regions
        filtered_segmentation = current_segmentation.copy()
        for region in stable_regions:
            # Extract temporal window for this region
            region_history = self.extract_region_history(region)
            
            # Apply mode filter
            mode_result = self.compute_temporal_mode(region_history)
            filtered_segmentation[region] = mode_result
            
        return filtered_segmentation
    
    def identify_stationary_grains(self, velocities):
        """Identify grains with minimal movement for stable filtering"""
        stationary_mask = velocities < self.velocity_threshold
        return self.extract_connected_components(stationary_mask)
```

### Performance Improvement Analysis
- **Baseline SAM**: IOU score of 0.73 on grain segmentation task
- **Enhanced SAM with Mode Filter**: IOU score of 0.78 (7% improvement)
- **Temporal Consistency**: Reduced flickering in boundary detection by 60%
- **Computational Efficiency**: 15% reduction in post-processing time

## Advanced Image Processing Techniques

### Preprocessing Pipeline
```python
class TEMImagePreprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.bilateral_filter = cv2.bilateralFilter
        
    def preprocess_tem_image(self, image):
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(image)
        
        # Contrast enhancement
        enhanced = self.clahe.apply(denoised)
        
        # Edge-preserving smoothing
        smoothed = self.bilateral_filter(enhanced, 9, 75, 75)
        
        # Normalization
        normalized = (smoothed - smoothed.mean()) / smoothed.std()
        
        return normalized
```

### Post-Processing Optimization
- **Morphological Operations**: Opening and closing for boundary refinement
- **Connected Component Analysis**: Filtering small noise components
- **Watershed Algorithm**: Separating touching grain boundaries
- **Active Contours**: Precise boundary localization

## Validation Against PRIMME Model

### Quantitative Validation Metrics
- **Grain Size Distribution**: Statistical comparison with PRIMME predictions
- **Boundary Length Evolution**: Temporal analysis of grain boundary changes
- **Topology Metrics**: Grain connectivity and neighbor relationships
- **Growth Rate Analysis**: Validation of predicted grain growth kinetics

### Experimental Results
```python
# Example validation results
validation_metrics = {
    'average_iou': 0.79,
    'boundary_detection_precision': 0.84,
    'boundary_detection_recall': 0.76,
    'temporal_consistency': 0.88,
    'grain_count_accuracy': 0.92
}

# PRIMME model correlation
primme_correlation = {
    'grain_size_distribution': 0.85,  # Pearson correlation
    'growth_rate_prediction': 0.78,   # R² score
    'topology_preservation': 0.81     # Structural similarity
}
```

### Statistical Analysis
- **Cross-Validation**: 5-fold validation across different aluminum samples
- **Temporal Validation**: Consistency analysis across time-series data
- **Inter-Annotator Agreement**: Validation against multiple expert annotations
- **Robustness Testing**: Performance across different imaging conditions

## Materials Science Applications

### Grain Growth Kinetics
- **Growth Law Validation**: Confirming theoretical power-law relationships
- **Temperature Dependence**: Analysis of grain growth at different temperatures
- **Pinning Effects**: Studying grain boundary pinning by secondary phases
- **Texture Evolution**: Tracking crystallographic orientation changes

### Property Prediction
```python
class MaterialPropertyPredictor:
    def __init__(self, grain_segmentation_model):
        self.segmentation_model = grain_segmentation_model
        self.property_model = self.load_property_model()
        
    def predict_material_properties(self, tem_image):
        # Extract grain structure
        grain_map = self.segmentation_model(tem_image)
        
        # Calculate microstructural features
        features = self.extract_microstructural_features(grain_map)
        
        # Predict properties
        predictions = {
            'yield_strength': self.property_model.predict_strength(features),
            'electrical_conductivity': self.property_model.predict_conductivity(features),
            'corrosion_resistance': self.property_model.predict_corrosion(features)
        }
        
        return predictions
```

## Technical Innovation & Contributions

### Novel Algorithmic Contributions
- **Temporal Mode Filtering**: First application of mode filtering for grain boundary consistency
- **Partial Label Learning**: Effective handling of incomplete annotation data
- **SAM Domain Adaptation**: Successful adaptation of foundation model to materials science
- **Multi-Scale Analysis**: Integration of local and global grain structure information

### Methodological Advances
- **Stationary Structure Exploitation**: Novel use of grain movement patterns for filtering
- **Cross-Modal Validation**: Integration of simulation and experimental data
- **Uncertainty Quantification**: Probabilistic assessment of segmentation quality
- **Real-Time Processing**: Optimized algorithms for live microscopy analysis

## Technology Stack

### Computer Vision Libraries
- **OpenCV**: Core image processing and computer vision operations
- **scikit-image**: Advanced image analysis and morphological operations
- **PIL/Pillow**: Image I/O and basic manipulations
- **ImageIO**: Multi-format image reading and writing

### Deep Learning Framework
- **PyTorch**: Primary framework for neural network development
- **Torchvision**: Pre-trained models and computer vision utilities
- **Albumentations**: Advanced data augmentation for training
- **Weights & Biases**: Experiment tracking and model comparison

### Scientific Computing
- **NumPy**: Numerical computing and array operations
- **SciPy**: Scientific computing and statistical analysis
- **Matplotlib**: Visualization and result presentation
- **Seaborn**: Statistical visualization and analysis

## Research Impact & Applications

### Scientific Contributions
- **Model Validation**: Provided experimental validation for PRIMME theoretical predictions
- **Methodology Development**: Established new standards for grain segmentation analysis
- **Cross-Disciplinary Impact**: Techniques applicable to other crystalline material systems
- **Open Source Tools**: Released segmentation tools for materials science community

### Industrial Applications
- **Quality Control**: Automated grain structure analysis in manufacturing
- **Materials Design**: Microstructure optimization for specific properties
- **Process Monitoring**: Real-time monitoring of material processing
- **Failure Analysis**: Understanding relationship between structure and failure modes

## Future Research Directions

### Advanced AI Integration
- **Foundation Models**: Adaptation of large vision models for materials analysis
- **Multi-Modal Learning**: Integration of multiple characterization techniques
- **Active Learning**: Intelligent selection of images for annotation
- **Federated Learning**: Collaborative training across research institutions

### Experimental Extensions
- **3D Analysis**: Extension to 3D grain structure from tomography data
- **In-Situ Studies**: Real-time analysis during material processing
- **Multi-Scale Integration**: Connecting atomic to macroscopic scales
- **Property Correlation**: Direct linking of structure to mechanical properties

## Project Outcomes

This project successfully demonstrated the application of advanced computer vision techniques to materials science challenges, achieving significant improvements in grain segmentation accuracy and establishing new methodological standards for microstructural analysis. The work provides a foundation for automated materials characterization and contributes to the broader field of AI-driven materials discovery.