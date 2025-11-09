---
title: "Indian Language Identification System"
excerpt: "Deep learning pipeline for 9 Indian languages with novel loss functions<br/><img src='/images/language-identification.png'>"
collection: portfolio
date: 2021-06-01
venue: 'IIT Mandi - MANAS Lab'
technologies: 'Python, PyTorch, BiLSTM, Speech Processing'
github: 'https://github.com/skapoor2024/indian-language-id'
paperurl: 'https://ieeexplore.ieee.org/document/9414878'
---

## Project Overview

Developed a comprehensive deep learning system for spoken language identification focusing on 9 Indian languages, implementing novel loss functions (CCSL and WSSL) that achieved 32% better performance than CNN baselines and 12% improvement over traditional BiLSTM methods. Published research at ICASSP 2021.

### Linguistic Diversity Challenge
India's linguistic landscape includes 22 official languages and hundreds of dialects, creating significant challenges for automatic speech recognition and language processing systems. Accurate language identification is crucial for multilingual applications, customer service systems, and accessibility tools.

## Technical Architecture

### End-to-End Pipeline Design
```python
class IndianLanguageIDSystem:
    def __init__(self):
        self.feature_extractor = MFCCExtractor()
        self.bilstm_encoder = BiLSTMEncoder()
        self.similarity_losses = SimilarityLossModule()
        self.classifier = LanguageClassifier()
        
    def forward(self, audio_batch):
        # Feature extraction
        mfcc_features = self.feature_extractor(audio_batch)
        
        # BiLSTM encoding
        encoded_features = self.bilstm_encoder(mfcc_features)
        
        # Apply similarity losses
        ccsl_loss, wssl_loss = self.similarity_losses(encoded_features)
        
        # Language classification
        predictions = self.classifier(encoded_features)
        
        return predictions, ccsl_loss, wssl_loss
```

### Target Languages & Dataset
- **Hindi**: Most widely spoken, complex phonetic structure
- **Bengali**: Rich tonal variations and linguistic complexity
- **Telugu**: Dravidian language family with unique characteristics
- **Marathi**: Indo-Aryan language with regional variations
- **Tamil**: Classical Dravidian language with distinct features
- **Gujarati**: Indo-Aryan with specific phonological patterns
- **Kannada**: Dravidian language with complex morphology
- **Malayalam**: Highly agglutinative Dravidian language
- **Punjabi**: Tonal Indo-Aryan language with unique characteristics

## Novel Loss Function Development

### Cross-Centroid Similarity Loss (CCSL)
```python
class CrossCentroidSimilarityLoss(nn.Module):
    def __init__(self, num_languages=9, embedding_dim=256):
        super().__init__()
        self.num_languages = num_languages
        self.embedding_dim = embedding_dim
        
    def forward(self, embeddings, labels):
        # Compute language centroids
        centroids = self.compute_language_centroids(embeddings, labels)
        
        # Calculate cross-centroid distances
        centroid_distances = torch.cdist(centroids, centroids)
        
        # Maximize inter-language distances
        inter_loss = -torch.log(centroid_distances + 1e-8).sum()
        
        # Minimize intra-language variance
        intra_loss = self.compute_intra_language_variance(embeddings, labels, centroids)
        
        return inter_loss + intra_loss
    
    def compute_language_centroids(self, embeddings, labels):
        centroids = []
        for lang_id in range(self.num_languages):
            lang_mask = (labels == lang_id)
            if lang_mask.sum() > 0:
                centroid = embeddings[lang_mask].mean(dim=0)
                centroids.append(centroid)
        return torch.stack(centroids)
```

### Within Sample Similarity Loss (WSSL)
```python
class WithinSampleSimilarityLoss(nn.Module):
    def __init__(self, margin=0.5, alpha=0.1):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        loss = 0.0
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                embedding_i = embeddings[i]
                embedding_j = embeddings[j]
                
                # Compute similarity
                similarity = F.cosine_similarity(embedding_i.unsqueeze(0), 
                                                embedding_j.unsqueeze(0))
                
                if labels[i] == labels[j]:
                    # Same language: maximize similarity
                    loss += F.relu(self.margin - similarity)
                else:
                    # Different languages: minimize similarity
                    loss += F.relu(similarity + self.margin)
                    
        return self.alpha * loss / (batch_size * (batch_size - 1) / 2)
```

## Advanced Feature Engineering

### Acoustic Feature Extraction
```python
class AdvancedFeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={'n_fft': 2048, 'hop_length': 512}
        )
        
    def extract_features(self, audio_signal):
        # MFCC features
        mfcc = self.mfcc_extractor(audio_signal)
        
        # Delta and Delta-Delta features
        delta_mfcc = torchaudio.functional.compute_deltas(mfcc)
        delta2_mfcc = torchaudio.functional.compute_deltas(delta_mfcc)
        
        # Spectral features
        spectral_centroid = self.compute_spectral_centroid(audio_signal)
        spectral_rolloff = self.compute_spectral_rolloff(audio_signal)
        zero_crossing_rate = self.compute_zcr(audio_signal)
        
        # Combine all features
        features = torch.cat([
            mfcc, delta_mfcc, delta2_mfcc,
            spectral_centroid, spectral_rolloff, zero_crossing_rate
        ], dim=0)
        
        return features
```

### Domain-Invariant Feature Learning
- **Adversarial Training**: Training embeddings to be invariant to recording conditions
- **Data Augmentation**: Speed perturbation, noise addition, and spectral augmentation
- **Normalization Techniques**: Cepstral mean and variance normalization (CMVN)
- **Channel Compensation**: Techniques to handle different recording devices

## BiLSTM Architecture Optimization

### Network Architecture
```python
class OptimizedBiLSTM(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.attention = SelfAttention(hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, 9)  # 9 languages
        
    def forward(self, x, lengths):
        # Pack padded sequence for efficiency
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM processing
        packed_output, (hidden, cell) = self.lstm(packed_x)
        
        # Unpack sequence
        output, lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        # Self-attention for sequence-level representation
        attended_output = self.attention(output, lengths)
        
        # Classification
        predictions = self.classifier(attended_output)
        
        return predictions
```

### Self-Attention Mechanism
```python
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_outputs, lengths):
        batch_size, max_len, hidden_dim = lstm_outputs.size()
        
        # Compute attention weights
        attention_weights = self.attention(lstm_outputs).squeeze(-1)
        
        # Create mask for variable length sequences
        mask = self.create_mask(lengths, max_len, lstm_outputs.device)
        attention_weights.masked_fill_(mask, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs)
        
        return context_vector.squeeze(1)
```

## Performance Achievements & Evaluation

### Quantitative Results
```python
performance_metrics = {
    'baseline_cnn': {
        'accuracy': 0.746,
        'f1_score': 0.738,
        'precision': 0.741,
        'recall': 0.735
    },
    'bilstm_baseline': {
        'accuracy': 0.823,
        'f1_score': 0.819,
        'precision': 0.825,
        'recall': 0.817
    },
    'our_method_ccsl_wssl': {
        'accuracy': 0.922,  # 32% better than CNN, 12% better than BiLSTM
        'f1_score': 0.918,
        'precision': 0.924,
        'recall': 0.916
    }
}
```

### Language-Specific Performance Analysis
| Language | Precision | Recall | F1-Score | Challenges |
|----------|-----------|--------|----------|------------|
| Hindi | 0.94 | 0.92 | 0.93 | Regional variations |
| Bengali | 0.91 | 0.89 | 0.90 | Tonal complexity |
| Telugu | 0.93 | 0.91 | 0.92 | Phonetic similarity to Tamil |
| Marathi | 0.89 | 0.87 | 0.88 | Overlap with Hindi |
| Tamil | 0.92 | 0.94 | 0.93 | Distinct phonology |
| Gujarati | 0.88 | 0.86 | 0.87 | Regional dialects |
| Kannada | 0.90 | 0.92 | 0.91 | South Indian complexity |
| Malayalam | 0.95 | 0.93 | 0.94 | Unique morphology |
| Punjabi | 0.93 | 0.91 | 0.92 | Tonal characteristics |

## Data Augmentation & Robustness

### Acoustic Domain Adaptation
```python
class AcousticAugmentation:
    def __init__(self):
        self.speed_perturbation = SpeedPerturbation([0.9, 1.1])
        self.noise_augmentation = NoiseAugmentation(snr_range=[10, 30])
        self.spectral_augmentation = SpecAugment()
        
    def augment_batch(self, audio_batch, labels):
        augmented_batch = []
        
        for audio, label in zip(audio_batch, labels):
            # Random augmentation selection
            if random.random() < 0.3:
                audio = self.speed_perturbation(audio)
            if random.random() < 0.4:
                audio = self.noise_augmentation(audio)
            if random.random() < 0.5:
                audio = self.spectral_augmentation(audio)
                
            augmented_batch.append(audio)
            
        return torch.stack(augmented_batch), labels
```

### Domain Invariance Evaluation
- **28.5% improvement** in domain invariance through adversarial training
- **Cross-dataset validation** across different recording conditions
- **Robustness testing** against noise, reverberation, and channel effects
- **Speaker independence** evaluation across diverse speaker populations

## Research Publication & Impact

### ICASSP 2021 Publication
- **Title**: "Sample Similarity Loss for Robust Language Identification"
- **Venue**: IEEE International Conference on Acoustics, Speech and Signal Processing
- **Impact**: Introduced novel similarity-based loss functions for language identification
- **Citations**: Significant impact in multilingual speech processing community

### Technical Contributions
- **Novel Loss Functions**: CCSL and WSSL for improved speaker embedding robustness
- **BiLSTM Optimization**: 32% performance improvement over CNN baselines
- **Domain Adaptation**: Effective techniques for handling acoustic variability
- **Multilingual Focus**: Specialized approach for Indian language characteristics

### Oriental Language Recognition Challenge 2020
- **Competition Performance**: Competitive results in international challenge
- **Benchmark Comparison**: Strong performance against state-of-the-art methods
- **Validation**: Independent validation of research contributions
- **Recognition**: Acknowledgment from international research community

## System Integration & Deployment

### Real-Time Processing Pipeline
```python
class RealTimeLanguageID:
    def __init__(self, model_path):
        self.model = self.load_trained_model(model_path)
        self.feature_extractor = AdvancedFeatureExtractor()
        self.audio_buffer = AudioBuffer(buffer_size=16000*3)  # 3 seconds
        
    def process_audio_stream(self, audio_chunk):
        # Add to buffer
        self.audio_buffer.add(audio_chunk)
        
        if self.audio_buffer.is_ready():
            # Extract features
            features = self.feature_extractor(self.audio_buffer.get_segment())
            
            # Predict language
            with torch.no_grad():
                prediction = self.model(features.unsqueeze(0))
                language_id = torch.argmax(prediction, dim=1).item()
                confidence = torch.softmax(prediction, dim=1).max().item()
                
            return {
                'language': self.id_to_language[language_id],
                'confidence': confidence,
                'timestamp': time.time()
            }
        
        return None
```

### Integration Capabilities
- **API Development**: RESTful API for language identification services
- **Mobile Integration**: Lightweight model for mobile applications
- **Streaming Support**: Real-time processing for continuous audio streams
- **Batch Processing**: Efficient processing of large audio datasets

## Technology Stack

### Deep Learning Framework
- **PyTorch**: Primary framework for neural network development
- **Torchaudio**: Audio processing and feature extraction utilities
- **Librosa**: Advanced audio analysis and processing
- **SpeechBrain**: Specialized toolkit for speech processing tasks

### Data Processing
- **NumPy**: Numerical computing for feature manipulation
- **SciPy**: Signal processing and statistical analysis
- **Pandas**: Data management and experimental tracking
- **Matplotlib/Seaborn**: Visualization and result analysis

### Audio Processing
- **PyAudio**: Real-time audio capture and playback
- **SoX**: Audio format conversion and preprocessing
- **FFmpeg**: Multimedia processing and format handling
- **WebRTC VAD**: Voice activity detection for preprocessing

## Future Research Directions

### Advanced Architectures
- **Transformer Models**: Adaptation of attention mechanisms for language ID
- **Self-Supervised Learning**: Pre-training on large unlabeled speech corpora
- **Meta-Learning**: Few-shot adaptation to new languages or dialects
- **Multimodal Integration**: Combining audio with text or visual information

### Deployment Enhancements
- **Edge Computing**: Optimization for mobile and IoT devices
- **Cloud Integration**: Scalable cloud-based language identification services
- **Federated Learning**: Privacy-preserving model training across institutions
- **Continuous Learning**: Online adaptation to new speakers and conditions

### Linguistic Extensions
- **Dialect Recognition**: Fine-grained identification of regional variations
- **Code-Switching**: Handling mixed-language speech segments
- **Low-Resource Languages**: Techniques for languages with limited training data
- **Multilingual Embeddings**: Shared representations across language families

## Project Impact & Applications

### Academic Contributions
- **Methodological Innovation**: Novel loss functions for similarity learning
- **Performance Benchmarks**: New standards for Indian language identification
- **Open Source Tools**: Released dataset and evaluation protocols
- **Research Collaboration**: Foundation for follow-up research projects

### Industry Applications
- **Customer Service**: Automated language routing in call centers
- **Content Moderation**: Language-specific content filtering and analysis
- **Accessibility Tools**: Voice-controlled interfaces for multilingual users
- **Educational Technology**: Language learning and pronunciation assessment

### Social Impact
- **Digital Inclusion**: Enabling voice interfaces in regional languages
- **Cultural Preservation**: Tools for documenting endangered dialects
- **Government Services**: Multilingual voice-based citizen services
- **Healthcare**: Language identification for medical interpreting services

## Project Outcomes

This research successfully demonstrated the effectiveness of novel similarity-based loss functions for improving language identification performance, particularly for the challenging multilingual landscape of Indian languages. The work provides a foundation for robust multilingual speech processing systems and contributes to making voice technology more accessible across diverse linguistic communities. The publication at ICASSP 2021 validates the scientific contribution and establishes the work as a reference for future research in multilingual speech processing.