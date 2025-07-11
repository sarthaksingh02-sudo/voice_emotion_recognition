# 🎭 Advanced Voice Emotion Recognition System

A comprehensive real-time emotion recognition system using advanced machine learning techniques, featuring multiple testing interfaces and high-performance audio processing.

## 🚀 Key Features

### 🎤 **Voice Testing Suite**
- **Manual Voice Testing** - Record and analyze your voice with detailed feedback
- **Batch Testing** - Test multiple emotions in sequence for comprehensive analysis
- **Real-time Recognition** - Live emotion detection with voice activity detection
- **Interactive Interface** - User-friendly testing with visual feedback

### 🧠 **Advanced ML Models**
- **Multiple Algorithms**: SVM, Random Forest, Gradient Boosting, MLP Neural Network, Logistic Regression
- **Feature Engineering**: 642 audio features per sample
- **Model Optimization**: Automated hyperparameter tuning and model selection
- **Performance Tracking**: Comprehensive confusion matrices and performance metrics

### 📊 **Supported Emotions**
- 😊 **Happy** - Joyful, excited expressions
- 😢 **Sad** - Sorrowful, melancholic tones
- 😠 **Angry** - Frustrated, aggressive speech
- 😨 **Fearful** - Scared, worried expressions
- 🤢 **Disgust** - Repulsed, disgusted tones
- 😐 **Neutral** - Calm, normal speech

### 🔧 **Technical Excellence**
- **RAVDESS Dataset**: Professional emotion dataset with 7,442 samples
- **Advanced Feature Extraction**: MFCC, spectral, chroma, mel-spectrogram, and pitch features
- **Real-time Processing**: Sub-second emotion detection
- **Voice Activity Detection**: Smart recording start/stop detection
- **Confidence Scoring**: Probability-based prediction confidence

## 🎯 **Current Performance**
- **Best Model**: SVM with 65% confidence on sample prediction
- **Real-time Processing**: < 1 second per prediction
- **Feature Extraction**: 642 features per audio sample
- **Voice Activity Detection**: Automatic speech detection

## 🛠️ **Installation**

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install librosa soundfile pyaudio
pip install matplotlib seaborn
pip install joblib
```

### Quick Setup
```bash
git clone https://github.com/sarthaksingh02-sudo/emotion_recognition_system.git
cd emotion_recognition_system
pip install -r requirements.txt
```

## 🎙️ **Voice Testing Options**

### 1. **Simple Voice Test** - Quick Single Test
```bash
python simple_voice_test.py
```
- Records 3 seconds of audio
- Instant emotion analysis
- Detailed confidence breakdown
- Perfect for quick testing

### 2. **Batch Voice Test** - Comprehensive Testing
```bash
python batch_voice_test.py
```
- Tests 6 different emotions in sequence
- Guided emotion prompts
- Accuracy summary and statistics
- Ideal for thorough evaluation

### 3. **Manual Voice Test** - Advanced Interface
```bash
python manual_voice_test.py
```
- Interactive testing interface
- Manual or continuous recording modes
- Detailed analysis with all emotion probabilities
- Recording management options

### 4. **Real-time Recognition**
```bash
python improved_real_time.py
```
- Voice Activity Detection
- Continuous emotion monitoring
- Live feedback with emojis

## 📈 **System Testing & Validation**

### Quick System Check
```bash
python test_final.py          # Complete pipeline test
python system_readiness_test.py  # System readiness validation
python comprehensive_test.py     # Detailed performance analysis
```

### Model Training
```bash
python train_model.py         # Train new models
python model_diagnostic.py    # Model performance diagnostics
```

## 📁 **Project Structure**

```
emotion_recognition_system/
├── 🎤 Voice Testing Suite
│   ├── simple_voice_test.py      # Quick single voice test
│   ├── batch_voice_test.py       # Comprehensive batch testing
│   ├── manual_voice_test.py      # Advanced interactive testing
│   └── improved_real_time.py     # Real-time recognition
├── 🧠 Core System
│   ├── train_model.py            # Model training pipeline
│   ├── model_trainer.py          # ML model implementations
│   ├── audio_processor.py        # Audio feature extraction
│   ├── data_parser.py            # RAVDESS dataset parser
│   └── config.py                 # Configuration settings
├── 🔧 Testing & Validation
│   ├── test_final.py             # Complete pipeline test
│   ├── system_readiness_test.py  # System validation
│   ├── comprehensive_test.py     # Performance analysis
│   └── model_diagnostic.py       # Model diagnostics
├── 📊 Generated Assets
│   ├── *.png                     # Confusion matrices
│   ├── models/                   # Trained models
│   ├── data/AudioWAV/            # RAVDESS dataset
│   └── recordings/               # Voice test recordings
└── 📝 Documentation
    ├── README.md                 # This file
    └── requirements.txt          # Dependencies
```

## 🧠 **Model Architecture**

### Feature Extraction Pipeline
1. **MFCC Features** (52): Mel-frequency cepstral coefficients with statistical measures
2. **Spectral Features** (24): Spectral centroid, rolloff, bandwidth, zero-crossing rate
3. **Chroma Features** (48): Chroma vector with statistical measures
4. **Mel-Spectrogram** (512): Mel-scale spectrogram with statistical measures
5. **Pitch Features** (6): Fundamental frequency characteristics
6. **Total**: 642 features per audio sample

### Available Models
| Model | Type | Status |
|-------|------|--------|
| **SVM** | Support Vector Machine | ✅ Production Ready |
| **Random Forest** | Ensemble Method | ✅ Production Ready |
| **Gradient Boosting** | Ensemble Method | ✅ Production Ready |
| **MLP Neural Network** | Deep Learning | ✅ Production Ready |
| **Logistic Regression** | Linear Model | ✅ Production Ready |

## 📊 **Performance Metrics**

### Model Comparison (Latest Results)
| Model | Primary Metric | Confidence Range |
|-------|---------------|------------------|
| **SVM** | 65% confidence | High (0.6-0.8) |
| **Random Forest** | Ensemble accuracy | Medium (0.4-0.7) |
| **Gradient Boosting** | Boosted performance | Medium (0.4-0.6) |
| **MLP Neural Network** | Deep learning | Variable (0.3-0.7) |
| **Logistic Regression** | Linear baseline | Low-Medium (0.3-0.6) |

### Confusion Matrices Available
- ✅ SVM Enhanced Confusion Matrix
- ✅ Random Forest Enhanced Confusion Matrix
- ✅ Gradient Boosting Enhanced Confusion Matrix
- ✅ MLP Neural Network Confusion Matrix
- ✅ Logistic Regression Enhanced Confusion Matrix
- ✅ Extra Trees Enhanced Confusion Matrix

## 🎤 **Voice Testing Guide**

### For Best Results:
1. **Environment**: Use a quiet room with minimal background noise
2. **Microphone**: Ensure your microphone is working and positioned correctly
3. **Expression**: Clearly express the intended emotion in your voice
4. **Duration**: Speak for 2-3 seconds with emotional content
5. **Clarity**: Speak clearly and at normal volume

### Emotion Testing Tips:
- **😊 Happy**: Sound joyful, excited, upbeat
- **😢 Sad**: Speak slowly, with a lower tone, sound melancholic
- **😠 Angry**: Use a louder, sharper tone, sound frustrated
- **😨 Fearful**: Sound worried, anxious, with a shaky voice
- **🤢 Disgust**: Express repulsion, sound disgusted
- **😐 Neutral**: Speak normally, calm and balanced

## 🔧 **Configuration**

Edit `config.py` to customize:
- Audio sampling rate and processing parameters
- Feature extraction settings
- Model training parameters
- Real-time processing settings

## 📊 **Dataset Information**

### RAVDESS Dataset
- **Total Samples**: 7,442 audio files
- **Actors**: 24 professional actors
- **Emotions**: 8 different emotions
- **Quality**: Professional studio recordings
- **Format**: 16-bit WAV files, 22050 Hz sampling rate

### Data Distribution
- **Training**: 80% of samples
- **Testing**: 20% of samples
- **Validation**: Cross-validation during training
- **Preprocessing**: Noise reduction, normalization, feature extraction

## 🚀 **Future Improvements**

- [ ] **Web Interface**: Browser-based emotion testing
- [ ] **Mobile App**: Android/iOS emotion recognition
- [ ] **API Service**: RESTful API for integration
- [ ] **Multi-language Support**: Support for different languages
- [ ] **Transformer Models**: State-of-the-art transformer architecture
- [ ] **Real-time Visualization**: Live audio waveform and emotion tracking
- [ ] **Batch Processing**: Process multiple files simultaneously
- [ ] **Model Ensemble**: Combine multiple models for better accuracy

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **RAVDESS Dataset**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **PyTorch Team**: Deep learning framework
- **Librosa**: Audio processing library
- **Scikit-learn**: Machine learning library
- **PyAudio**: Real-time audio I/O library

## 📧 **Contact & Support**

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/sarthaksingh02-sudo/emotion_recognition_system/issues)
- **Discussions**: [Join the discussion](https://github.com/sarthaksingh02-sudo/emotion_recognition_system/discussions)

---

**🎭 Made with ❤️ for advancing emotion recognition technology**

*"Understanding emotions through voice - bridging the gap between human expression and machine learning."*
