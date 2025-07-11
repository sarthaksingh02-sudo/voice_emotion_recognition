# 🎯 Advanced Emotion Recognition System

A high-performance real-time emotion recognition system using advanced machine learning techniques and voice activity detection.

## 🚀 Features

### Core Features
- **Real-time emotion recognition** from speech
- **Voice Activity Detection (VAD)** - automatically detects when you start/stop speaking
- **Advanced feature extraction** with MFCC, spectral, chroma, mel-spectrogram, and pitch features
- **Deep learning model** using PyTorch MLP Neural Network
- **High accuracy** with ensemble of optimized models

### Supported Emotions
- 😊 Happy
- 😢 Sad  
- 😡 Angry
- 😨 Fear
- 🤢 Disgust
- 😐 Neutral

### Technical Features
- **Optimized feature extraction** with parallel processing
- **Feature caching** for faster processing
- **Robust audio preprocessing** with noise reduction
- **Automatic model selection** based on performance
- **Confidence scoring** for predictions

## 📊 Performance
- **Model Accuracy**: 53.8% (MLP Neural Network)
- **Real-time processing**:  1 second per prediction
- **Voice Activity Detection**: Smart start/stop detection
- **Feature extraction**: 640+ audio features per sample

## 🛠️ Installation

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install librosa soundfile pyaudio
pip install webrtcvad
pip install matplotlib seaborn
pip install joblib
```

### Quick Setup
```bash
git clone https://github.com/your-username/emotion_recognition_system.git
cd emotion_recognition_system
pip install -r requirements.txt
```

## 🎯 Usage

### 1. Real-time Emotion Recognition (Improved)
```bash
python improved_real_time.py
```
**Features:**
- Voice Activity Detection automatically detects when you speak
- No need to press buttons - just start talking!
- Intelligent silence detection
- Visual feedback with emojis and confidence indicators

### 2. Basic Real-time Recognition
```bash
python real_time_inference.py
```

### 3. Train Your Own Model
```bash
python train_model.py
```

### 4. Test on Audio Files
```bash
python quick_test.py
```

## 📁 Project Structure

```
emotion_recognition_system/
├── improved_real_time.py      # 🎯 Advanced real-time recognition with VAD
├── real_time_inference.py     # Basic real-time recognition
├── train_model.py             # Model training pipeline
├── model_trainer.py           # ML model implementations
├── audio_processor.py         # Audio feature extraction
├── data_parser.py             # Dataset parsing utilities
├── config.py                  # Configuration settings
├── quick_test.py              # Quick testing utilities
├── models/                    # Trained models directory
├── data/                      # Training data directory
├── temp/                      # Temporary files and cache
└── logs/                      # Training logs
```

## 🧠 Model Architecture

### MLP Neural Network (Best Model)
- **Input Layer**: 640+ audio features
- **Hidden Layers**: 
  - Layer 1: 256 neurons + BatchNorm + Dropout(0.5)
  - Layer 2: 128 neurons + BatchNorm + Dropout(0.5)  
  - Layer 3: 64 neurons + BatchNorm + Dropout(0.3)
- **Output Layer**: 6 emotion classes
- **Activation**: ReLU + Softmax
- **Optimizer**: Adam with learning rate scheduling

### Feature Extraction Pipeline
1. **MFCC Features**: 13 coefficients × 4 statistical measures = 52 features
2. **Spectral Features**: 4 spectral measures × 6 statistics = 24 features
3. **Chroma Features**: 12 chroma bins × 4 statistics = 48 features
4. **Mel-Spectrogram**: 128 mel bins × 4 statistics = 512 features
5. **Pitch Features**: 6 pitch-based measurements
6. **Total**: 640+ features per audio sample

## 📈 Training Process

### Data Preparation
- **Dataset**: CREMA-D (7,442 audio samples)
- **Preprocessing**: Noise reduction, normalization, padding
- **Feature Extraction**: Parallel processing with caching
- **Data Split**: 80% training, 20% testing

### Model Training
- **Multiple Models**: Random Forest, SVM, Gradient Boosting, MLP
- **Hyperparameter Tuning**: Grid search optimization
- **Early Stopping**: Prevent overfitting
- **Model Selection**: Automatic best model selection

## 🎤 Real-time Recognition Guide

### Using the Improved System
1. Run `python improved_real_time.py`
2. Wait for "🎤 Listening... Start speaking!"
3. Speak naturally (1-10 seconds)
4. Stop speaking and wait for results
5. See emotion prediction with confidence score

### Tips for Best Results
- **Clear speech**: Speak clearly and at normal volume
- **Emotional expression**: Express the emotion you want to test
- **Quiet environment**: Minimize background noise
- **Normal pace**: Don't speak too fast or too slow

## 🔧 Configuration

Edit `config.py` to customize:
- Audio sampling rate and processing parameters
- Feature extraction settings
- Model training parameters
- Real-time processing settings

## 📊 Performance Metrics

### Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MLP Neural Network | 53.8% | 0.54 | 0.54 | 0.53 |
| SVM | 52.2% | 0.52 | 0.52 | 0.52 |
| Logistic Regression | 51.8% | 0.51 | 0.52 | 0.51 |
| Gradient Boosting | 51.2% | 0.51 | 0.51 | 0.51 |

### Per-Emotion Performance
| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 0.68 | 0.70 | 0.69 |
| Sad | 0.52 | 0.71 | 0.60 |
| Neutral | 0.48 | 0.60 | 0.53 |
| Happy | 0.48 | 0.49 | 0.49 |
| Disgust | 0.58 | 0.39 | 0.46 |
| Fear | 0.50 | 0.36 | 0.42 |

## 🚀 Future Improvements

- [ ] Add more emotion categories
- [ ] Implement transformer-based models
- [ ] Add real-time audio visualization
- [ ] Support for multiple languages
- [ ] Web interface for easy usage
- [ ] Mobile app integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CREMA-D dataset for emotion recognition research
- PyTorch team for the deep learning framework
- librosa for audio processing capabilities
- WebRTC team for voice activity detection

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for advancing emotion recognition technology**

# Voice Emotion Recognition System

A machine learning system for recognizing emotions from voice recordings, designed for mental health monitoring applications.

## Features

- **Audio Processing**: Advanced feature extraction using MFCC, spectral, chroma, mel-spectrogram, and pitch features
- **Multiple ML Models**: Random Forest, SVM, Gradient Boosting, MLP, and Logistic Regression
- **Real-time Recognition**: Live emotion detection from microphone input
- **CREMA-D Dataset Support**: Pre-configured for the CREMA-D emotion dataset
- **6 Emotion Classes**: Happy, Sad, Angry, Neutral, Fear, Disgust

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sarthaksingh02-sudo/voice_emotion_recognition.git
cd voice_emotion_recognition
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset Setup

This project uses the CREMA-D dataset. Place your audio files in the `data/AudioWAV/` directory.

The CREMA-D dataset contains 7,442 audio files from 91 actors speaking 12 sentences with 6 different emotions:
- **Angry** (ANG): 1,271 files
- **Disgust** (DIS): 1,271 files  
- **Fear** (FEA): 1,271 files
- **Happy** (HAP): 1,271 files
- **Sad** (SAD): 1,271 files
- **Neutral** (NEU): 1,087 files

## Usage

### Training the Model

Run the training script to train emotion recognition models:

```bash
python train_model.py
```

This will:
- Load and parse the CREMA-D dataset
- Extract 642 audio features per file
- Train 5 different ML models
- Save the best performing model
- Generate confusion matrices for evaluation

### Real-time Emotion Recognition

For live emotion detection from microphone:

```bash
python real_time_inference.py
```

Choose option 1 for real-time recognition or option 2 to test on a specific audio file.

### Testing on Audio Files

To test the model on a specific audio file:

```python
from model_trainer import EmotionModelTrainer
from audio_processor import AudioProcessor

# Load trained model
trainer = EmotionModelTrainer()
trainer.load_models()

# Process audio file
processor = AudioProcessor()
features = processor.process_audio_file("path/to/audio.wav")

# Predict emotion
emotion, confidence = trainer.predict_emotion(features)
print(f"Emotion: {emotion}, Confidence: {confidence:.2f}")
```

## Model Performance

Current model performance on CREMA-D dataset (500 sample subset):

| Model | Accuracy | Cross-Validation Score |
|-------|----------|----------------------|
| **Logistic Regression** | **53.0%** | **46.3% ± 8.5%** |
| MLP Neural Network | 48.0% | 38.5% ± 7.8% |
| Random Forest | 38.0% | 38.3% ± 10.1% |
| Gradient Boosting | 37.0% | 35.8% ± 5.2% |
| SVM | 34.0% | 35.5% ± 8.8% |

The Logistic Regression model shows the best performance and is used as the default model.

## Feature Extraction

The system extracts 642 features from each audio file:

- **MFCC Features** (52): Mel-frequency cepstral coefficients with statistical measures
- **Spectral Features** (24): Spectral centroid, rolloff, bandwidth, zero-crossing rate
- **Chroma Features** (48): Chroma vector with statistical measures  
- **Mel-spectrogram Features** (512): Mel-scale spectrogram with statistical measures
- **Pitch Features** (6): Fundamental frequency characteristics

## Project Structure

```
emotion_recognition_system/
├── config.py              # Configuration parameters
├── audio_processor.py     # Audio feature extraction
├── data_parser.py         # CREMA-D dataset parser
├── model_trainer.py       # ML model training and evaluation
├── train_model.py         # Training script
├── real_time_inference.py # Real-time emotion recognition
├── requirements.txt       # Python dependencies
├── data/                  # Dataset directory
│   └── AudioWAV/         # CREMA-D audio files
├── models/               # Saved models and preprocessing objects
├── temp/                 # Temporary files
└── logs/                 # Log files
```

## Requirements

- Python 3.13+
- NumPy, Pandas, Scikit-learn
- Librosa, SoundFile, PyAudio
- Matplotlib, Seaborn
- See `requirements.txt` for complete list

## Applications

This system can be integrated into:

- **Virtual Counseling Platforms**: Real-time emotional feedback during therapy sessions
- **Mental Health Monitoring**: Continuous emotion tracking for patients
- **Voice Assistants**: Emotion-aware responses
- **Call Centers**: Customer emotion analysis
- **Research**: Emotional speech analysis studies

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- CREMA-D dataset creators for providing the emotion recognition dataset
- Librosa library for audio processing capabilities
- Scikit-learn for machine learning algorithms
