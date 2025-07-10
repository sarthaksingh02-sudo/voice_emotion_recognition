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
