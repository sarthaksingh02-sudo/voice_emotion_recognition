"""
Configuration file for Emotion Recognition System v1.1
Comprehensive settings for GUI, API, Real-time processing, and Advanced features
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ================================
# VERSION INFORMATION
# ================================
VERSION = "1.1.0"
VERSION_NAME = "Advanced Emotion Recognition System"
RELEASE_DATE = "2025-07-11"
AUTHOR = "Sarthak Singh"

# ================================
# BASE PATHS
# ================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
DATABASE_DIR = BASE_DIR / "database"
GUI_ASSETS_DIR = BASE_DIR / "gui_assets"
API_UPLOADS_DIR = BASE_DIR / "api_uploads"
REPORTS_DIR = BASE_DIR / "reports"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, TEMP_DIR, LOGS_DIR, CACHE_DIR, 
                 DATABASE_DIR, GUI_ASSETS_DIR, API_UPLOADS_DIR, 
                 REPORTS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(exist_ok=True)

# Emotion labels
EMOTION_LABELS = {
    0: "happy",
    1: "sad", 
    2: "angry",
    3: "neutral",
    4: "fear",
    5: "disgust",
    6: "surprise"
}

# Audio processing parameters
SAMPLE_RATE = 16000  # Reduced from 22050 for faster processing
FRAME_SIZE = 2048
HOP_SIZE = 512
N_MFCC = 13
N_MELS = 128
MAX_AUDIO_LENGTH = 10  # Reduce from 30 to 10 seconds for speed

# Feature extraction optimization
ENABLE_PITCH_FEATURES = True   # Enable for accuracy
ENABLE_MEL_FEATURES = True     # Enable for accuracy
USE_PARALLEL_PROCESSING = True
MAX_WORKERS = 6  # Increase parallel workers
CACHE_FEATURES = True  # Enable feature caching
USE_FEATURE_SELECTION = True  # Enable feature selection

# Model parameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# API configuration
API_HOST = "127.0.0.1"
API_PORT = 8000
DEBUG = True

# Real-time processing
CHUNK_SIZE = 1024
RECORD_SECONDS = 3  # Duration for each emotion prediction
CONFIDENCE_THRESHOLD = 0.6

# File paths
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MODEL_PATH = MODEL_DIR / "emotion_model.pkl"
KERAS_MODEL_PATH = MODEL_DIR / "emotion_model.h5"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
