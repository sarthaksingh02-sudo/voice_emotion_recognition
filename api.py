from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Dict
import librosa
import numpy as np
import uuid
import joblib
from audio_processor import AudioProcessor
from database_logger import log_prediction, initialize_database
from pathlib import Path
import sqlite3
from contextlib import closing
from config import EMOTION_LABELS, MODEL_DIR

app = FastAPI(title="Emotion Recognition API", version="1.1.0")

audio_processor = AudioProcessor()

# Load trained models
try:
    model = joblib.load(MODEL_DIR / "emotion_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    MODEL_LOADED = True
    print("✅ Models loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ Error loading models: {e}")

# Ensure database is initialized
initialize_database()

# ================================
# API: Health Check Endpoint
# ================================
@app.get("/health", response_model=Dict[str, str])
def health_check():
    """Check API health"""
    return {"status": "healthy"}

# ================================
# API: Emotion Prediction
# ================================
@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """Predict emotion from uploaded audio file"""
    session_id = str(uuid.uuid4())
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Load audio
        audio_data, _ = librosa.load(file.file, sr=audio_processor.sample_rate)
        audio_data = audio_processor.preprocess_audio(audio_data)
        
        if audio_data is None:
            raise HTTPException(status_code=400, detail="Invalid or corrupted audio file")
        
        # Extract features
        features = audio_processor.extract_all_features(audio_data)
        
        if features is None:
            raise HTTPException(status_code=400, detail="Failed to extract features from audio")
        
        # Check if models are loaded
        if not MODEL_LOADED:
            raise HTTPException(status_code=500, detail="Models not loaded. Please train models first.")
        
        # Predict using the loaded model
        try:
            # Reshape features for prediction
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Get prediction
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Get emotion name
            emotion_idx = label_encoder.inverse_transform([prediction])[0]
            emotion = EMOTION_LABELS.get(emotion_idx, "unknown")
            
            # Get confidence (max probability)
            confidence = float(np.max(probabilities))
            
            # Get model name
            model_name = type(model).__name__
            
            # Log the prediction
            log_prediction(emotion, confidence, model_name, session_id)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "model": model_name,
            "session_id": session_id,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ================================
# API: Logs Retrieval
# ================================
@app.get("/logs")
def get_logs(limit: int = 100):
    """Retrieve system logs"""
    try:
        DATABASE_PATH = Path("database/logs.db")
        with closing(sqlite3.connect(DATABASE_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT timestamp, emotion, confidence, model, session_id
            FROM emotion_logs
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (limit,))
            
            logs = []
            for row in cursor.fetchall():
                logs.append({
                    "timestamp": row[0],
                    "emotion": row[1],
                    "confidence": row[2],
                    "model": row[3],
                    "session_id": row[4]
                })
            
            return {"logs": logs, "count": len(logs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")
