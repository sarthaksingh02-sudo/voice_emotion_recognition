#!/usr/bin/env python3
"""Final test script for emotion recognition system"""

import numpy as np
import joblib
from audio_processor import AudioProcessor
import os

def test_complete_pipeline():
    """Test complete emotion recognition pipeline"""
    print("🧪 Testing complete emotion recognition pipeline...")
    
    # Initialize audio processor
    processor = AudioProcessor()
    print("✓ Audio processor initialized")
    
    # Load models
    model_data = joblib.load('models/emotion_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    print("✓ Models loaded")
    
    # Test with a sample audio file
    sample_files = [f for f in os.listdir('data/AudioWAV') if f.endswith('.wav')]
    if sample_files:
        sample_path = os.path.join('data/AudioWAV', sample_files[0])
        print(f"✓ Testing with sample file: {sample_files[0]}")
        
        # Extract features
        features = processor.process_audio_file(sample_path)
        if features is not None:
            print(f"✓ Features extracted successfully (shape: {features.shape})")
            
            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))
            print("✓ Features scaled")
            
            # Predict with the model (now the model is directly loaded, not in a dict)
            best_model = model_data  # model is now directly loaded
            prediction = best_model.predict(features_scaled)[0]
            probabilities = best_model.predict_proba(features_scaled)[0]
            
            emotion = label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            print(f"🎯 Prediction: {emotion} (confidence: {confidence:.2f})")
            print(f"📊 Model type: {type(best_model).__name__}")
            print("✅ End-to-end pipeline test successful!")
            return True
        else:
            print("✗ Feature extraction failed")
            return False
    else:
        print("✗ No sample audio files found")
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n🎉 System is ready for deployment!")
    else:
        print("\n❌ System needs troubleshooting")
