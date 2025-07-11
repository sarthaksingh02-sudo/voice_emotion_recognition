#!/usr/bin/env python3
"""
Fix model format for compatibility with real-time inference
"""

import joblib
import pickle
from pathlib import Path

def fix_model_format():
    """Fix the model format to be compatible with EmotionModelTrainer"""
    
    print("🔧 Fixing model format for compatibility...")
    
    # Load current model data
    model_data = joblib.load('models/emotion_model.pkl')
    print(f"Current model structure: {list(model_data.keys())}")
    
    # Extract the actual model
    best_model = model_data['model']
    print(f"Model type: {type(best_model)}")
    
    # Save the model directly (not in a dictionary)
    joblib.dump(best_model, 'models/emotion_model.pkl')
    print("✅ Model saved in correct format!")
    
    # Check metadata
    try:
        metadata = joblib.load('models/metadata.pkl')
        print(f"Metadata available: {list(metadata.keys())}")
        print(f"Best model name: {metadata['best_model_name']}")
    except Exception as e:
        print(f"Note: {e}")
    
    print("🎉 Model format fixed successfully!")

if __name__ == "__main__":
    fix_model_format()
