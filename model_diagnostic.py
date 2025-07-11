#!/usr/bin/env python3
"""
Model diagnostic script
"""

import joblib
import numpy as np

def diagnose_model():
    """Diagnose model performance and configuration"""
    
    # Check model details
    model_data = joblib.load('models/emotion_model.pkl')
    print('🔍 Model Diagnostic Information')
    print('=' * 40)

    print(f'Model type: {type(model_data["model"])}')
    print(f'Training accuracy: {model_data["accuracy"]:.2%}')

    # Check if there are other models available
    print(f'\nModel data keys: {list(model_data.keys())}')

    # Check label encoder
    label_encoder = joblib.load('models/label_encoder.pkl')
    print(f'\nEmotion classes: {list(label_encoder.classes_)}')

    # Check if there's metadata
    try:
        metadata = joblib.load('models/metadata.pkl')
        print(f'\nMetadata available: {list(metadata.keys())}')
        if 'best_models' in metadata:
            print(f'Best models: {metadata["best_models"]}')
    except:
        print('\nNo metadata file found')

    # Check scaler
    scaler = joblib.load('models/scaler.pkl')
    print(f'\nScaler type: {type(scaler)}')
    
    return model_data, label_encoder

if __name__ == "__main__":
    model_data, label_encoder = diagnose_model()
