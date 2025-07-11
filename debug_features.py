"""
Debug script to test feature extraction and model compatibility
"""
import numpy as np
from audio_processor import AudioProcessor
from model_trainer import EmotionModelTrainer
import librosa
import tempfile
import wave
import os

def test_feature_extraction():
    """Test feature extraction with different audio inputs"""
    processor = AudioProcessor()
    
    print("=== Feature Extraction Debug ===")
    print(f"Sample rate: {processor.sample_rate}")
    print(f"N_MFCC: {processor.n_mfcc}")
    print(f"N_MELS: {processor.n_mels}")
    print(f"Max length: {processor.max_length}")
    
    # Test 1: Generate synthetic audio
    print("\n1. Testing with synthetic audio...")
    duration = 3  # 3 seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    synthetic_audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine wave
    
    features = processor.extract_all_features(synthetic_audio)
    if features is not None:
        print(f"✓ Synthetic audio features: {len(features)} features")
        print(f"Feature shape: {features.shape}")
        print(f"Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
    else:
        print("✗ Synthetic audio feature extraction failed")
    
    # Test 2: Create a temporary WAV file
    print("\n2. Testing with temporary WAV file...")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    
    # Create a simple WAV file
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Convert to 16-bit integers
        audio_data = (synthetic_audio * 32767).astype(np.int16)
        wf.writeframes(audio_data.tobytes())
    
    try:
        features = processor.process_audio_file(temp_file.name)
        if features is not None:
            print(f"✓ WAV file features: {len(features)} features")
            print(f"Feature shape: {features.shape}")
        else:
            print("✗ WAV file feature extraction failed")
    finally:
        os.unlink(temp_file.name)
    
    # Test 3: Test individual feature extraction methods
    print("\n3. Testing individual feature extraction methods...")
    
    # Test MFCC
    try:
        mfcc_features = processor.extract_mfcc_features(synthetic_audio)
        if mfcc_features is not None:
            print(f"✓ MFCC features: {len(mfcc_features)} features")
        else:
            print("✗ MFCC feature extraction failed")
    except Exception as e:
        print(f"✗ MFCC extraction error: {e}")
    
    # Test Spectral
    try:
        spectral_features = processor.extract_spectral_features(synthetic_audio)
        if spectral_features is not None:
            print(f"✓ Spectral features: {len(spectral_features)} features")
        else:
            print("✗ Spectral feature extraction failed")
    except Exception as e:
        print(f"✗ Spectral extraction error: {e}")
    
    # Test Chroma
    try:
        chroma_features = processor.extract_chroma_features(synthetic_audio)
        if chroma_features is not None:
            print(f"✓ Chroma features: {len(chroma_features)} features")
        else:
            print("✗ Chroma feature extraction failed")
    except Exception as e:
        print(f"✗ Chroma extraction error: {e}")
    
    return features

def test_model_compatibility():
    """Test model compatibility with current features"""
    print("\n=== Model Compatibility Test ===")
    
    # Load model
    trainer = EmotionModelTrainer()
    success = trainer.load_models()
    
    if not success:
        print("✗ Could not load models")
        return
    
    print(f"✓ Models loaded successfully")
    print(f"Best model: {trainer.best_model_name}")
    
    # Check model input dimensions
    for name, model in trainer.models.items():
        if hasattr(model, 'coef_'):
            expected_features = model.coef_.shape[1]
            print(f"{name}: expects {expected_features} features")
        elif hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            print(f"{name}: expects {expected_features} features")
    
    # Test prediction with synthetic features
    features = test_feature_extraction()
    if features is not None:
        print(f"\nTesting prediction with {len(features)} features...")
        try:
            emotion, confidence = trainer.predict_emotion(features)
            print(f"✓ Prediction successful: {emotion} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
    else:
        print("Cannot test prediction - feature extraction failed")

if __name__ == "__main__":
    test_feature_extraction()
    test_model_compatibility()
