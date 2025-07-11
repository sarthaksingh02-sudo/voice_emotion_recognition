#!/usr/bin/env python3
"""
Comprehensive system readiness test for emotion recognition system
Tests all major components and functionality
"""

import os
import sys
import time
import traceback
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    required_modules = [
        'numpy', 'pandas', 'sklearn', 'librosa', 'soundfile', 
        'matplotlib', 'seaborn', 'joblib', 'scipy'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            failed_imports.append(module)
    
    # Test custom modules
    custom_modules = [
        'config', 'audio_processor', 'model_trainer', 'real_time_inference'
    ]
    
    for module in custom_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_data_availability():
    """Test if dataset is available"""
    print("\n🗂️  Testing data availability...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("  ✗ Data directory not found")
        return False
    
    # Check for RAVDESS dataset
    audio_wav_dir = data_dir / "AudioWAV"
    if not audio_wav_dir.exists():
        print("  ✗ AudioWAV directory not found")
        return False
    
    # Count audio files
    audio_files = list(audio_wav_dir.glob("*.wav"))
    print(f"  ✓ Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("  ✗ No audio files found")
        return False
    
    return True

def test_model_availability():
    """Test if trained models are available"""
    print("\n🤖 Testing model availability...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("  ✗ Models directory not found")
        return False
    
    required_files = [
        'emotion_model.pkl', 'scaler.pkl', 'label_encoder.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = models_dir / file
        if file_path.exists():
            print(f"  ✓ {file} ({file_path.stat().st_size} bytes)")
        else:
            print(f"  ✗ {file} not found")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_feature_extraction():
    """Test feature extraction on sample audio"""
    print("\n🎵 Testing feature extraction...")
    
    try:
        from audio_processor import AudioProcessor
        
        processor = AudioProcessor()
        print("  ✓ AudioProcessor initialized")
        
        # Find a sample audio file
        audio_files = list(Path("data/AudioWAV").glob("*.wav"))
        if not audio_files:
            print("  ✗ No audio files found for testing")
            return False
        
        sample_file = audio_files[0]
        features = processor.process_audio_file(str(sample_file))
        
        if features is not None:
            print(f"  ✓ Feature extraction successful (shape: {features.shape})")
            return True
        else:
            print("  ✗ Feature extraction failed")
            return False
    
    except Exception as e:
        print(f"  ✗ Error during feature extraction: {e}")
        return False

def test_model_loading():
    """Test if models can be loaded and used"""
    print("\n🧠 Testing model loading...")
    
    try:
        import joblib
        
        # Load models
        model_data = joblib.load('models/emotion_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        print("  ✓ Models loaded successfully")
        print(f"  ✓ Emotion classes: {len(label_encoder.classes_)}")
        print(f"  ✓ Model type: {type(model_data).__name__}")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error loading models: {e}")
        return False

def test_end_to_end_prediction():
    """Test complete end-to-end prediction pipeline"""
    print("\n🎯 Testing end-to-end prediction...")
    
    try:
        import numpy as np
        import joblib
        from audio_processor import AudioProcessor
        
        # Initialize components
        processor = AudioProcessor()
        model_data = joblib.load('models/emotion_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        # Test with sample audio
        audio_files = list(Path("data/AudioWAV").glob("*.wav"))
        sample_file = audio_files[0]
        
        # Extract features
        features = processor.process_audio_file(str(sample_file))
        if features is None:
            print("  ✗ Feature extraction failed")
            return False
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict (model is now directly loaded, not in a dict)
        model = model_data  # model is now directly loaded
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        emotion = label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        print(f"  ✓ Prediction successful: {emotion} (confidence: {confidence:.2f})")
        return True
    
    except Exception as e:
        print(f"  ✗ Error during prediction: {e}")
        traceback.print_exc()
        return False

def test_real_time_components():
    """Test real-time inference components"""
    print("\n🎤 Testing real-time components...")
    
    try:
        from real_time_inference import RealTimeEmotionRecognition
        print("  ✓ RealTimeEmotionRecognition imported successfully")
        
        # Note: We don't instantiate it to avoid pyaudio issues
        print("  ✓ Real-time inference system available")
        return True
    
    except Exception as e:
        print(f"  ✗ Error with real-time components: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running comprehensive system readiness test...")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        ("Import Test", test_imports),
        ("Data Availability Test", test_data_availability),
        ("Model Availability Test", test_model_availability),
        ("Feature Extraction Test", test_feature_extraction),
        ("Model Loading Test", test_model_loading),
        ("End-to-End Prediction Test", test_end_to_end_prediction),
        ("Real-Time Components Test", test_real_time_components)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            failed += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏱️  Duration: {duration:.2f} seconds")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment!")
        print("\nYou can now:")
        print("  • Run real-time emotion recognition: python real_time_inference.py")
        print("  • Test on audio files: python test_final.py")
        print("  • Train new models: python train_model.py")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Please fix the issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
