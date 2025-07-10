"""
Quick test script to validate improvements
"""
import numpy as np
import time
from pathlib import Path

from audio_processor import AudioProcessor
from data_parser import CremaDataParser
from model_trainer import EmotionModelTrainer

def test_feature_extraction():
    """Test feature extraction on a sample of files"""
    print("=== Testing Feature Extraction ===")
    
    # Test data parser
    data_parser = CremaDataParser()
    data_dir = Path("data/AudioWAV")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Get a sample of files
    audio_files, emotions = data_parser.load_crema_data(data_dir)
    print(f"Found {len(audio_files)} files")
    
    # Test feature extraction on first 100 files
    audio_processor = AudioProcessor()
    sample_files = audio_files[:100]
    
    valid_features = 0
    failed_features = 0
    
    start_time = time.time()
    
    for i, file_path in enumerate(sample_files):
        features = audio_processor.process_audio_file(file_path)
        if features is not None:
            valid_features += 1
        else:
            failed_features += 1
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/100 files - Valid: {valid_features}, Failed: {failed_features}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nFeature Extraction Results:")
    print(f"  Valid features: {valid_features}")
    print(f"  Failed features: {failed_features}")
    print(f"  Success rate: {valid_features / (valid_features + failed_features) * 100:.1f}%")
    print(f"  Processing time: {processing_time:.2f} seconds")
    print(f"  Average time per file: {processing_time / 100:.3f} seconds")
    
    return valid_features, failed_features

def test_quick_training():
    """Test training on a smaller subset"""
    print("\n=== Testing Quick Training ===")
    
    trainer = EmotionModelTrainer()
    data_dir = Path("data/AudioWAV")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Load data
    data_parser = CremaDataParser()
    audio_files, emotions = data_parser.load_crema_data(data_dir)
    
    # Use first 200 files for quick test
    sample_files = audio_files[:200]
    sample_emotions = emotions[:200]
    
    features = []
    valid_labels = []
    
    print("Extracting features from 200 sample files...")
    for i, file_path in enumerate(sample_files):
        feature_vector = trainer.audio_processor.process_audio_file(file_path)
        if feature_vector is not None:
            features.append(feature_vector)
            valid_labels.append(sample_emotions[i])
    
    print(f"Valid samples: {len(features)}")
    
    if len(features) < 50:
        print("Not enough valid samples for training")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(features, valid_labels)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train only a few quick models
    print("\nTraining quick models...")
    start_time = time.time()
    
    # Test optimized Random Forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Quick Random Forest Accuracy: {accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    return accuracy

if __name__ == "__main__":
    print("Voice Emotion Recognition - Quick Test")
    print("=" * 50)
    
    # Test feature extraction
    valid_count, failed_count = test_feature_extraction()
    
    # Test quick training if we have enough valid features
    if valid_count > 50:
        accuracy = test_quick_training()
        print(f"\nQuick test completed with {accuracy:.1%} accuracy")
    else:
        print(f"\nInsufficient valid features ({valid_count}) for training test")
    
    print("\nQuick test completed!")
