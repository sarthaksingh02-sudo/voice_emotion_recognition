"""
Training script for emotion recognition model using CREMA-D dataset
"""
from model_trainer import EmotionModelTrainer
from data_parser import CremaDataParser
import numpy as np

def main():
    print("=== Voice Emotion Recognition Training ===")
    print("Using CREMA-D dataset")
    
    # Initialize trainer
    trainer = EmotionModelTrainer()
    
    # Data directory
    data_dir = "data"
    
    try:
        # First test the data parser
        print("\n1. Testing data parser...")
        parser = CremaDataParser()
        audio_files, emotions = parser.load_crema_data(data_dir)
        
        if len(audio_files) == 0:
            print("No audio files found! Please check the data directory.")
            return
        
        print(f"Found {len(audio_files)} audio files")
        print(f"Unique emotions: {list(set(emotions))}")
        
        # Test feature extraction on a few files
        print("\n2. Testing feature extraction...")
        test_files = audio_files[:5]  # Test on first 5 files
        
        for i, audio_file in enumerate(test_files):
            print(f"Processing file {i+1}/5: {audio_file}")
            features = trainer.audio_processor.process_audio_file(audio_file)
            if features is not None:
                print(f"  ✓ Extracted {len(features)} features")
            else:
                print(f"  ✗ Failed to extract features")
        
        # Start training pipeline (with limited data for testing)
        print("\n3. Starting training pipeline...")
        
        # For initial testing, use only a subset of data
        subset_size = min(100, len(audio_files))  # Use 100 files or all if less
        print(f"Using {subset_size} files for initial training...")
        
        # Train the model
        best_model, best_score = trainer.train_complete_pipeline(data_dir)
        
        print(f"\nTraining completed!")
        print(f"Best model: {trainer.best_model_name}")
        print(f"Best accuracy: {best_score:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
