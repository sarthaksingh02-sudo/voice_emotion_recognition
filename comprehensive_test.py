#!/usr/bin/env python3
"""
Comprehensive test script for emotion recognition system
"""

import os
import numpy as np
import joblib
from audio_processor import AudioProcessor
from pathlib import Path
from collections import defaultdict

def run_comprehensive_test():
    """Run comprehensive emotion recognition test"""
    
    # Initialize components
    processor = AudioProcessor()
    model_data = joblib.load('models/emotion_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    model = model_data['model']

    print('🎭 Comprehensive Emotion Recognition Test')
    print('=' * 50)

    # Get sample files from different emotions
    audio_files = list(Path('data/AudioWAV').glob('*.wav'))

    # Test with 50 files for better statistics
    test_files = audio_files[:50]

    emotion_map = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful', 'HAP': 'happy', 
                  'NEU': 'neutral', 'SAD': 'sad', 'SUP': 'surprised', 'CAL': 'calm'}

    results = []
    confusion_data = defaultdict(lambda: defaultdict(int))

    print(f'Testing {len(test_files)} audio files...')

    for i, audio_file in enumerate(test_files, 1):
        if i % 10 == 0:
            print(f'  Processed {i}/{len(test_files)} files...')
        
        # Extract features
        features = processor.process_audio_file(str(audio_file))
        
        if features is not None:
            # Scale and predict
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            predicted_emotion = label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            # Get actual emotion from filename
            filename_parts = audio_file.name.split('_')
            if len(filename_parts) >= 3:
                emotion_code = filename_parts[2]
                actual_emotion = emotion_map.get(emotion_code, 'unknown')
                
                if actual_emotion != 'unknown':
                    results.append({
                        'file': audio_file.name,
                        'actual': actual_emotion,
                        'predicted': predicted_emotion,
                        'confidence': confidence,
                        'correct': actual_emotion == predicted_emotion
                    })
                    
                    confusion_data[actual_emotion][predicted_emotion] += 1

    # Calculate statistics
    correct_predictions = sum(1 for r in results if r['correct'])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f'\n📊 Results Summary:')
    print(f'  Total files tested: {total_predictions}')
    print(f'  Correct predictions: {correct_predictions}')
    print(f'  Accuracy: {accuracy:.2%}')

    # Per-emotion accuracy
    emotion_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for result in results:
        emotion = result['actual']
        emotion_stats[emotion]['total'] += 1
        if result['correct']:
            emotion_stats[emotion]['correct'] += 1

    print(f'\n📈 Per-Emotion Performance:')
    for emotion in sorted(emotion_stats.keys()):
        stats = emotion_stats[emotion]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f'  {emotion.upper()}: {acc:.2%} ({stats["correct"]}/{stats["total"]})')

    # Show some examples
    print(f'\n🎯 Example Predictions:')
    for i, result in enumerate(results[:10]):
        status = '✅' if result['correct'] else '❌'
        print(f'  {result["file"]}: {result["actual"]} → {result["predicted"]} ({result["confidence"]:.2f}) {status}')

    # Show confusion matrix data
    print(f'\n🔄 Common Misclassifications:')
    for actual in sorted(confusion_data.keys()):
        for predicted in sorted(confusion_data[actual].keys()):
            count = confusion_data[actual][predicted]
            if actual != predicted and count > 0:
                print(f'  {actual} → {predicted}: {count} times')

    print(f'\n🎉 Comprehensive test completed!')
    return accuracy

if __name__ == "__main__":
    accuracy = run_comprehensive_test()
