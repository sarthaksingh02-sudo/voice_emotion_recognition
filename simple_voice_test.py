#!/usr/bin/env python3
"""
Simple Voice Testing Script for Emotion Recognition System
"""

import numpy as np
import joblib
import pyaudio
import wave
import time
import os
from datetime import datetime
from audio_processor import AudioProcessor

def record_voice(filename, duration=3):
    """Record voice from microphone"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    
    try:
        audio = pyaudio.PyAudio()
        
        # Start recording
        stream = audio.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
        
        print(f"🎤 Recording for {duration} seconds...")
        print("   Say something emotional - be happy, sad, angry, etc!")
        print("   3... 2... 1... GO!")
        
        frames = []
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("🔴 Recording finished!")
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save the recording
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"💾 Recording saved as: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Recording error: {e}")
        return False

def analyze_emotion(audio_file):
    """Analyze emotion from audio file"""
    try:
        # Load models
        model = joblib.load('models/emotion_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        # Initialize audio processor
        processor = AudioProcessor()
        
        # Extract features
        features = processor.process_audio_file(audio_file)
        
        if features is None:
            print("❌ Failed to extract features")
            return None
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get emotion label
        emotion = label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        # Get all probabilities
        emotion_probs = {}
        for i, prob in enumerate(probabilities):
            emotion_name = label_encoder.inverse_transform([i])[0]
            emotion_probs[emotion_name] = prob
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': emotion_probs
        }
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return None

def display_results(results):
    """Display emotion recognition results"""
    if not results:
        return
    
    emotion = results['emotion']
    confidence = results['confidence']
    probabilities = results['probabilities']
    
    emotion_colors = {
        'angry': '🔴',
        'disgust': '🟤', 
        'fearful': '🟡',
        'happy': '🟢',
        'neutral': '⚪',
        'sad': '🔵',
        'surprised': '🟣',
        'calm': '🟢'
    }
    
    print("\n" + "="*50)
    print("🎭 EMOTION RECOGNITION RESULTS")
    print("="*50)
    
    # Main prediction
    emoji = emotion_colors.get(emotion, '⚪')
    print(f"\n🎯 PRIMARY EMOTION: {emoji} {emotion.upper()}")
    print(f"📊 CONFIDENCE: {confidence:.2%}")
    
    # Confidence bar
    bar_length = 30
    filled_length = int(bar_length * confidence)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f"   [{bar}] {confidence:.1%}")
    
    # All emotion probabilities
    print("\n📈 ALL EMOTION PROBABILITIES:")
    sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for emotion_name, prob in sorted_emotions:
        emoji = emotion_colors.get(emotion_name, '⚪')
        bar_length = 20
        filled_length = int(bar_length * prob)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"   {emoji} {emotion_name.upper():12} [{bar}] {prob:.1%}")
    
    # Interpretation
    print(f"\n💡 INTERPRETATION:")
    if confidence > 0.7:
        print("   🟢 High confidence - Clear emotional expression detected")
    elif confidence > 0.5:
        print("   🟡 Moderate confidence - Some emotional indicators present")
    else:
        print("   🔴 Low confidence - Emotion unclear or neutral")
    
    print("\n" + "="*50)

def main():
    """Main function"""
    print("🎙️  SIMPLE VOICE EMOTION RECOGNITION TEST")
    print("="*50)
    print("This will record your voice for 3 seconds and analyze the emotion")
    print("Make sure your microphone is working!")
    print()
    
    # Create recordings directory
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{recordings_dir}/voice_test_{timestamp}.wav"
    
    print("Get ready to express an emotion!")
    print("Try different emotions: happy, sad, angry, fearful, surprised, etc.")
    print()
    input("Press Enter when ready to record...")
    
    # Record voice
    if record_voice(filename):
        print("\n🔍 Analyzing your voice...")
        
        # Analyze emotion
        results = analyze_emotion(filename)
        
        if results:
            display_results(results)
            
            print(f"\n💾 Your recording is saved as: {filename}")
            print("You can test again by running this script multiple times!")
            
        else:
            print("❌ Failed to analyze emotion")
    else:
        print("❌ Recording failed")

if __name__ == "__main__":
    main()
