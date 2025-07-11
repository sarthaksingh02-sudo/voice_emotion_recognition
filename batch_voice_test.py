#!/usr/bin/env python3
"""
Batch Voice Testing Script - Test multiple recordings in sequence
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
            return None
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get emotion label
        emotion = label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities
        }
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return None

def main():
    """Main function for batch testing"""
    print("🎙️  BATCH VOICE EMOTION RECOGNITION TEST")
    print("="*50)
    print("This will help you test multiple emotions in sequence")
    print("Make sure your microphone is working!")
    print()
    
    # Create recordings directory
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    
    # List of emotions to test
    emotions_to_test = [
        ("😊 HAPPY", "Try to sound joyful and excited!"),
        ("😢 SAD", "Try to sound sorrowful and melancholy"),
        ("😠 ANGRY", "Try to sound frustrated and mad"),
        ("😨 FEARFUL", "Try to sound scared and worried"),
        ("😐 NEUTRAL", "Try to sound calm and normal"),
        ("🤢 DISGUSTED", "Try to sound repulsed and disgusted")
    ]
    
    results = []
    
    for i, (emotion_name, instruction) in enumerate(emotions_to_test, 1):
        print(f"\n🎭 TEST {i}/{len(emotions_to_test)}: {emotion_name}")
        print("-" * 30)
        print(f"📝 {instruction}")
        print()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        emotion_file = emotion_name.split()[1].lower()
        filename = f"{recordings_dir}/test_{emotion_file}_{timestamp}.wav"
        
        input("Press Enter when ready to record...")
        
        # Record voice
        if record_voice(filename):
            print("🔍 Analyzing...")
            
            # Analyze emotion
            result = analyze_emotion(filename)
            
            if result:
                predicted_emotion = result['emotion']
                confidence = result['confidence']
                
                # Quick result display
                print(f"   🎯 Predicted: {predicted_emotion.upper()} ({confidence:.1%})")
                
                results.append({
                    'intended': emotion_file,
                    'predicted': predicted_emotion,
                    'confidence': confidence,
                    'filename': filename
                })
            else:
                print("   ❌ Analysis failed")
        else:
            print("   ❌ Recording failed")
        
        # Small pause between recordings
        if i < len(emotions_to_test):
            print("   Preparing for next recording...")
            time.sleep(2)
    
    # Display summary
    print("\n" + "="*50)
    print("📊 BATCH TEST RESULTS SUMMARY")
    print("="*50)
    
    if results:
        print(f"\n✅ Completed {len(results)} tests")
        print("\n📈 Results:")
        
        for i, result in enumerate(results, 1):
            intended = result['intended']
            predicted = result['predicted']
            confidence = result['confidence']
            
            # Check if prediction matches intention
            match = "✅" if intended.lower() in predicted.lower() or predicted.lower() in intended.lower() else "❌"
            
            print(f"   {i}. {intended.upper():10} → {predicted.upper():10} ({confidence:.1%}) {match}")
        
        # Calculate accuracy
        correct = sum(1 for r in results if r['intended'].lower() in r['predicted'].lower() or r['predicted'].lower() in r['intended'].lower())
        accuracy = correct / len(results) if results else 0
        
        print(f"\n📊 Overall Accuracy: {accuracy:.1%} ({correct}/{len(results)})")
        
        # Show files
        print(f"\n💾 Recordings saved in '{recordings_dir}' folder:")
        for result in results:
            print(f"   - {result['filename']}")
    else:
        print("❌ No successful tests completed")
    
    print("\n🎉 Batch testing completed!")

if __name__ == "__main__":
    main()
