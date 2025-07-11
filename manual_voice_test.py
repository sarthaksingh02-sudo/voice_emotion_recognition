#!/usr/bin/env python3
"""
Manual Voice Testing Script for Emotion Recognition System
Record your voice and test emotion recognition in real-time
"""

import numpy as np
import joblib
import pyaudio
import wave
import time
import os
import threading
from datetime import datetime
from audio_processor import AudioProcessor

class VoiceEmotionTester:
    def __init__(self):
        self.processor = AudioProcessor()
        self.load_models()
        self.setup_audio()
        self.recording = False
        self.emotion_colors = {
            'angry': '🔴',
            'disgust': '🟤', 
            'fearful': '🟡',
            'happy': '🟢',
            'neutral': '⚪',
            'sad': '🔵',
            'surprised': '🟣',
            'calm': '🟢'
        }
        
    def load_models(self):
        """Load the trained models"""
        try:
            self.model = joblib.load('models/emotion_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

    def setup_audio(self):
        """Setup audio recording parameters"""
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 22050
        self.RECORD_SECONDS = 3  # Record for 3 seconds
        
    def record_voice(self, filename):
        """Record voice from microphone"""
        try:
            audio = pyaudio.PyAudio()
            
            # Start recording
            stream = audio.open(format=self.FORMAT,
                              channels=self.CHANNELS,
                              rate=self.RATE,
                              input=True,
                              frames_per_buffer=self.CHUNK)
            
            print(f"🎤 Recording for {self.RECORD_SECONDS} seconds...")
            print("   Say something with emotion!")
            
            frames = []
            for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK)
                frames.append(data)
            
            print("🔴 Recording finished!")
            
            # Stop recording
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Save the recording
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return False
    
    def analyze_emotion(self, audio_file):
        """Analyze emotion from audio file"""
        try:
            # Extract features
            features = self.processor.process_audio_file(audio_file)
            
            if features is None:
                print("❌ Failed to extract features")
                return None
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get emotion label
            emotion = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            # Get all probabilities for detailed analysis
            emotion_probs = {}
            for i, prob in enumerate(probabilities):
                emotion_name = self.label_encoder.inverse_transform([i])[0]
                emotion_probs[emotion_name] = prob
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': emotion_probs
            }
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None
    
    def display_results(self, results):
        """Display emotion recognition results"""
        if not results:
            return
        
        emotion = results['emotion']
        confidence = results['confidence']
        probabilities = results['probabilities']
        
        print("\n" + "="*50)
        print("🎭 EMOTION RECOGNITION RESULTS")
        print("="*50)
        
        # Main prediction
        emoji = self.emotion_colors.get(emotion, '⚪')
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
            emoji = self.emotion_colors.get(emotion_name, '⚪')
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
    
    def run_manual_test(self):
        """Run interactive manual testing"""
        print("🎙️  MANUAL VOICE EMOTION RECOGNITION TEST")
        print("="*50)
        print("This tool will record your voice and analyze the emotion")
        print("Make sure your microphone is working!")
        print()
        
        # Create recordings directory
        recordings_dir = "recordings"
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)
        
        test_count = 1
        
        while True:
            print(f"\n🎭 TEST #{test_count}")
            print("-" * 20)
            
            # Get user input
            print("\nChoose an option:")
            print("1. Record and analyze your voice")
            print("2. Test with existing audio file")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                # Record voice
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{recordings_dir}/test_{timestamp}.wav"
                
                input("\nPress Enter when ready to record...")
                
                if self.record_voice(filename):
                    print(f"💾 Recording saved as: {filename}")
                    
                    # Analyze emotion
                    print("\n🔍 Analyzing emotion...")
                    results = self.analyze_emotion(filename)
                    
                    if results:
                        self.display_results(results)
                        
                        # Ask if user wants to keep the recording
                        keep = input("\nKeep this recording? (y/n): ").strip().lower()
                        if keep != 'y':
                            try:
                                os.remove(filename)
                                print("🗑️  Recording deleted")
                            except:
                                pass
                    else:
                        print("❌ Failed to analyze emotion")
                        
            elif choice == '2':
                # Test with existing file
                filepath = input("Enter audio file path: ").strip()
                
                if os.path.exists(filepath):
                    print(f"\n🔍 Analyzing {filepath}...")
                    results = self.analyze_emotion(filepath)
                    
                    if results:
                        self.display_results(results)
                    else:
                        print("❌ Failed to analyze emotion")
                else:
                    print("❌ File not found")
                    
            elif choice == '3':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice")
                
            test_count += 1
    
    def run_continuous_test(self):
        """Run continuous emotion monitoring"""
        print("🎙️  CONTINUOUS EMOTION MONITORING")
        print("="*50)
        print("This will continuously record and analyze your voice")
        print("Press Ctrl+C to stop")
        print()
        
        recordings_dir = "recordings"
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)
        
        try:
            count = 1
            while True:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{recordings_dir}/continuous_{timestamp}.wav"
                
                print(f"\n🎤 Recording #{count} - Speak now...")
                
                if self.record_voice(filename):
                    results = self.analyze_emotion(filename)
                    
                    if results:
                        emotion = results['emotion']
                        confidence = results['confidence']
                        emoji = self.emotion_colors.get(emotion, '⚪')
                        
                        print(f"   {emoji} {emotion.upper()} ({confidence:.1%})")
                    
                    # Clean up temporary file
                    try:
                        os.remove(filename)
                    except:
                        pass
                        
                count += 1
                time.sleep(1)  # Short pause between recordings
                
        except KeyboardInterrupt:
            print("\n\n👋 Continuous monitoring stopped")

def main():
    """Main function"""
    try:
        tester = VoiceEmotionTester()
        
        print("Choose testing mode:")
        print("1. Manual testing (record when you want)")
        print("2. Continuous monitoring (automatic recording)")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            tester.run_manual_test()
        elif choice == '2':
            tester.run_continuous_test()
        else:
            print("❌ Invalid choice")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have a microphone connected and pyaudio installed")
        print("To install pyaudio: pip install pyaudio")

if __name__ == "__main__":
    main()
