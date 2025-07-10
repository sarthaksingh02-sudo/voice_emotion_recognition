"""
Real-time emotion recognition from microphone input
"""
import pyaudio
import numpy as np
import wave
import tempfile
import os
from model_trainer import EmotionModelTrainer
from audio_processor import AudioProcessor
import time

class RealTimeEmotionRecognition:
    def __init__(self):
        self.trainer = EmotionModelTrainer()
        self.audio_processor = AudioProcessor()
        
        # Audio recording parameters
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 22050
        self.record_seconds = 3  # Record 3 seconds for each prediction
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Load trained model
        self.load_model()
    
    def load_model(self):
        """Load the trained emotion recognition model"""
        try:
            success = self.trainer.load_models()
            if success:
                print("✓ Model loaded successfully!")
                print(f"Using: {self.trainer.best_model_name}")
            else:
                print("✗ Failed to load model. Please train the model first.")
                return False
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Please run 'python train_model.py' first to train the model.")
            return False
        return True
    
    def record_audio(self):
        """Record audio from microphone"""
        stream = self.audio.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk,
            input=True
        )
        
        print(f"Recording for {self.record_seconds} seconds...")
        frames = []
        
        for _ in range(0, int(self.sample_rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        return b''.join(frames)
    
    def save_temp_audio(self, audio_data):
        """Save audio data to temporary file"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.sample_format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        
        return temp_file.name
    
    def predict_emotion(self, audio_file):
        """Predict emotion from audio file"""
        try:
            # Extract features
            features = self.audio_processor.process_audio_file(audio_file)
            if features is None:
                return None, 0.0
            
            # Predict emotion
            emotion, confidence = self.trainer.predict_emotion(features)
            return emotion, confidence
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, 0.0
    
    def run_continuous_recognition(self):
        """Run continuous emotion recognition"""
        if not self.load_model():
            return
        
        print("\n=== Real-Time Emotion Recognition ===")
        print("Press Ctrl+C to stop")
        print("Speak into your microphone...")
        
        try:
            while True:
                # Record audio
                audio_data = self.record_audio()
                
                # Save to temporary file
                temp_file = self.save_temp_audio(audio_data)
                
                try:
                    # Predict emotion
                    emotion, confidence = self.predict_emotion(temp_file)
                    
                    if emotion:
                        print(f"Detected emotion: {emotion.upper()} (confidence: {confidence:.2f})")
                        
                        # Add confidence threshold
                        if confidence < 0.3:
                            print("(Low confidence - unclear emotion)")
                    else:
                        print("Could not detect emotion")
                
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file)
                
                # Wait a moment before next recording
                print("Waiting for next recording...\n")
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nStopping emotion recognition...")
        finally:
            self.audio.terminate()
    
    def test_single_prediction(self, audio_file_path):
        """Test prediction on a single audio file"""
        if not self.load_model():
            return
        
        print(f"\nTesting emotion recognition on: {audio_file_path}")
        
        emotion, confidence = self.predict_emotion(audio_file_path)
        
        if emotion:
            print(f"Predicted emotion: {emotion.upper()}")
            print(f"Confidence: {confidence:.2f}")
        else:
            print("Could not detect emotion")

def main():
    recognizer = RealTimeEmotionRecognition()
    
    print("Choose an option:")
    print("1. Real-time emotion recognition from microphone")
    print("2. Test on audio file")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            recognizer.run_continuous_recognition()
        elif choice == "2":
            file_path = input("Enter audio file path: ").strip()
            recognizer.test_single_prediction(file_path)
        else:
            print("Invalid choice")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
