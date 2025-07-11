"""
Improved real-time emotion recognition with Voice Activity Detection (VAD)
"""
import pyaudio
import numpy as np
import wave
import tempfile
import os
import threading
import time
from collections import deque
from model_trainer import EmotionModelTrainer
from audio_processor import AudioProcessor
import webrtcvad

class ImprovedRealTimeEmotionRecognition:
    def __init__(self):
        self.trainer = EmotionModelTrainer()
        self.audio_processor = AudioProcessor()
        
        # Audio recording parameters
        self.chunk = 480  # 30ms chunks for VAD
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.min_speech_duration = 1.0  # Minimum speech duration in seconds
        self.max_speech_duration = 10.0  # Maximum speech duration in seconds
        self.silence_threshold = 0.5  # Silence duration to stop recording
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
        
        # Load trained model
        self.load_model()
        
        # Buffer for audio data
        self.audio_buffer = deque(maxlen=int(self.sample_rate * self.max_speech_duration / self.chunk))
        
    def load_model(self):
        """Load the trained emotion recognition model"""
        try:
            success = self.trainer.load_models()
            if success:
                print("✓ Model loaded successfully!")
                print(f"Using: {self.trainer.best_model_name}")
                print(f"Accuracy: {self.trainer.best_score:.2%}")
            else:
                print("✗ Failed to load model. Please train the model first.")
                return False
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
        return True
    
    def is_speech(self, frame):
        """Check if audio frame contains speech using VAD"""
        try:
            # Convert frame to bytes if needed
            if isinstance(frame, np.ndarray):
                frame = frame.tobytes()
            return self.vad.is_speech(frame, self.sample_rate)
        except:
            return False
    
    def record_until_silence(self):
        """Record audio until silence is detected"""
        stream = self.audio.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk,
            input=True
        )
        
        print("🎤 Listening... Start speaking!")
        
        frames = []
        speech_detected = False
        silence_start = None
        recording_start = time.time()
        
        try:
            while True:
                data = stream.read(self.chunk)
                frames.append(data)
                
                # Convert to numpy array for VAD
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                # Check if this chunk contains speech
                is_speech_chunk = self.is_speech(data)
                
                if is_speech_chunk:
                    speech_detected = True
                    silence_start = None
                    print("🗣️ Speech detected...", end="\r")
                else:
                    if speech_detected:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.silence_threshold:
                            print("\n🔇 Silence detected. Processing...")
                            break
                
                # Check maximum duration
                if time.time() - recording_start > self.max_speech_duration:
                    print(f"\n⏰ Maximum recording time ({self.max_speech_duration}s) reached.")
                    break
                    
                # Check minimum duration before allowing stop
                if speech_detected and time.time() - recording_start < self.min_speech_duration:
                    silence_start = None
                    
        finally:
            stream.stop_stream()
            stream.close()
        
        # Only process if we detected speech
        if speech_detected and frames:
            duration = time.time() - recording_start
            print(f"📊 Recorded {duration:.1f}s of audio")
            return b''.join(frames)
        else:
            print("❌ No speech detected.")
            return None
    
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
    
    def get_confidence_emoji(self, confidence):
        """Get emoji based on confidence level"""
        if confidence >= 0.8:
            return "🟢"
        elif confidence >= 0.6:
            return "🟡"
        elif confidence >= 0.4:
            return "🟠"
        else:
            return "🔴"
    
    def get_emotion_emoji(self, emotion):
        """Get emoji for emotion"""
        emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😡',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐',
            'surprise': '😮'
        }
        return emotion_emojis.get(emotion.lower(), '❓')
    
    def run_continuous_recognition(self):
        """Run continuous emotion recognition with VAD"""
        if not self.load_model():
            return
        
        print("\n" + "="*50)
        print("🎯 IMPROVED REAL-TIME EMOTION RECOGNITION")
        print("="*50)
        print("🎤 Voice Activity Detection: ON")
        print("📊 Enhanced Feature Extraction: ON")
        print("🧠 Deep Learning Model: MLP Neural Network")
        print("🎯 Accuracy: {:.1%}".format(self.trainer.best_score))
        print("\n📝 Instructions:")
        print("• Start speaking when you see 'Listening...'")
        print("• Stop speaking and wait for results")
        print("• Press Ctrl+C to exit")
        print("="*50)
        
        try:
            session_count = 0
            while True:
                session_count += 1
                print(f"\n🔄 Session {session_count}")
                
                # Record audio with VAD
                audio_data = self.record_until_silence()
                
                if audio_data is None:
                    print("⏳ No speech detected. Trying again...")
                    continue
                
                # Save to temporary file
                temp_file = self.save_temp_audio(audio_data)
                
                try:
                    # Predict emotion
                    print("🧠 Analyzing emotion...")
                    emotion, confidence = self.predict_emotion(temp_file)
                    
                    if emotion:
                        confidence_emoji = self.get_confidence_emoji(confidence)
                        emotion_emoji = self.get_emotion_emoji(emotion)
                        
                        print(f"\n🎯 RESULT:")
                        print(f"   Emotion: {emotion_emoji} {emotion.upper()}")
                        print(f"   Confidence: {confidence_emoji} {confidence:.1%}")
                        
                        if confidence < 0.4:
                            print("   ⚠️  Low confidence - unclear emotion")
                        elif confidence >= 0.8:
                            print("   ✅ High confidence result!")
                    else:
                        print("❌ Could not detect emotion")
                
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file)
                
                print("\n" + "-"*30)
        
        except KeyboardInterrupt:
            print("\n\n👋 Stopping emotion recognition...")
        finally:
            self.audio.terminate()

def main():
    try:
        # Install webrtcvad if not available
        import webrtcvad
    except ImportError:
        print("Installing webrtcvad for voice activity detection...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'webrtcvad'])
        import webrtcvad
    
    recognizer = ImprovedRealTimeEmotionRecognition()
    recognizer.run_continuous_recognition()

if __name__ == "__main__":
    main()
