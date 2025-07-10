"""
Data parser for CREMA-D dataset
"""
import re
from pathlib import Path
import pandas as pd

class CremaDataParser:
    def __init__(self):
        # CREMA-D emotion mapping
        self.emotion_mapping = {
            'ANG': 'angry',
            'DIS': 'disgust', 
            'FEA': 'fear',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
        
    def parse_filename(self, filename):
        """
        Parse CREMA-D filename to extract emotion
        Format: ActorID_Sentence_Emotion_Intensity.wav
        Example: 1001_DFA_ANG_XX.wav
        """
        # Remove file extension
        name = filename.stem if hasattr(filename, 'stem') else filename.split('.')[0]
        
        # Split by underscore
        parts = name.split('_')
        
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in self.emotion_mapping:
                return self.emotion_mapping[emotion_code]
        
        return None
    
    def load_crema_data(self, data_dir):
        """
        Load CREMA-D dataset from directory
        """
        data_dir = Path(data_dir)
        audio_files = []
        emotions = []
        
        # Look for AudioWAV subdirectory
        audio_wav_dir = data_dir / "AudioWAV"
        if audio_wav_dir.exists():
            wav_files = list(audio_wav_dir.glob("*.wav"))
        else:
            wav_files = list(data_dir.glob("*.wav"))
        
        print(f"Found {len(wav_files)} audio files")
        
        for wav_file in wav_files:
            emotion = self.parse_filename(wav_file)
            if emotion:
                audio_files.append(str(wav_file))
                emotions.append(emotion)
        
        print(f"Successfully parsed {len(audio_files)} files with emotions")
        
        # Print emotion distribution
        emotion_counts = pd.Series(emotions).value_counts()
        print("\nEmotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count}")
        
        return audio_files, emotions
    
    def get_emotion_stats(self, emotions):
        """Get statistics about emotion distribution"""
        emotion_counts = pd.Series(emotions).value_counts()
        return emotion_counts.to_dict()
