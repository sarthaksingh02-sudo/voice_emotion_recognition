"""
RAVDESS dataset parser for emotion recognition
"""
import os
import glob
from pathlib import Path
import numpy as np

class RavdessDataParser:
    def __init__(self):
        # RAVDESS emotion mapping
        self.emotion_map = {
            '01': 'neutral',
            '02': 'calm', 
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
    def parse_filename(self, filename):
        """Parse RAVDESS filename to extract emotion
        
        Filename format: Modality-Vocal channel-Emotion-Emotional intensity-Statement-Repetition-Actor
        Example: 03-01-06-01-02-01-12.wav
        """
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            return self.emotion_map.get(emotion_code, 'unknown')
        return 'unknown'
    
    def load_ravdess_data(self, data_dir):
        """Load RAVDESS dataset from directory structure"""
        data_dir = Path(data_dir)
        audio_files = []
        emotions = []
        
        print(f"Loading RAVDESS data from: {data_dir}")
        
        # Handle both speech and song directories
        for subdir in data_dir.iterdir():
            if subdir.is_dir() and ('Speech' in subdir.name or 'Song' in subdir.name):
                print(f"Processing {subdir.name}...")
                
                # Look for Actor directories
                for actor_dir in subdir.iterdir():
                    if actor_dir.is_dir() and actor_dir.name.startswith('Actor_'):
                        print(f"  Processing {actor_dir.name}...")
                        
                        # Get all wav files in actor directory
                        wav_files = list(actor_dir.glob('*.wav'))
                        
                        for wav_file in wav_files:
                            emotion = self.parse_filename(wav_file.name)
                            if emotion != 'unknown':
                                audio_files.append(str(wav_file))
                                emotions.append(emotion)
                            else:
                                print(f"    Warning: Could not parse emotion from {wav_file.name}")
        
        print(f"Found {len(audio_files)} audio files")
        
        # Print emotion distribution
        if emotions:
            unique_emotions, counts = np.unique(emotions, return_counts=True)
            print("\nEmotion distribution:")
            for emotion, count in zip(unique_emotions, counts):
                print(f"  {emotion}: {count}")
        
        return audio_files, emotions
    
    def load_combined_data(self, data_dir):
        """Load combined RAVDESS + CREMA-D data"""
        data_dir = Path(data_dir)
        audio_files = []
        emotions = []
        
        print("Loading combined RAVDESS + CREMA-D dataset...")
        
        # Load RAVDESS data
        ravdess_files, ravdess_emotions = self.load_ravdess_data(data_dir)
        audio_files.extend(ravdess_files)
        emotions.extend(ravdess_emotions)
        
        # Load CREMA-D data if available
        crema_dir = data_dir / 'AudioWAV'
        if crema_dir.exists():
            print(f"Also loading CREMA-D data from: {crema_dir}")
            from data_parser import CremaDataParser
            crema_parser = CremaDataParser()
            crema_files, crema_emotions = crema_parser.load_crema_data(crema_dir)
            audio_files.extend(crema_files)
            emotions.extend(crema_emotions)
        
        print(f"Total combined dataset: {len(audio_files)} files")
        
        # Print final emotion distribution
        if emotions:
            unique_emotions, counts = np.unique(emotions, return_counts=True)
            print("\nFinal emotion distribution:")
            for emotion, count in zip(unique_emotions, counts):
                print(f"  {emotion}: {count}")
        
        return audio_files, emotions
