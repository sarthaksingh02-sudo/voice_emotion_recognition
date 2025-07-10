"""
Audio preprocessing and feature extraction for emotion recognition
"""
import numpy as np
import librosa
import soundfile as sf
from python_speech_features import mfcc
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

from config import *

class AudioProcessor:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.n_mfcc = N_MFCC
        self.n_mels = N_MELS
        self.max_length = MAX_AUDIO_LENGTH
        
    def load_audio(self, file_path, duration=None):
        """Load audio file with specified duration"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None, None
    
    def preprocess_audio(self, audio):
        """Preprocess audio signal"""
        if audio is None:
            return None
            
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Remove silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Pad or truncate to fixed length
        target_length = self.sample_rate * self.max_length
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            
        return audio
    
    def extract_mfcc_features(self, audio):
        """Extract MFCC features"""
        if audio is None:
            return None
            
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        
        # Statistical features
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_skew = skew(mfccs, axis=1)
        mfcc_kurt = kurtosis(mfccs, axis=1)
        
        return np.concatenate([mfcc_mean, mfcc_std, mfcc_skew, mfcc_kurt])
    
    def extract_spectral_features(self, audio):
        """Extract spectral features"""
        if audio is None:
            return None
            
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Statistical measures
        features = []
        for feature in [spectral_centroids, spectral_rolloff, spectral_bandwidth, zero_crossing_rate]:
            features.extend([
                np.mean(feature),
                np.std(feature),
                np.max(feature),
                np.min(feature),
                skew(feature),
                kurtosis(feature)
            ])
        
        return np.array(features)
    
    def extract_chroma_features(self, audio):
        """Extract chroma features"""
        if audio is None:
            return None
            
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        
        # Statistical measures
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        chroma_skew = skew(chroma, axis=1)
        chroma_kurt = kurtosis(chroma, axis=1)
        
        return np.concatenate([chroma_mean, chroma_std, chroma_skew, chroma_kurt])
    
    def extract_mel_features(self, audio):
        """Extract mel-spectrogram features"""
        if audio is None:
            return None
            
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Statistical measures
        mel_mean = np.mean(mel_spec_db, axis=1)
        mel_std = np.std(mel_spec_db, axis=1)
        mel_skew = skew(mel_spec_db, axis=1)
        mel_kurt = kurtosis(mel_spec_db, axis=1)
        
        return np.concatenate([mel_mean, mel_std, mel_skew, mel_kurt])
    
    def extract_pitch_features(self, audio):
        """Extract pitch-based features"""
        if audio is None:
            return None
            
        # Fundamental frequency
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        
        # Extract pitch values
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) == 0:
            return np.zeros(6)
        
        pitch_values = np.array(pitch_values)
        
        # Statistical measures
        features = [
            np.mean(pitch_values),
            np.std(pitch_values),
            np.max(pitch_values),
            np.min(pitch_values),
            skew(pitch_values),
            kurtosis(pitch_values)
        ]
        
        return np.array(features)
    
    def extract_all_features(self, audio):
        """Extract all audio features"""
        if audio is None:
            return None
            
        # Extract different types of features
        mfcc_features = self.extract_mfcc_features(audio)
        spectral_features = self.extract_spectral_features(audio)
        chroma_features = self.extract_chroma_features(audio)
        mel_features = self.extract_mel_features(audio)
        pitch_features = self.extract_pitch_features(audio)
        
        # Combine all features
        all_features = np.concatenate([
            mfcc_features,
            spectral_features,
            chroma_features,
            mel_features,
            pitch_features
        ])
        
        return all_features
    
    def process_audio_file(self, file_path, duration=None):
        """Process audio file and extract features"""
        # Load audio
        audio, sr = self.load_audio(file_path, duration)
        if audio is None:
            return None
            
        # Preprocess
        audio = self.preprocess_audio(audio)
        
        # Extract features
        features = self.extract_all_features(audio)
        
        return features
    
    def save_audio(self, audio, file_path):
        """Save audio to file"""
        sf.write(file_path, audio, self.sample_rate)
    
    def get_feature_names(self):
        """Get names of all features"""
        feature_names = []
        
        # MFCC features (13 * 4 = 52)
        for i in range(self.n_mfcc):
            feature_names.extend([
                f'mfcc_{i}_mean', f'mfcc_{i}_std', 
                f'mfcc_{i}_skew', f'mfcc_{i}_kurt'
            ])
        
        # Spectral features (4 * 6 = 24)
        spectral_types = ['centroid', 'rolloff', 'bandwidth', 'zcr']
        for spec_type in spectral_types:
            feature_names.extend([
                f'{spec_type}_mean', f'{spec_type}_std', f'{spec_type}_max',
                f'{spec_type}_min', f'{spec_type}_skew', f'{spec_type}_kurt'
            ])
        
        # Chroma features (12 * 4 = 48)
        for i in range(12):
            feature_names.extend([
                f'chroma_{i}_mean', f'chroma_{i}_std',
                f'chroma_{i}_skew', f'chroma_{i}_kurt'
            ])
        
        # Mel features (128 * 4 = 512)
        for i in range(self.n_mels):
            feature_names.extend([
                f'mel_{i}_mean', f'mel_{i}_std',
                f'mel_{i}_skew', f'mel_{i}_kurt'
            ])
        
        # Pitch features (6)
        feature_names.extend([
            'pitch_mean', 'pitch_std', 'pitch_max',
            'pitch_min', 'pitch_skew', 'pitch_kurt'
        ])
        
        return feature_names
