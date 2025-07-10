"""
Audio preprocessing and feature extraction for emotion recognition
"""
import numpy as np
import librosa
import soundfile as sf
from python_speech_features import mfcc
from scipy.stats import skew, kurtosis
import warnings
import hashlib
import os
from pathlib import Path
warnings.filterwarnings('ignore')

from config import *

class AudioProcessor:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.n_mfcc = N_MFCC
        self.n_mels = N_MELS
        self.max_length = MAX_AUDIO_LENGTH
        
        # Create cache directory
        self.cache_dir = Path("temp/features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track processing stats
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'feature_extraction_success': 0,
            'feature_extraction_failed': 0,
            'load_failed': 0,
            'preprocess_failed': 0
        }
    
    def _get_cache_path(self, file_path):
        """Generate cache file path based on audio file path"""
        # Create hash of file path for unique cache filename
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        return self.cache_dir / f"{file_hash}.npy"
    
    def _load_cached_features(self, file_path):
        """Load features from cache if available"""
        if not CACHE_FEATURES:
            return None
            
        cache_path = self._get_cache_path(file_path)
        if cache_path.exists():
            try:
                features = np.load(cache_path)
                self.stats['cache_hits'] += 1
                return features
            except Exception as e:
                # If cache file is corrupted, delete it
                cache_path.unlink(missing_ok=True)
                return None
        return None
    
    def _save_cached_features(self, file_path, features):
        """Save features to cache"""
        if not CACHE_FEATURES or features is None:
            return
            
        cache_path = self._get_cache_path(file_path)
        try:
            np.save(cache_path, features)
        except Exception as e:
            print(f"Warning: Could not save cache for {file_path}: {e}")
    
    def get_processing_stats(self):
        """Get processing statistics"""
        return self.stats.copy()
        
    def load_audio(self, file_path, duration=None):
        """Load audio file with specified duration"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None, None
    
    def preprocess_audio(self, audio):
        """Preprocess audio signal with robust error handling"""
        if audio is None:
            return None
            
        try:
            # Check for empty audio
            if len(audio) == 0:
                return None
            
            # Remove NaN and infinite values
            audio = audio[~np.isnan(audio)]
            audio = audio[~np.isinf(audio)]
            
            if len(audio) == 0:
                return None
            
            # Normalize audio (handle zero variance)
            if np.std(audio) > 0:
                audio = librosa.util.normalize(audio)
            
            # Remove silence with better parameters
            try:
                audio, _ = librosa.effects.trim(audio, top_db=20)
            except:
                # If trimming fails, use original audio
                pass
            
            # Check if audio is too short
            if len(audio) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                return None
            
            # Pad or truncate to fixed length
            target_length = self.sample_rate * self.max_length
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
                
            return audio
            
        except Exception as e:
            return None
    
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
        """Extract pitch-based features with better error handling"""
        if audio is None:
            return None
            
        try:
            # Fundamental frequency with better parameters
            pitches, magnitudes = librosa.piptrack(
                y=audio, 
                sr=self.sample_rate, 
                threshold=0.1,
                fmin=50,
                fmax=2000
            )
            
            # Extract pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) == 0:
                # Use alternative method if piptrack fails
                try:
                    f0 = librosa.yin(audio, fmin=50, fmax=2000, sr=self.sample_rate)
                    pitch_values = f0[f0 > 0]
                except:
                    return np.zeros(6)
            
            if len(pitch_values) == 0:
                return np.zeros(6)
            
            pitch_values = np.array(pitch_values)
            
            # Statistical measures with safe calculations
            features = [
                np.mean(pitch_values) if len(pitch_values) > 0 else 0,
                np.std(pitch_values) if len(pitch_values) > 1 else 0,
                np.max(pitch_values) if len(pitch_values) > 0 else 0,
                np.min(pitch_values) if len(pitch_values) > 0 else 0,
                skew(pitch_values) if len(pitch_values) > 2 else 0,
                kurtosis(pitch_values) if len(pitch_values) > 3 else 0
            ]
            
            # Check for NaN values
            features = np.array(features)
            if np.isnan(features).any():
                return np.zeros(6)
            
            return features
            
        except Exception as e:
            # Return zeros if pitch extraction fails
            return np.zeros(6)
    
    def extract_all_features(self, audio):
        """Extract optimized audio features with robust error handling"""
        if audio is None:
            return None
            
        try:
            # Extract core features (faster set)
            mfcc_features = self.extract_mfcc_features(audio)
            spectral_features = self.extract_spectral_features(audio)
            chroma_features = self.extract_chroma_features(audio)
            
            # Check core features
            if mfcc_features is None or spectral_features is None or chroma_features is None:
                return None
            
            # Optional features (based on config)
            feature_list = [mfcc_features, spectral_features, chroma_features]
            
            if ENABLE_MEL_FEATURES:
                mel_features = self.extract_mel_features(audio)
                if mel_features is not None:
                    feature_list.append(mel_features)
            
            if ENABLE_PITCH_FEATURES:
                pitch_features = self.extract_pitch_features(audio)
                if pitch_features is not None:
                    feature_list.append(pitch_features)
                else:
                    # Add zeros if pitch extraction fails
                    feature_list.append(np.zeros(6))
            
            # Combine all features
            all_features = np.concatenate(feature_list)
            
            # Check for NaN or infinite values
            if np.isnan(all_features).any() or np.isinf(all_features).any():
                return None
            
            return all_features
            
        except Exception as e:
            # Return None if any feature extraction fails
            return None
    
    def process_audio_file(self, file_path, duration=None):
        """Process audio file and extract features with caching"""
        self.stats['total_processed'] += 1
        
        # Try to load from cache first
        cached_features = self._load_cached_features(file_path)
        if cached_features is not None:
            return cached_features
        
        # Load audio
        audio, sr = self.load_audio(file_path, duration)
        if audio is None:
            self.stats['load_failed'] += 1
            return None
            
        # Preprocess
        audio = self.preprocess_audio(audio)
        if audio is None:
            self.stats['preprocess_failed'] += 1
            return None
        
        # Extract features
        features = self.extract_all_features(audio)
        
        if features is not None:
            self.stats['feature_extraction_success'] += 1
            # Save to cache
            self._save_cached_features(file_path, features)
        else:
            self.stats['feature_extraction_failed'] += 1
        
        return features
    
    def save_audio(self, audio, file_path):
        """Save audio to file"""
        sf.write(file_path, audio, self.sample_rate)
    
    def get_feature_names(self):
        """Get names of all features based on enabled features"""
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
        
        # Optional Mel features
        if ENABLE_MEL_FEATURES:
            for i in range(self.n_mels):
                feature_names.extend([
                    f'mel_{i}_mean', f'mel_{i}_std',
                    f'mel_{i}_skew', f'mel_{i}_kurt'
                ])
        
        # Optional Pitch features
        if ENABLE_PITCH_FEATURES:
            feature_names.extend([
                'pitch_mean', 'pitch_std', 'pitch_max',
                'pitch_min', 'pitch_skew', 'pitch_kurt'
            ])
        
        return feature_names
