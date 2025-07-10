"""
Machine learning model trainer for emotion recognition
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# TensorFlow not available for Python 3.13 yet
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *
from audio_processor import AudioProcessor
from data_parser import CremaDataParser

class EmotionModelTrainer:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        
    def load_data_from_directory(self, data_dir):
        """Load audio data from directory structure"""
        data_dir = Path(data_dir)
        features = []
        labels = []
        
        print("Loading audio files and extracting features...")
        
        for emotion_dir in data_dir.iterdir():
            if emotion_dir.is_dir():
                emotion_label = emotion_dir.name.lower()
                print(f"Processing {emotion_label} files...")
                
                audio_files = [f for f in emotion_dir.glob("*.wav")]
                
                for audio_file in audio_files:
                    try:
                        # Extract features
                        feature_vector = self.audio_processor.process_audio_file(str(audio_file))
                        if feature_vector is not None:
                            features.append(feature_vector)
                            labels.append(emotion_label)
                    except Exception as e:
                        print(f"Error processing {audio_file}: {e}")
                        continue
        
        return np.array(features), np.array(labels)
    
    def prepare_data(self, features, labels):
        """Prepare data for training"""
        # Convert to numpy array if needed
        features = np.array(features)
        
        # Handle NaN values
        print(f"Features shape before imputation: {features.shape}")
        print(f"NaN count: {np.isnan(features).sum()}")
        
        # Impute missing values
        features_imputed = self.imputer.fit_transform(features)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_imputed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_encoded, 
            test_size=TEST_SIZE, 
            random_state=42, 
            stratify=labels_encoded
        )
        
        print(f"Final training shape: {X_train.shape}")
        print(f"Final test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_traditional_models(self, X_train, X_test, y_train, y_test):
        """Train traditional ML models"""
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Update best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                self.best_model_name = name
        
        self.models.update(results)
        return results
    
    # Neural network methods commented out - TensorFlow not available
    # def create_neural_network(self, input_dim, num_classes):
    #     """Create deep neural network model"""
    #     pass
    # 
    # def train_neural_network(self, X_train, X_test, y_train, y_test):
    #     """Train deep neural network"""
    #     pass
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for best models"""
        print("Performing hyperparameter tuning...")
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        rf_grid.fit(X_train, y_train)
        
        print(f"Best RF parameters: {rf_grid.best_params_}")
        print(f"Best RF score: {rf_grid.best_score_:.4f}")
        
        return rf_grid.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        if model_name == "Neural Network":
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = model.predict(X_test)
        
        # Print classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()  # Close plot instead of showing it
        
        return accuracy_score(y_test, y_pred)
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("Saving models and preprocessing objects...")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        joblib.dump(self.imputer, MODEL_DIR / 'imputer.pkl')
        
        # Save best traditional model
        if self.best_model_name != "Neural Network":
            joblib.dump(self.best_model, MODEL_PATH)
        
        # Neural network model saving skipped
        
        # Save model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'emotion_labels': list(self.label_encoder.classes_),
            'feature_names': self.audio_processor.get_feature_names()
        }
        
        with open(MODEL_DIR / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Models saved successfully!")
        print(f"Best model: {self.best_model_name} with accuracy: {self.best_score:.4f}")
    
    def load_models(self):
        """Load saved models"""
        try:
            self.scaler = joblib.load(SCALER_PATH)
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            self.imputer = joblib.load(MODEL_DIR / 'imputer.pkl')
            
            # Load metadata
            with open(MODEL_DIR / 'metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            self.best_model_name = metadata['best_model_name']
            self.best_score = metadata['best_score']
            
            # Load appropriate model
            if self.best_model_name == "Neural Network":
                print("Neural Network model loading not supported")
                return False
            else:
                self.best_model = joblib.load(MODEL_PATH)
            
            print(f"Models loaded successfully!")
            print(f"Best model: {self.best_model_name} with accuracy: {self.best_score:.4f}")
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_emotion(self, audio_features):
        """Predict emotion from audio features"""
        if self.best_model is None:
            print("No model loaded!")
            return None
        
        # Impute and scale features
        features_imputed = self.imputer.transform([audio_features])
        features_scaled = self.scaler.transform(features_imputed)
        
        # Predict
        if self.best_model_name == "Neural Network":
            predictions = self.best_model.predict(features_scaled)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions, axis=1)[0]
        else:
            predicted_class = self.best_model.predict(features_scaled)[0]
            probabilities = self.best_model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
        
        # Convert to emotion label
        emotion = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return emotion, confidence
    
    def train_complete_pipeline(self, data_dir):
        """Complete training pipeline"""
        print("Starting complete training pipeline...")
        
        # Load data
        data_parser = CremaDataParser()
        audio_files, emotions = data_parser.load_crema_data(data_dir)
        features = []
        
        print("Extracting features from audio files...")
        # Limit to first 500 files for faster testing
        limited_files = audio_files[:500]
        limited_emotions = emotions[:500]
        
        for i, audio_file in enumerate(limited_files):
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(limited_files)} files")
            
            feature_vector = self.audio_processor.process_audio_file(audio_file)
            if feature_vector is not None:
                features.append(feature_vector)
        
        # Update labels to match processed features
        labels = limited_emotions[:len(features)]
        print(f"Loaded {len(features)} samples with {len(np.unique(labels))} emotion classes")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(features, labels)
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train traditional models
        traditional_results = self.train_traditional_models(X_train, X_test, y_train, y_test)
        
        # Neural network training skipped (TensorFlow not available)
        print("Neural network training skipped - TensorFlow not available for Python 3.13")
        
        # Evaluate all models
        print("\nFinal Model Evaluation:")
        for name, result in self.models.items():
            if name != "Neural Network":
                accuracy = self.evaluate_model(result['model'], X_test, y_test, name)
            else:
                accuracy = self.evaluate_model(result['model'], X_test, y_test, name)
        
        # Save models
        self.save_models()
        
        return self.best_model, self.best_score
