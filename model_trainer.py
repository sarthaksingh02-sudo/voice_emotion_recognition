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
# PyTorch for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
warnings.filterwarnings('ignore')

from config import *
from audio_processor import AudioProcessor
from data_parser import CremaDataParser
from ravdess_parser import RavdessDataParser

def extract_features_for_file(args):
    """Helper function for parallel feature extraction"""
    file_path, emotion, processor = args
    try:
        features = processor.process_audio_file(file_path)
        if features is not None:
            return features, emotion, file_path
        else:
            return None, None, file_path
    except Exception as e:
        return None, None, file_path

class EmotionMLP(nn.Module):
    """Enhanced PyTorch MLP model for high-accuracy emotion recognition"""
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(EmotionMLP, self).__init__()
        
        # Input layer with feature normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Deep architecture for better feature learning
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_rate * 0.5)
        
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(dropout_rate * 0.3)
        
        # Output layer
        self.fc_out = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Input normalization
        x = self.input_bn(x)
        
        # Deep layers with residual-like connections
        x1 = F.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout1(x1)
        
        x2 = F.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2)
        
        x3 = F.relu(self.bn3(self.fc3(x2)))
        x3 = self.dropout3(x3)
        
        x4 = F.relu(self.bn4(self.fc4(x3)))
        x4 = self.dropout4(x4)
        
        x5 = F.relu(self.bn5(self.fc5(x4)))
        x5 = self.dropout5(x5)
        
        # Output layer
        output = self.fc_out(x5)
        return output

class EmotionCNN(nn.Module):
    """1D CNN model for emotion recognition from audio features"""
    def __init__(self, input_dim, num_classes):
        super(EmotionCNN, self).__init__()
        
        # Reshape input for conv1d (batch, channels, length)
        self.input_dim = input_dim
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate the size after conv layers
        conv_output_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def _get_conv_output_size(self):
        """Calculate the output size of conv layers"""
        with torch.no_grad():
            x = torch.randn(1, 1, self.input_dim)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            return x.view(1, -1).size(1)
        
    def forward(self, x):
        # Reshape for conv1d: (batch, 1, features)
        x = x.unsqueeze(1)
        
        # CNN layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

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
            
            # Cross-validation (skip for speed)
            # cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            # print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Update best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                self.best_model_name = name
        
        self.models.update(results)
        return results
    
    def train_optimized_models(self, X_train, X_test, y_train, y_test):
        """Train optimized ML models for better accuracy targeting 85%"""
        # Define highly optimized models for 85% accuracy target
        models = {
            'Random Forest Enhanced': RandomForestClassifier(
                n_estimators=500,  # Increased for better accuracy
                max_depth=30,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                oob_score=True,
                class_weight='balanced'  # Handle class imbalance
            ),
            'Extra Trees Enhanced': RandomForestClassifier(
                n_estimators=500,
                max_depth=35,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                bootstrap=False,
                class_weight='balanced'
            ),
            'SVM Enhanced': SVC(
                kernel='rbf', 
                C=50.0,  # Increased regularization
                gamma='scale',
                random_state=42, 
                probability=True,
                class_weight='balanced',
                decision_function_shape='ovr'
            ),
            'Gradient Boosting Enhanced': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,  # Lower learning rate for better accuracy
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=10
            ),
            'MLP Enhanced': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64),  # Deeper network
                activation='relu',
                solver='adam',
                alpha=0.0001,  # Reduced regularization
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                learning_rate='adaptive'
            ),
            'Logistic Regression Enhanced': LogisticRegression(
                C=5.0,  # Increased regularization
                solver='lbfgs',
                random_state=42, 
                max_iter=3000,
                multi_class='multinomial',
                class_weight='balanced'
            )
        }
        
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
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
                
                # Update best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        self.models.update(results)
        return results
    
    def train_mlp_model(self, X_train, X_test, y_train, y_test):
        """Train PyTorch MLP model with enhanced training"""
        # Prepare data
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Validation split
        val_size = int(0.2 * len(X_train))
        train_size = len(X_train) - val_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        model = EmotionMLP(input_dim, num_classes)
        
        # Loss and optimizer with scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Training with validation
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # Training
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = correct / total
            epoch_loss = running_loss / len(train_subset)
            val_loss = val_loss / len(val_subset)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            scheduler.step()
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions = torch.argmax(outputs, 1)
            accuracy = (predictions == y_test_tensor).float().mean().item()
        
        print(f"Final MLP Accuracy: {accuracy:.4f}")
        
        # Update best model if this is better
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_model = model
            self.best_model_name = "MLP Neural Network"

        self.models["MLP Neural Network"] = {
            'model': model,
            'accuracy': accuracy
        }
        
        return model, accuracy
    
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
        if model_name == "MLP Neural Network":
            # PyTorch model
            model.eval()
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(X_test_tensor)
                y_pred = torch.argmax(outputs, 1).numpy()
        else:
            # Sklearn model
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
        
        # Save best model
        if self.best_model_name == "MLP Neural Network":
            torch.save(self.best_model.state_dict(), MODEL_DIR / 'mlp_model.pth')
            # Save model architecture info
            model_info = {
                'input_dim': self.best_model.fc1.in_features,
                'num_classes': self.best_model.fc4.out_features
            }
            with open(MODEL_DIR / 'mlp_info.pkl', 'wb') as f:
                pickle.dump(model_info, f)
        else:
            joblib.dump(self.models[self.best_model_name], MODEL_PATH)
        
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
            if self.best_model_name == "MLP Neural Network":
                # Load PyTorch model
                with open(MODEL_DIR / 'mlp_info.pkl', 'rb') as f:
                    model_info = pickle.load(f)
                
                # Recreate model architecture
                model = EmotionMLP(model_info['input_dim'], model_info['num_classes'])
                model.load_state_dict(torch.load(MODEL_DIR / 'mlp_model.pth'))
                model.eval()
                self.best_model = model
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
        if self.best_model_name == "MLP Neural Network":
            # PyTorch model prediction
            self.best_model.eval()
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            with torch.no_grad():
                outputs = self.best_model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
        else:
            predicted_class = self.best_model.predict(features_scaled)[0]
            probabilities = self.best_model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
        
        # Convert to emotion label
        emotion = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return emotion, confidence
    
    def train_complete_pipeline(self, data_dir):
        """Complete training pipeline with parallel processing using RAVDESS dataset only"""
        print("🚀 Starting optimized training pipeline with RAVDESS dataset only...")
        start_time = time.time()
        
        # Load RAVDESS data only (no CREMA-D)
        data_parser = RavdessDataParser()
        audio_files, emotions = data_parser.load_ravdess_data(data_dir)
        
        print(f"📁 Loaded {len(audio_files)} audio files")
        print(f"🎯 Target: Extract features from all {len(audio_files)} files")
        
        # Extract features with parallel processing
        features, valid_labels = self._extract_features_parallel(audio_files, emotions)
        
        # Get processing statistics
        stats = self.audio_processor.get_processing_stats()
        print(f"\n📊 Feature Extraction Results:")
        print(f"- Total processed: {stats['total_processed']}")
        print(f"- Cache hits: {stats['cache_hits']}")
        print(f"- Successfully extracted: {stats['feature_extraction_success']}")
        print(f"- Failed extraction: {stats['feature_extraction_failed']}")
        print(f"- Load failures: {stats['load_failed']}")
        print(f"- Preprocess failures: {stats['preprocess_failed']}")
        
        success_rate = (stats['feature_extraction_success'] / stats['total_processed']) * 100 if stats['total_processed'] > 0 else 0
        print(f"- Success rate: {success_rate:.1f}%")
        
        if len(features) == 0:
            print("❌ No valid features extracted. Exiting.")
            return None, 0.0
        
        print(f"✅ Loaded {len(features)} samples with {len(np.unique(valid_labels))} emotion classes")
        
        # Prepare data with proper train/test split
        X_train, X_test, y_train, y_test = self.prepare_data(features, valid_labels)
        print(f"🎯 Training set: {len(X_train)} samples")
        print(f"🎯 Test set: {len(X_test)} samples")
        
        # Train optimized models
        print("\n🤖 Training optimized models...")
        traditional_results = self.train_optimized_models(X_train, X_test, y_train, y_test)
        
        # Train MLP model
        print("\n🧠 Training PyTorch MLP model...")
        mlp_model, _ = self.train_mlp_model(X_train, X_test, y_train, y_test)
        
        # Evaluate all models
        print("\n📈 Final Model Evaluation:")
        for name, result in self.models.items():
            accuracy = self.evaluate_model(result['model'], X_test, y_test, name)
        
        # Save models
        self.save_models()
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total pipeline time: {total_time:.2f} seconds")
        print(f"🏆 Best model: {self.best_model_name} with accuracy: {self.best_score:.4f}")
        
        return self.best_model, self.best_score
    
    def _extract_features_parallel(self, audio_files, emotions):
        """Extract features using parallel processing"""
        print("\n🔄 Starting parallel feature extraction...")
        start_time = time.time()
        
        features = []
        valid_labels = []
        
        if USE_PARALLEL_PROCESSING and len(audio_files) > 100:
            print(f"⚡ Using {MAX_WORKERS} parallel workers")
            
            # Prepare arguments for parallel processing
            args_list = [(file_path, emotion, self.audio_processor) 
                        for file_path, emotion in zip(audio_files, emotions)]
            
            # Use ThreadPoolExecutor for I/O bound tasks
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_args = {executor.submit(extract_features_for_file, args): args 
                                 for args in args_list}
                
                # Process completed tasks
                for i, future in enumerate(as_completed(future_to_args)):
                    try:
                        feature_vector, emotion, file_path = future.result()
                        if feature_vector is not None:
                            features.append(feature_vector)
                            valid_labels.append(emotion)
                    except Exception as e:
                        print(f"Error processing file: {e}")
                    
                    # Progress update every 500 files
                    if (i + 1) % 500 == 0:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / (i + 1)
                        remaining = (len(args_list) - i - 1) * avg_time
                        print(f"📈 Processed {i + 1}/{len(args_list)} files | "
                              f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
        else:
            print("🔄 Using sequential processing")
            for i, (audio_file, emotion) in enumerate(zip(audio_files, emotions)):
                if (i + 1) % 200 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = (len(audio_files) - i - 1) * avg_time
                    print(f"📈 Processed {i + 1}/{len(audio_files)} files | "
                          f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
                
                feature_vector = self.audio_processor.process_audio_file(audio_file)
                if feature_vector is not None:
                    features.append(feature_vector)
                    valid_labels.append(emotion)
        
        extraction_time = time.time() - start_time
        avg_time_per_file = extraction_time / len(audio_files) if len(audio_files) > 0 else 0
        
        print(f"\n⏱️ Feature extraction completed in {extraction_time:.2f} seconds")
        print(f"📊 Average time per file: {avg_time_per_file:.3f} seconds")
        
        return features, valid_labels
