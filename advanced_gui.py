"""
Advanced GUI for Emotion Recognition System v1.1
Professional interface with real-time visualization, multiple models, and comprehensive features
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit, 
                             QComboBox, QGroupBox, QGridLayout, QTabWidget, QFrame,
                             QSplitter, QSlider, QSpinBox, QCheckBox, QListWidget,
                             QTableWidget, QTableWidgetItem, QMessageBox, QStatusBar)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QSize
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPalette, QColor
import pyqtgraph as pg
from pyqtgraph import PlotWidget
import librosa
import joblib
from pathlib import Path
import requests
import json
import threading
import time
from datetime import datetime

class EmotionRecognitionAdvancedGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎭 Advanced Emotion Recognition System v1.1")
        self.setGeometry(100, 100, 1200, 800)
        self.currentFile = None
        self.api_base_url = "http://127.0.0.1:8000"
        self.setupUI()
        self.setupStyle()
        self.checkAPIConnection()

    def setupUI(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel (Controls)
        left_panel = self.createLeftPanel()
        
        # Right panel (Visualization and Results)
        right_panel = self.createRightPanel()
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready - Load an audio file to begin")

    def createLeftPanel(self):
        # Left panel container
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        # Title
        title_label = QLabel("🎭 Emotion Recognition")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)
        
        # File Operations Group
        file_group = QGroupBox("📁 File Operations")
        file_layout = QVBoxLayout()
        
        self.load_button = QPushButton("🔊 Load Audio File")
        self.load_button.clicked.connect(self.loadAudio)
        file_layout.addWidget(self.load_button)
        
        self.file_label = QLabel("Selected File: None")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # Prediction Group
        prediction_group = QGroupBox("🔮 Prediction")
        prediction_layout = QVBoxLayout()
        
        self.predict_button = QPushButton("🎯 Predict Emotion")
        self.predict_button.clicked.connect(self.predictEmotion)
        self.predict_button.setEnabled(False)
        prediction_layout.addWidget(self.predict_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        prediction_layout.addWidget(self.progress_bar)
        
        prediction_group.setLayout(prediction_layout)
        left_layout.addWidget(prediction_group)
        
        # Results Group
        results_group = QGroupBox("📊 Results")
        results_layout = QVBoxLayout()
        
        self.result_label = QLabel("Prediction: N/A")
        self.result_label.setFont(QFont("Arial", 12, QFont.Bold))
        results_layout.addWidget(self.result_label)
        
        self.confidence_label = QLabel("Confidence: N/A")
        results_layout.addWidget(self.confidence_label)
        
        self.model_label = QLabel("Model: N/A")
        results_layout.addWidget(self.model_label)
        
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)
        
        # API Status Group
        api_group = QGroupBox("🌐 API Status")
        api_layout = QVBoxLayout()
        
        self.api_status_label = QLabel("Status: Checking...")
        api_layout.addWidget(self.api_status_label)
        
        self.refresh_api_button = QPushButton("🔄 Refresh API Status")
        self.refresh_api_button.clicked.connect(self.checkAPIConnection)
        api_layout.addWidget(self.refresh_api_button)
        
        api_group.setLayout(api_layout)
        left_layout.addWidget(api_group)
        
        # Recent Predictions Group
        recent_group = QGroupBox("📋 Recent Predictions")
        recent_layout = QVBoxLayout()
        
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(150)
        recent_layout.addWidget(self.recent_list)
        
        self.refresh_logs_button = QPushButton("📊 Load Recent Logs")
        self.refresh_logs_button.clicked.connect(self.loadRecentLogs)
        recent_layout.addWidget(self.refresh_logs_button)
        
        recent_group.setLayout(recent_layout)
        left_layout.addWidget(recent_group)
        
        # Stretch to push everything to top
        left_layout.addStretch()
        
        return left_widget

    def createRightPanel(self):
        # Right panel with tabs
        right_widget = QTabWidget()
        
        # Waveform tab
        waveform_tab = QWidget()
        waveform_layout = QVBoxLayout()
        
        # Waveform plot
        self.plot_widget = PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setTitle('Audio Waveform')
        waveform_layout.addWidget(self.plot_widget)
        
        waveform_tab.setLayout(waveform_layout)
        right_widget.addTab(waveform_tab, "🔊 Waveform")
        
        # Spectrogram tab (placeholder for future)
        spectrogram_tab = QWidget()
        spectrogram_layout = QVBoxLayout()
        spectrogram_label = QLabel("Spectrogram visualization will be added here")
        spectrogram_label.setAlignment(Qt.AlignCenter)
        spectrogram_layout.addWidget(spectrogram_label)
        spectrogram_tab.setLayout(spectrogram_layout)
        right_widget.addTab(spectrogram_tab, "📊 Spectrogram")
        
        # Statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Property", "Value"])
        stats_layout.addWidget(self.stats_table)
        
        stats_tab.setLayout(stats_layout)
        right_widget.addTab(stats_tab, "📈 Statistics")
        
        return right_widget

    def setupStyle(self):
        # Set a modern style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLabel {
                color: #333333;
            }
            QListWidget {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
            }
        """)

    def checkAPIConnection(self):
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=3)
            if response.status_code == 200:
                self.api_status_label.setText("Status: ✅ Connected")
                self.api_status_label.setStyleSheet("color: green;")
                self.predict_button.setEnabled(self.currentFile is not None)
            else:
                self.api_status_label.setText("Status: ❌ API Error")
                self.api_status_label.setStyleSheet("color: red;")
                self.predict_button.setEnabled(False)
        except:
            self.api_status_label.setText("Status: ❌ Disconnected")
            self.api_status_label.setStyleSheet("color: red;")
            self.predict_button.setEnabled(False)

    def loadAudio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", 
            "Audio Files (*.wav *.mp3 *.m4a *.flac);;All Files (*)"
        )
        
        if file_path:
            self.currentFile = file_path
            self.file_label.setText(f"Selected File: {Path(file_path).name}")
            self.displayWaveform(file_path)
            self.displayAudioStats(file_path)
            self.predict_button.setEnabled(True)
            self.statusBar.showMessage(f"Loaded: {Path(file_path).name}")

    def displayWaveform(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=None)
            time = np.linspace(0, len(y) / sr, num=len(y))
            
            # Clear previous plot
            self.plot_widget.clear()
            
            # Plot waveform
            self.plot_widget.plot(time, y, pen='b', name='Waveform')
            self.plot_widget.setTitle(f'Audio Waveform - {Path(file_path).name}')
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load waveform: {str(e)}")

    def displayAudioStats(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=None)
            
            # Calculate statistics
            duration = len(y) / sr
            max_amplitude = np.max(np.abs(y))
            rms_energy = np.sqrt(np.mean(y**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Populate stats table
            stats = [
                ("Duration", f"{duration:.2f} seconds"),
                ("Sample Rate", f"{sr} Hz"),
                ("Samples", f"{len(y):,}"),
                ("Max Amplitude", f"{max_amplitude:.4f}"),
                ("RMS Energy", f"{rms_energy:.4f}"),
                ("Zero Crossing Rate", f"{zero_crossing_rate:.4f}")
            ]
            
            self.stats_table.setRowCount(len(stats))
            for i, (prop, value) in enumerate(stats):
                self.stats_table.setItem(i, 0, QTableWidgetItem(prop))
                self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))
            
            self.stats_table.resizeColumnsToContents()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to calculate statistics: {str(e)}")

    def predictEmotion(self):
        if not self.currentFile:
            QMessageBox.warning(self, "Warning", "Please select an audio file first.")
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.predict_button.setEnabled(False)
        self.statusBar.showMessage("Predicting emotion...")
        
        # Start prediction in a separate thread
        threading.Thread(target=self._predict_worker, daemon=True).start()

    def _predict_worker(self):
        try:
            with open(self.currentFile, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.api_base_url}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                emotion = result.get('emotion', 'Unknown')
                confidence = result.get('confidence', 0)
                model = result.get('model', 'Unknown')
                
                # Update UI in main thread
                self.updatePredictionResult(emotion, confidence, model)
            else:
                self.updatePredictionResult("Error", 0, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.updatePredictionResult("Error", 0, str(e))

    def updatePredictionResult(self, emotion, confidence, model):
        # This method should be called from the main thread
        self.result_label.setText(f"Prediction: {emotion}")
        self.confidence_label.setText(f"Confidence: {confidence*100:.1f}%")
        self.model_label.setText(f"Model: {model}")
        
        # Add to recent predictions
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.recent_list.addItem(f"[{timestamp}] {emotion} ({confidence*100:.1f}%)")
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.predict_button.setEnabled(True)
        self.statusBar.showMessage(f"Prediction complete: {emotion}")

    def loadRecentLogs(self):
        try:
            response = requests.get(f"{self.api_base_url}/logs?limit=10")
            if response.status_code == 200:
                logs = response.json().get('logs', [])
                self.recent_list.clear()
                for log in logs:
                    timestamp = log.get('timestamp', '')
                    emotion = log.get('emotion', '')
                    confidence = log.get('confidence', 0)
                    item_text = f"[{timestamp}] {emotion} ({confidence*100:.1f}%)"
                    self.recent_list.addItem(item_text)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load logs: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Advanced Emotion Recognition System")
    app.setApplicationVersion("1.1")
    
    # Create and show main window
    window = EmotionRecognitionAdvancedGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
