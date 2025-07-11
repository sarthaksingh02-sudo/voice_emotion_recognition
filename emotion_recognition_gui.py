"""
Basic GUI setup for Emotion Recognition System v1.1
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QProgressBar, QTextEdit, QComboBox,
                             QGroupBox, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon
import pyqtgraph as pg
from pyqtgraph import PlotWidget
import librosa
import joblib
from pathlib import Path
from config import EMOTION_LABELS, MODEL_DIR
from audio_processor import AudioProcessor
import requests
import json

class EmotionRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Recognition System v1.1")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        # Main widget
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        # Layout
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        # Load audio button
        self.loadButton = QPushButton("Load Audio File")
        self.loadButton.clicked.connect(self.loadAudio)
        self.layout.addWidget(self.loadButton)

        # Label to display selected file
        self.fileLabel = QLabel("Selected File: None")
        self.layout.addWidget(self.fileLabel)

        # Predict button
        self.predictButton = QPushButton("Predict Emotion")
        self.predictButton.clicked.connect(self.predictEmotion)
        self.layout.addWidget(self.predictButton)

        # Prediction result
        self.resultLabel = QLabel("Prediction Result: N/A")
        self.layout.addWidget(self.resultLabel)

        # Waveform plot
        self.plotWidget = PlotWidget()
        self.layout.addWidget(self.plotWidget)
        self.plotItem = self.plotWidget.getPlotItem()
        self.plotItem.setTitle("Audio Waveform")
        self.plotItem.setLabel("left", "Amplitude")
        self.plotItem.setLabel("bottom", "Time", units="s")

    def loadAudio(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Audio File")
        if filePath:
            self.fileLabel.setText(f"Selected File: {filePath}")
            self.displayWaveform(filePath)

    def predictEmotion(self):
        # Predict emotion using the API
        try:
            currentFile = self.fileLabel.text().replace('Selected File: ', '')
            if currentFile == 'None':
                self.resultLabel.setText("No file selected!")
                return
            
            with open(currentFile, 'rb') as f:
                files = {'file': f}
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                
            if response.status_code == 200:
                result = response.json()
                emotion = result.get('emotion')
                confidence = result.get('confidence')
                self.resultLabel.setText(f"Prediction: {emotion} ({confidence*100:.2f}%)")
            else:
                self.resultLabel.setText(f"Prediction Error: {response.status_code}")
        except Exception as e:
            self.resultLabel.setText(f"Error: {e}")

    def displayWaveform(self, filePath):
        try:
            y, sr = librosa.load(filePath, sr=None)
            time = np.linspace(0, len(y) / sr, num=len(y))
            self.plotItem.clear()
            self.plotItem.plot(time, y, pen="b")
        except Exception as e:
            self.fileLabel.setText(f"Error displaying waveform: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionRecognitionGUI()
    window.show()
    sys.exit(app.exec_())

