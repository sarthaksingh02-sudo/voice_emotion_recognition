"""
Test script for Emotion Recognition API v1.1
Tests all endpoints and functionality
"""

import requests
import json
from pathlib import Path
import glob

API_BASE_URL = "http://127.0.0.1:8000"

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")

def test_predict_endpoint():
    """Test the predict endpoint with sample audio"""
    print("\n🔍 Testing predict endpoint...")
    
    # Look for sample audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
        audio_files.extend(glob.glob(f"recordings/{ext}"))
    
    if not audio_files:
        print("❌ No audio files found in recordings/ directory")
        return
    
    test_file = audio_files[0]
    print(f"📁 Using test file: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE_URL}/predict", files=files)
            
        if response.status_code == 200:
            print("✅ Predict endpoint working!")
            result = response.json()
            print(f"📊 Prediction Result:")
            print(f"   Emotion: {result.get('emotion')}")
            print(f"   Confidence: {result.get('confidence'):.2f}")
            print(f"   Model: {result.get('model')}")
            print(f"   Session ID: {result.get('session_id')}")
        else:
            print(f"❌ Predict endpoint failed with status {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Predict endpoint error: {e}")

def test_logs_endpoint():
    """Test the logs endpoint"""
    print("\n🔍 Testing logs endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/logs?limit=5")
        if response.status_code == 200:
            print("✅ Logs endpoint working!")
            result = response.json()
            print(f"📋 Retrieved {result.get('count', 0)} logs")
            for log in result.get('logs', [])[:3]:  # Show first 3
                print(f"   {log.get('timestamp')}: {log.get('emotion')} ({log.get('confidence'):.2f})")
        else:
            print(f"❌ Logs endpoint failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Logs endpoint error: {e}")

def main():
    print("🧪 Testing Emotion Recognition API v1.1")
    print("=" * 50)
    
    # Test all endpoints
    test_health_endpoint()
    test_predict_endpoint()
    test_logs_endpoint()
    
    print("\n" + "=" * 50)
    print("🏁 Testing complete!")

if __name__ == "__main__":
    main()
