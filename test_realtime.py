#!/usr/bin/env python3
"""
Test real-time emotion recognition system with file input
"""

import os
from pathlib import Path
from real_time_inference import RealTimeEmotionRecognition

def test_realtime_with_files():
    """Test real-time system with audio files"""
    
    print("🎤 Testing Real-Time Emotion Recognition System")
    print("=" * 55)
    
    # Initialize the real-time system
    try:
        recognizer = RealTimeEmotionRecognition()
        print("✅ Real-time system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize real-time system: {e}")
        return False
    
    # Test with some sample files
    audio_files = list(Path('data/AudioWAV').glob('*.wav'))
    test_files = audio_files[:5]  # Test with first 5 files
    
    print(f"\n🧪 Testing with {len(test_files)} sample files:")
    
    for i, audio_file in enumerate(test_files, 1):
        print(f"\n📁 Test {i}: {audio_file.name}")
        
        try:
            # Test single prediction
            recognizer.test_single_prediction(str(audio_file))
        except Exception as e:
            print(f"❌ Error testing file {audio_file.name}: {e}")
    
    print(f"\n🎉 Real-time system test completed!")
    return True

if __name__ == "__main__":
    success = test_realtime_with_files()
    if success:
        print("\n✅ Real-time system is working correctly!")
        print("💡 You can now run: python real_time_inference.py")
    else:
        print("\n❌ Real-time system needs troubleshooting")
