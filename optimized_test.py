"""
Optimized test script to validate speed and accuracy improvements
"""
import time
import numpy as np
from pathlib import Path

from audio_processor import AudioProcessor  
from model_trainer import EmotionModelTrainer
from config import *

def test_optimized_pipeline():
    """Test the optimized pipeline"""
    print("🚀 Testing Optimized Voice Emotion Recognition Pipeline")
    print("=" * 60)
    
    # Configuration info
    print(f"📋 Current Configuration:")
    print(f"  - Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  - Pitch Features: {'✅ Enabled' if ENABLE_PITCH_FEATURES else '❌ Disabled'}")
    print(f"  - Mel Features: {'✅ Enabled' if ENABLE_MEL_FEATURES else '❌ Disabled'}")
    print(f"  - Parallel Processing: {'✅ Enabled' if USE_PARALLEL_PROCESSING else '❌ Disabled'}")
    print(f"  - Caching: {'✅ Enabled' if CACHE_FEATURES else '❌ Disabled'}")
    print(f"  - Max Workers: {MAX_WORKERS}")
    print()
    
    # Initialize trainer
    trainer = EmotionModelTrainer()
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Run the complete optimized pipeline
    print("🎯 Running complete optimized pipeline...")
    start_time = time.time()
    
    try:
        best_model, best_score = trainer.train_complete_pipeline(data_dir)
        
        total_time = time.time() - start_time
        
        if best_model is not None:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            print(f"🏆 Best model accuracy: {best_score:.4f} ({best_score*100:.1f}%)")
            print(f"🤖 Best model: {trainer.best_model_name}")
            
            # Calculate performance metrics
            stats = trainer.audio_processor.get_processing_stats()
            if stats['total_processed'] > 0:
                success_rate = (stats['feature_extraction_success'] / stats['total_processed']) * 100
                avg_time_per_file = total_time / stats['total_processed']
                print(f"📊 Feature extraction success rate: {success_rate:.1f}%")
                print(f"⚡ Average time per file: {avg_time_per_file:.3f} seconds")
                
                # Check if we hit our targets
                print(f"\n🎯 Target Achievement:")
                if success_rate >= 90:
                    print(f"  ✅ Success rate target (90%+): {success_rate:.1f}%")
                else:
                    print(f"  ❌ Success rate target (90%+): {success_rate:.1f}%")
                
                if avg_time_per_file <= 0.2:
                    print(f"  ✅ Speed target (≤0.2s): {avg_time_per_file:.3f}s")
                else:
                    print(f"  ❌ Speed target (≤0.2s): {avg_time_per_file:.3f}s")
                
                if best_score >= 0.85:
                    print(f"  ✅ Accuracy target (≥85%): {best_score*100:.1f}%")
                else:
                    print(f"  ❌ Accuracy target (≥85%): {best_score*100:.1f}%")
            
            return True
        else:
            print("❌ Pipeline failed - no model trained")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_optimized_pipeline()
    
    if success:
        print("\n🎉 Optimization test completed successfully!")
        print("🚀 Ready for GitHub push!")
    else:
        print("\n❌ Optimization test failed.")
        print("🔧 Check the errors above and try again.")
