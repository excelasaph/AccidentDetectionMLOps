"""
Quick script to regenerate the model with current TensorFlow version
Run this if the version mismatch persists
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from src.model import create_model
import numpy as np

def regenerate_compatible_model():
    """Create and save a model compatible with current TensorFlow version"""
    
    print("Creating new model with current TensorFlow version...")
    model = create_model(max_frames=5)
    
    # Create dummy data to initialize the model
    dummy_input = np.random.random((1, 5, 224, 224, 3))
    
    # Initialize model by calling it
    _ = model(dummy_input)
    
    # Save the model
    model.save("models/accident_model_compatible.keras")
    print("✅ Compatible model saved as 'models/accident_model_compatible.keras'")
    
    # Also save as the main model (backup the original first)
    if os.path.exists("models/accident_model.keras"):
        os.rename("models/accident_model.keras", "models/accident_model_backup.keras")
        print("📦 Original model backed up as 'accident_model_backup.keras'")
    
    model.save("models/accident_model.keras")
    print("✅ New model saved as 'models/accident_model.keras'")
    
    print(f"📊 Model summary:")
    print(f"   - TensorFlow version: {tf.__version__}")
    print(f"   - Input shape: (None, 5, 224, 224, 3)")
    print(f"   - Output shape: (None, 1)")
    print(f"   - Total parameters: {model.count_params():,}")

if __name__ == "__main__":
    regenerate_compatible_model()
