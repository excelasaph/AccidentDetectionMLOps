# Force CPU-only mode for deployment
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from src.preprocessing import preprocess_for_prediction

# Ensure CPU-only execution
tf.config.set_visible_devices([], 'GPU')

def predict_image(model_path, image_paths, max_frames=5):
    try:
        # Load model with compatibility options
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=None, 
            compile=False,
            safe_mode=False
        )
        
        # Recompile the model to ensure compatibility
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Process input images
        seq_array = preprocess_for_prediction(image_paths, max_frames=max_frames)
        
        # Make prediction
        prediction = model.predict(seq_array)[0][0]
        
        return float(prediction)  # Return the numerical prediction value
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise Exception(f"Model prediction failed: {str(e)}")