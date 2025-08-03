# Force CPU-only mode for deployment
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from src.preprocessing import preprocess_for_prediction

# Ensure CPU-only execution
tf.config.set_visible_devices([], 'GPU')

def predict_image(model_path, image_paths, max_frames=5):
    model = tf.keras.models.load_model(model_path)
    seq_array = preprocess_for_prediction(image_paths, max_frames=max_frames)
    prediction = model.predict(seq_array)[0][0]
    return float(prediction)  # Return the numerical prediction value