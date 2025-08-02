import tensorflow as tf
from src.preprocessing import preprocess_for_prediction
import os

def predict_image(model_path, image_paths, max_frames=5):
    model = tf.keras.models.load_model(model_path)
    seq_array = preprocess_for_prediction(image_paths, max_frames=max_frames)
    prediction = model.predict(seq_array)[0][0]
    return 'Accident' if prediction > 0.5 else 'Non_Accident'