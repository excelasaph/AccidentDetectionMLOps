from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from pydantic import BaseModel
from typing import List, Optional
from src.preprocessing import load_sequence_data, preprocess_for_prediction
from src.model import create_model, train_model, retrain_model
from src.database import db
import tensorflow as tf
import numpy as np
import shutil
import os
import time
import glob
import seaborn as sns
import matplotlib.pyplot as plt

router = APIRouter()

# Default data paths
TRAIN_DATA_PATH = "data/train"
TEST_DATA_PATH = "data/test"
VAL_DATA_PATH = "data/val"
UPLOADED_DATA_DIR = "data/uploaded"

def get_data_source():
    """Get sequence data from MongoDB if available, else from files."""
    if db.is_connected():
        train_seqs = db.load_from_collection("train")
        test_seqs = db.load_from_collection("test")
        val_seqs = db.load_from_collection("val")
        if train_seqs and test_seqs and val_seqs:
            return train_seqs, test_seqs, val_seqs
    train_seqs = db.load_sequences_from_files(TRAIN_DATA_PATH, max_frames=5)
    test_seqs = db.load_sequences_from_files(TEST_DATA_PATH, max_frames=5)
    val_seqs = db.load_sequences_from_files(VAL_DATA_PATH, max_frames=5)
    return train_seqs, test_seqs, val_seqs

# Prediction endpoint
class PredictionInput(BaseModel):
    image_paths: List[str]

@router.post("/predict")
async def predict(input: PredictionInput):
    try:
        if not all(os.path.exists(path) for path in input.image_paths):
            raise ValueError("Some image paths do not exist")
        processed_sequence = preprocess_for_prediction(input.image_paths, max_frames=5)
        model = tf.keras.models.load_model("models/accident_model.keras")
        prediction = model.predict(processed_sequence, verbose=0)[0][0]
        predicted_class = "Accident" if prediction > 0.5 else "Non-Accident"
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        return {"prediction": predicted_class, "confidence": float(confidence), "raw_score": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error predicting: {str(e)}")

# Default visualizations endpoint
@router.get("/visualizations/default")
async def get_default_visualizations():
    visualizations = []
    try:
        # Class Distribution
        train_seqs, _, _ = get_data_source()
        accident_count = sum(1 for seq in train_seqs if seq["label"] == 1)
        non_accident_count = sum(1 for seq in train_seqs if seq["label"] == 0)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=['Accident', 'Non_Accident'], y=[accident_count, non_accident_count], palette=['#1f77b4', '#ff7f0e'])
        plt.title('Class Distribution')
        plt.ylabel('Number of Sequences')
        plot_path = 'static/images/class_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations.append({
            "title": "Class Distribution",
            "plot_path": plot_path,
            "interpretation": "This bar plot shows the number of accident and non-accident sequences in the training set."
        })

        # Sample Images
        accident_seqs = [s for s in train_seqs if s["label"] == 1][:1]
        non_accident_seqs = [s for s in train_seqs if s["label"] == 0][:1]
        images = (accident_seqs[0]["image_paths"][:3] if accident_seqs else []) + (non_accident_seqs[0]["image_paths"][:3] if non_accident_seqs else [])
        if images:
            titles = ['Accident Frame 1', 'Accident Frame 2', 'Accident Frame 3', 
                      'Non Accident Frame 1', 'Non Accident Frame 2', 'Non Accident Frame 3']
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            for i, ax in enumerate(axes.flatten()):
                img = tf.keras.preprocessing.image.load_img(images[i])
                ax.imshow(img)
                ax.set_title(titles[i])
                ax.axis('off')
            plt.tight_layout()
            plot_path = 'static/images/sample_images.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append({
                "title": "Sample Sequences",
                "plot_path": plot_path,
                "interpretation": "This displays sample frames from accident and non-accident sequences."
            })

        return visualizations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating default visualizations: {str(e)}")

# Upload endpoint
@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload image sequence, save to MongoDB, and update dataset."""
    try:
        if not all(file.filename.endswith(('.jpg', '.jpeg', '.png')) for file in files):
            raise ValueError("Only image files (.jpg, .jpeg, .png) are allowed")
        if len(files) > 5:
            raise ValueError("Maximum 5 images per sequence")
        
        timestamp = int(time.time())
        sequence_id = f"uploaded_{timestamp}"
        uploaded_paths = []
        os.makedirs(UPLOADED_DATA_DIR, exist_ok=True)
        for file in files:
            unique_filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(UPLOADED_DATA_DIR, unique_filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            uploaded_paths.append(file_path)
        
        # Assume user specifies label (e.g., via query param or form; here, default to 0)
        sequence = {
            "sequence_id": sequence_id,
            "image_paths": uploaded_paths,
            "label": 0,  # Update based on UI input if needed
            "split": "train"  # Default to train; adjust based on UI
        }
        if db.is_connected():
            db.save_to_collection([sequence], "train")
        else:
            # Save to file system
            target_dir = os.path.join(TRAIN_DATA_PATH, "Non_Accident" if sequence["label"] == 0 else "Accident")
            os.makedirs(target_dir, exist_ok=True)
            for path in uploaded_paths:
                shutil.move(path, os.path.join(target_dir, os.path.basename(path)))
        
        return {"message": f"Uploaded sequence {sequence_id} with {len(uploaded_paths)} frames"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading files: {str(e)}")

# Retrain endpoint
@router.post("/retrain")
async def retrain():
    """Retrain the model using MongoDB or file data."""
    try:
        model_path = "models/accident_model.keras"
        backup_path = "models/accident_model_backup.keras"
        old_metrics = None

        if os.path.exists(model_path):
            shutil.copy(model_path, backup_path)
            train_seqs, test_seqs, val_seqs = get_data_source()
            _, test_steps = load_sequence_data(TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH)
            model = tf.keras.models.load_model(model_path)
            test_generator, test_steps = load_sequence_data(TEST_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH)[1]
            all_predictions = []
            all_labels = []
            for _ in range(test_steps):
                test_images, test_labels = next(test_generator)
                predictions = model.predict(test_images, verbose=0)
                predictions_binary = (predictions > 0.5).astype(int).flatten()
                all_predictions.extend(predictions_binary)
                all_labels.extend(test_labels)
            old_metrics = {
                "accuracy": float(accuracy_score(all_labels, all_predictions)),
                "precision": float(precision_score(all_labels, all_predictions, zero_division=0)),
                "recall": float(recall_score(all_labels, all_predictions, zero_division=0)),
                "f1": float(f1_score(all_labels, all_predictions, zero_division=0))
            }

        train_generator, train_steps = load_sequence_data(TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH)[0]
        val_generator, val_steps = load_sequence_data(TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH)[2]
        
        if os.path.exists(model_path):
            history = retrain_model(model_path, train_generator, val_generator, train_steps, val_steps, epochs=5)
        else:
            model = create_model(max_frames=5)
            history = train_model(model, train_generator, val_generator, train_steps, val_steps, epochs=50)

        test_generator, test_steps = load_sequence_data(TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH)[1]
        all_predictions = []
        all_labels = []
        for _ in range(test_steps):
            test_images, test_labels = next(test_generator)
            predictions = model.predict(test_images, verbose=0)
            predictions_binary = (predictions > 0.5).astype(int).flatten()
            all_predictions.extend(predictions_binary)
            all_labels.extend(test_labels)
        new_metrics = {
            "accuracy": float(accuracy_score(all_labels, all_predictions)),
            "precision": float(precision_score(all_labels, all_predictions, zero_division=0)),
            "recall": float(recall_score(all_labels, all_predictions, zero_division=0)),
            "f1": float(f1_score(all_labels, all_predictions, zero_division=0))
        }

        return {
            "message": "Model retrained successfully",
            "old_metrics": old_metrics,
            "new_metrics": new_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retraining: {str(e)}")

# Evaluate endpoint
@router.get("/evaluate")
async def evaluate():
    """Evaluate the current model on test data."""
    try:
        model_path = "models/accident_model.keras"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        model = tf.keras.models.load_model(model_path)
        _, test_generator, test_steps = load_sequence_data(TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH)
        all_predictions = []
        all_labels = []
        for _ in range(test_steps):
            test_images, test_labels = next(test_generator)
            predictions = model.predict(test_images, verbose=0)
            predictions_binary = (predictions > 0.5).astype(int).flatten()
            all_predictions.extend(predictions_binary)
            all_labels.extend(test_labels)
        metrics = {
            "accuracy": float(accuracy_score(all_labels, all_predictions)),
            "precision": float(precision_score(all_labels, all_predictions, zero_division=0)),
            "recall": float(recall_score(all_labels, all_predictions, zero_division=0)),
            "f1": float(f1_score(all_labels, all_predictions, zero_division=0))
        }
        cm = confusion_matrix(all_labels, all_predictions)
        cm_data = {
            "labels": ["Non-Accident", "Accident"],
            "matrix": cm.tolist()
        }
        return {"metrics": metrics, "confusion_matrix": cm_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error evaluating: {str(e)}")

# Save retrain endpoint
@router.post("/save_retrain")
async def save_retrain(save: bool = Body(..., embed=False)):
    try:
        backup_path = "models/accident_model_backup.keras"
        model_path = "models/accident_model.keras"
        if save:
            if os.path.exists(backup_path):
                os.remove(backup_path)
            return {"message": "New model saved"}
        else:
            if os.path.exists(backup_path):
                shutil.move(backup_path, model_path)
            return {"message": "Reverted to previous model"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving/reverting model: {str(e)}")