"""
Minimal FastAPI implementation for assignment requirements.
Focuses on core functionality without excessive data storage.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from src.prediction import predict_image
from src.model import retrain_model
from src.preprocessing import load_sequence_data
from src.database_minimal import minimal_db
import os
import shutil
from typing import List

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    """Make prediction with minimal database storage."""
    os.makedirs("uploads", exist_ok=True)
    image_paths = []
    
    # Store images temporarily for prediction only
    for file in files:
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        image_paths.append(file_path)
    
    # Make prediction
    prediction = predict_image("models/accident_model.keras", image_paths)
    
    # Store ONLY essential prediction data (not images)
    if minimal_db.is_connected():
        minimal_db.store_prediction_result(float(prediction), len(image_paths))
    
    # Clean up temporary files
    for path in image_paths:
        try:
            os.remove(path)
        except:
            pass
    
    return {
        "prediction": prediction,
        "message": "Prediction completed"
    }

@app.post("/retrain")
async def retrain():
    """Retrain model with minimal logging."""
    # Use file-based data loading only
    (train_generator, train_steps), _, (val_generator, val_steps) = load_sequence_data(
        train_dir="data/train", 
        test_dir="data/test", 
        val_dir="data/val"
    )
    
    history = retrain_model("models/accident_model.keras", train_generator, val_generator, train_steps, val_steps)
    
    # Store only essential training results
    if minimal_db.is_connected() and history:
        final_accuracy = history.history.get('val_accuracy', [0])[-1]
        final_loss = history.history.get('val_loss', [1])[-1]
        epochs = len(history.history.get('loss', []))
        
        minimal_db.store_training_session(
            model_version="v1.0",
            accuracy=float(final_accuracy),
            loss=float(final_loss),
            epochs=epochs
        )
    
    return {"status": "Retraining completed"}

@app.post("/upload")
async def upload_data(files: List[UploadFile] = File(...), label: str = Form(...)):
    """Upload training data to file system only."""
    os.makedirs(f"data/train/{label}", exist_ok=True)
    
    uploaded_count = 0
    for file in files:
        file_path = os.path.join(f"data/train/{label}", file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        uploaded_count += 1
    
    return {
        "status": "Files uploaded successfully",
        "files_count": uploaded_count,
        "location": f"data/train/{label}"
    }

@app.get("/stats")
async def get_stats():
    """Get basic system statistics."""
    stats = {
        "model_status": "Ready" if os.path.exists("models/accident_model.keras") else "Missing",
        "database_connected": minimal_db.is_connected()
    }
    
    if minimal_db.is_connected():
        prediction_stats = minimal_db.get_prediction_stats()
        training_history = minimal_db.get_training_history(limit=5)
        stats.update({
            "prediction_stats": prediction_stats,
            "recent_training": training_history
        })
    
    return stats

@app.get("/health")
async def health():
    """Simple health check."""
    return {
        "status": "healthy",
        "database_connected": minimal_db.is_connected()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
