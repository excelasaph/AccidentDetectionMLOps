# Force CPU-only mode for deployment
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from src.prediction import predict_image
from src.model import retrain_model
from src.preprocessing import load_sequence_data, preprocess_for_prediction, load_sequence_data_with_mongodb
from src.database import db
import shutil
import glob
from typing import List

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    os.makedirs("uploads", exist_ok=True)
    image_paths = []
    image_file_ids = []
    
    # Store images to disk for prediction
    for file in files:
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        image_paths.append(file_path)
    
    # Make prediction
    prediction = predict_image("models/accident_model.keras", image_paths)
    
    # Store prediction images in MongoDB GridFS if connected
    if db.is_connected():
        try:
            from datetime import datetime
            timestamp = datetime.utcnow().isoformat()
            
            # Store each image in GridFS with prediction metadata
            for i, img_path in enumerate(image_paths):
                metadata = {
                    'purpose': 'prediction',
                    'timestamp': timestamp,
                    'frame_number': i + 1,
                    'total_frames': len(image_paths),
                    'prediction_result': float(prediction),
                    'accident_detected': float(prediction) >= 0.5
                }
                
                file_id = db.store_image_in_gridfs(img_path, metadata)
                if file_id:
                    image_file_ids.append(file_id)
            
            # Store prediction record
            if image_file_ids:
                confidence = float(prediction) if float(prediction) >= 0.5 else (1.0 - float(prediction))
                db.store_prediction_record(image_file_ids, float(prediction), confidence, timestamp)
            
        except Exception as e:
            print(f"Error storing prediction in MongoDB: {e}")
    
    return {
        "prediction": prediction,
        "stored_in_mongodb": len(image_file_ids) > 0,
        "images_stored": len(image_file_ids)
    }

@app.post("/retrain")
async def retrain():
    # Use MongoDB-aware data loading that combines file-based and uploaded data
    (train_generator, train_steps), _, (val_generator, val_steps) = load_sequence_data_with_mongodb(
        train_dir="data/train", 
        test_dir="data/test", 
        val_dir="data/val",
        use_mongodb=True
    )
    history = retrain_model("models/accident_model.keras", train_generator, val_generator, train_steps, val_steps)
    return {"status": "Retraining completed", "history": history.history}

@app.get("/training-data-sources")
async def get_training_data_sources():
    """Get information about available training data sources."""
    sources = {
        "file_based": {
            "train": len(glob.glob("data/train/*/*")) if os.path.exists("data/train") else 0,
            "test": len(glob.glob("data/test/*/*")) if os.path.exists("data/test") else 0,
            "val": len(glob.glob("data/val/*/*")) if os.path.exists("data/val") else 0
        },
        "mongodb": {
            "connected": db.is_connected(),
            "train": 0,
            "test": 0, 
            "val": 0
        }
    }
    
    # Get MongoDB counts if connected
    if db.is_connected():
        try:
            sources["mongodb"]["train"] = len(db.load_sequences_with_images("train"))
            sources["mongodb"]["test"] = len(db.load_sequences_with_images("test"))
            sources["mongodb"]["val"] = len(db.load_sequences_with_images("val"))
            sources["mongodb"]["gridfs_stats"] = db.get_gridfs_stats()
        except Exception as e:
            sources["mongodb"]["error"] = str(e)
    
    return sources

@app.get("/prediction-history")
async def get_prediction_history(limit: int = 50):
    """Get recent prediction history."""
    history = db.get_prediction_history(limit)
    return {
        "prediction_history": history,
        "total_records": len(history)
    }

@app.get("/prediction-stats")
async def get_prediction_statistics():
    """Get prediction statistics and analytics."""
    stats = db.get_prediction_stats()
    return stats

@app.post("/add-prediction-to-training")
async def add_prediction_to_training(prediction_id: str = Form(...), label: str = Form(...)):
    """Add a prediction's images to training data with a specified label."""
    if not db.is_connected():
        return {"error": "Database not connected", "success": False}
    
    try:
        # Get the prediction record
        collection = db.db['predictions']
        prediction_record = collection.find_one({'prediction_id': prediction_id}, {"_id": 0})
        
        if not prediction_record:
            return {"error": "Prediction not found", "success": False}
        
        # Create training sequence from prediction images
        sequence_data = {
            'sequence_id': f"pred_to_train_{prediction_id[:8]}",
            'image_file_ids': prediction_record['image_file_ids'],
            'label': 1 if label.lower() == 'accident' else 0,
            'split': 'train',
            'source': 'prediction_conversion',
            'original_prediction': prediction_record['prediction_result'],
            'storage_type': 'gridfs'
        }
        
        # Save to training collection
        train_collection = db.db['train']
        train_collection.insert_one(sequence_data)
        
        return {
            "success": True,
            "message": f"Added prediction {prediction_id[:8]} to training data as '{label}'",
            "images_added": len(prediction_record['image_file_ids'])
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/upload")
async def upload_data(files: List[UploadFile] = File(...), label: str = Form(...)):
    os.makedirs(f"data/train/{label}", exist_ok=True)
    
    # Save files to disk
    file_paths = []
    for file in files:
        file_path = os.path.join(f"data/train/{label}", file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(file_path)
    
    # Optionally store in MongoDB with GridFS
    if db.is_connected():
        try:
            # Create a sequence for uploaded files
            sequence_data = {
                'sequence_id': f"uploaded_{len(file_paths)}",
                'image_paths': file_paths,
                'label': 1 if label.lower() == 'accident' else 0,
                'split': 'train'
            }
            
            # Store with images in GridFS
            db.save_sequences_with_images([sequence_data], "train", store_images=True)
            
            return {
                "status": "Files uploaded successfully", 
                "saved_to_disk": True,
                "saved_to_mongodb": True,
                "files_count": len(file_paths)
            }
        except Exception as e:
            return {
                "status": "Files uploaded to disk only", 
                "saved_to_disk": True,
                "saved_to_mongodb": False,
                "error": str(e),
                "files_count": len(file_paths)
            }
    
    return {
        "status": "Files uploaded successfully",
        "saved_to_disk": True,
        "saved_to_mongodb": False,
        "files_count": len(file_paths)
    }

@app.get("/storage-stats")
async def get_storage_stats():
    """Get MongoDB GridFS storage statistics."""
    if not db.is_connected():
        return {"error": "MongoDB not connected"}
    
    return db.get_gridfs_stats()

@app.delete("/clear-predictions")
async def clear_predictions():
    """Clear all prediction data from database and GridFS."""
    result = db.clear_prediction_data()
    return result

@app.delete("/clear-training-data")
async def clear_training_data():
    """Clear all uploaded training data from database and GridFS."""
    result = db.clear_training_data()
    return result

@app.get("/health")
async def health():
    return {
        "status": "Model is up and running",
        "mongodb_connected": db.is_connected(),
        "gridfs_available": hasattr(db, 'fs') and db.fs is not None
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)