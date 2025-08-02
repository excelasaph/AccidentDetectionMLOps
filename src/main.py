from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from src.prediction import predict_image
from src.model import retrain_model
from src.preprocessing import load_sequence_data
import os
import shutil
from typing import List

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    os.makedirs("uploads", exist_ok=True)
    image_paths = []
    for file in files:
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        image_paths.append(file_path)
    prediction = predict_image("models/improved_accident_model.h5", image_paths)
    return {"prediction": prediction}

@app.post("/retrain")
async def retrain():
    (train_generator, train_steps), _, (val_generator, val_steps) = load_sequence_data("data/train", "data/test", "data/val")
    history = retrain_model("models/improved_accident_model.h5", train_generator, val_generator, train_steps, val_steps)
    return {"status": "Retraining completed", "history": history.history}

@app.post("/upload")
async def upload_data(files: List[UploadFile] = File(...), label: str = Form(...)):
    os.makedirs(f"data/train/{label}", exist_ok=True)
    for file in files:
        file_path = os.path.join(f"data/train/{label}", file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    return {"status": "Files uploaded successfully"}

@app.get("/health")
async def health():
    return {"status": "Model is up and running"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)