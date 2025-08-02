# Accident Detection MLOps Project

## Project Description

This project implements an **end-to-end Machine Learning pipeline** for detecting accidents from CCTV footage images using **TensorFlow**. It includes:

- Data preprocessing
- Model training and evaluation
- FastAPI backend
- HTML/CSS/JavaScript UI
- Deployment on Render
- Single-image predictions
- Bulk data uploads for retraining
- Visualizations of dataset features
- Load testing with Locust

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd AccidentDetectionMLOps
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset Setup

1. Download from **Kaggle**
2. Place images in the following structure:
   - `data/train/` with `Accident/` and `Non_Accident/` subfolders
   - `data/test/` with `Accident/` and `Non_Accident/` subfolders
   - `data/val/` with `Accident/` and `Non_Accident/` subfolders

### 4. Run Jupyter Notebook

```bash
jupyter notebook notebook/accident_detection.ipynb
```

### 5. Run FastAPI Server

```bash
uvicorn src.main:app --reload
```

### 6. Serve UI

- **Option 1:** Open `static/index.html` in a browser
- **Option 2:** Access via FastAPI at `http://localhost:8000`

### 7. Deploy on Render

Follow the deployment steps in the **Deploy on Render** section.

### 8. Run Load Testing

```bash
locust -f src/locustfile.py --host=https://<your-render-url>
```



## Demo & Resources

### 📺 Video Demo
[YouTube Link to Video Demo](Replace with your YouTube link)

### 🔗 Live Application
**API/UI:** [https://<your-render-url>](https://<your-render-url>)

## 🚀 Performance Testing Results

### Load Testing with Locust

| Configuration | Latency | Response Time | Users |
|---------------|---------|---------------|-------|
| **1 Instance** | ~[TBD]ms | ~[TBD]ms | 100 users |
| **2 Instances** | ~[TBD]ms | ~[TBD]ms | 100 users |

> **Note:** Update these metrics after running Locust performance tests.
