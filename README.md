# Accident Detection MLOps Pipeline

A machine learning pipeline for real-time accident detection from CCTV footage using deep learning, with complete MLOps infrastructure including FastAPI backend, MongoDB storage, Streamlit dashboard, and automated deployment.

## Project Overview

This project implements an end-to-end MLOps pipeline for accident detection that analyzes sequential images from CCTV footage to predict accident occurrences. The system features a CNN+LSTM architecture, comprehensive data management, and production-ready deployment infrastructure.

### Key Features

- **Deep Learning Pipeline**: CNN+LSTM model with MobileNetV2 backbone for temporal sequence analysis
- **Real-time API**: FastAPI with 11+ endpoints for predictions, training, and data management  
- **Data Management**: MongoDB with GridFS for image storage and metadata tracking
- **Interactive Dashboard**: Streamlit web interface for predictions, analytics, and model management
- **MLOps Infrastructure**:  training, evaluation, model versioning, and deployment
- **Performance Testing**: Locust-based load testing 
- **Cloud Deployment**: Render deployment 

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │    MongoDB      │
│   (Dashboard)   │◄──►│  (11 Endpoints) │◄──►│   (GridFS)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Load Testing   │    │  CNN+LSTM Model │    │  File Storage   │
│   (Locust)      │    │  (TensorFlow)   │    │  (Data/Models)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Model Architecture

### CNN+LSTM Temporal Model
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Temporal Processing**: LSTM layers for sequence analysis
- **Input**: 2-5 sequential images (224x224x3)
- **Output**: Binary classification (Accident/No Accident)
- **Performance**: 92.5% validation accuracy
- **Features**: Transfer learning, dropout regularization, batch normalization

### Training Pipeline
- **Data Augmentation**: Rotation, brightness, zoom, horizontal flip
- **Optimization**: Adam optimizer with learning rate scheduling
- **Callbacks**: Early stopping, model checkpointing, reduce LR on plateau
- **Evaluation**: Accuracy, precision, recall, F1-score metrics


## Deployment
   - FastAPI (on RENDER): [http://accidentdetectionmlops/docs](https://accidentdetectionmlops.onrender.com/docs)
   - Streamlit Dashboard: [http://accidentdetection.streamlit.app](https://accidentdetection.streamlit.app/)

## Demo Video
- YouTube : [http://accidentdetectionmlops/docs](https://accidentdetectionmlops.onrender.com/docs)

## Quick Start

### Prerequisites
- Python 3.9+
- MongoDB database (optional for enhanced features)
- TensorFlow 2.19.0
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/excelasaph/AccidentDetectionMLOps.git
   cd AccidentDetectionMLOps
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up dataset structure**
   ```
   data/
   ├── train/
   │   ├── Accident/
   │   └── Non Accident/
   ├── test/
   │   ├── Accident/
   │   └── Non Accident/
   └── val/
       ├── Accident/
       └── Non Accident/
   ```

4. **Start the FastAPI server**
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Launch Streamlit dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Access the application**
   - FastAPI Docs: http://localhost:8000/docs
   - Streamlit Dashboard: http://localhost:8501

## API Documentation

### Base URL
- **Local**: `http://localhost:8000`
- **Production**: `https://accidentdetectionmlops.onrender.com`
- **API Docs**: `/docs` (Interactive Swagger documentation)

### Core Endpoints

#### Prediction & Analysis
- `POST /predict` - Accident prediction from image sequence (2-5 images)
- `GET /prediction-history?limit=50` - Get recent prediction history
- `GET /prediction-stats` - Prediction analytics and statistics

#### Training & Model Management
- `POST /retrain` - Retrain model with new data (MongoDB + file-based)
- `GET /training-data-sources` - Get available training data information
- `POST /upload` - Upload new training images with labels

#### Data Management
- `POST /add-prediction-to-training` - Convert predictions to training data
- `GET /storage-stats` - MongoDB GridFS storage statistics
- `DELETE /clear-predictions` - Clear all prediction history
- `DELETE /clear-training-data` - Clear uploaded training data

#### System Health
- `GET /health` - API health status and system information

### Example API Usage

#### Make a Prediction
```bash
curl -X POST "https://accidentdetectionmlops.onrender.com/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

#### Upload Training Data
```bash
curl -X POST "https://accidentdetectionmlops.onrender.com/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@accident1.jpg" \
  -F "files=@accident2.jpg" \
  -F "label=Accident"
```

#### Retrain Model
```bash
curl -X POST "https://accidentdetectionmlops.onrender.com/retrain" \
  -H "Content-Type: application/json"
```

#### Get Prediction Statistics
```bash
curl -X GET "https://accidentdetectionmlops.onrender.com/prediction-stats"
```

## Machine Learning Pipeline

### Data Processing
- **Image Preprocessing**: Resize to 224x224, normalization, augmentation
- **Sequence Generation**: 2-5 frame temporal sequences for LSTM input
- **Data Augmentation**: Rotation, brightness adjustment, zoom, horizontal flip
- **Validation**: 80/10/10 train/validation/test split

### Model Training
The project includes a comprehensive Jupyter notebook (`notebook/accident_detection.ipynb`) with:

- **Data Exploration**: Class distribution, image quality analysis
- **Model Architecture**: CNN+LSTM with transfer learning
- **Training Process**: Early stopping, learning rate scheduling, model checkpointing
- **Evaluation**: Comprehensive metrics including confusion matrix, ROC curves

### Model Performance
- **Architecture**: MobileNetV2 + LSTM temporal model
- **Validation Accuracy**: 92.5%
- **Training Features**: Transfer learning, dropout regularization, batch normalization
- **Input Format**: Sequential images (224x224x3)
- **Classes**: Binary classification (Accident/No Accident)

### Retraining Pipeline
1. **Data Ingestion**: Combines file-based data with MongoDB uploaded images
2. **Model Loading**: Loads existing model or creates new architecture
3. **Training**: Automated training with callbacks and validation
4. **Evaluation**: Performance metrics on test data
5. **Model Saving**: Automatic model versioning and backup

## Streamlit Dashboard Features

### Interactive Pages
- **Dashboard**: Overview of system status and key metrics
- **Prediction Interface**: Upload images and get real-time predictions
- **Analytics**: Prediction history, statistics, and visualizations
- **Training Management**: Upload training data and trigger retraining
- **Model Performance**: Training history and evaluation metrics
- **Data Visualizations**: Dataset insights and model analysis
- **System Status**: API health, database status, storage metrics

### Key Capabilities
- **Real-time Predictions**: Upload 2-5 images for instant accident detection
- **Data Upload**: Bulk upload of training images with labeling
- **Model Retraining**: One-click model retraining with progress monitoring
- **Analytics Dashboard**: Comprehensive prediction analytics and trends
- **Data Management**: Clear prediction history and training data

## Load Testing & Performance

### Locust Configuration
- **Test File**: `src/locustfile.py`
- **User Classes**: AccidentDetectionUser, DataUploadUser, StressTestUser
- **Test Scenarios**: Health checks, predictions, data uploads, analytics

### Performance Testing
```bash
# Local testing
locust -f src/locustfile.py --host=http://localhost:8000

# Production testing
locust -f src/locustfile.py --host=https://accidentdetectionmlops.onrender.com

# Headless testing
locust -f src/locustfile.py --host=https://accidentdetectionmlops.onrender.com \
  --users 50 --spawn-rate 2 --run-time 5m --headless
```

### Performance Results
| Configuration | Avg Response Time | 95th Percentile | Requests/sec | Users | Failures |
|---------------|------------------|-----------------|--------------|-------|----------|
| **Health Check** | ~150ms | ~200ms | 25 RPS | 50 users | <1% |
| **Prediction** | ~2.5s | ~4s | 8 RPS | 20 users | <2% |
| **Analytics** | ~300ms | ~500ms | 15 RPS | 30 users | <1% |

## Project Structure

```
AccidentDetectionMLOps/
├── src/                         
│   ├── main.py                  
│   ├── model.py                 
│   ├── preprocessing.py         
│   ├── prediction.py           
│   ├── database.py              
│   ├── locustfile.py            
│   └── api/                     
├── data/                       
│   ├── train/                  
│   ├── test/                    
│   └── val/                     
├── models/                      
│   └── accident_model.keras     
├── notebook/                    
│   └── accident_detection.ipynb 
├── static/                      
├── streamlit_app.py            
├── requirements.txt            
├── runtime.txt                  
└── README.md                    
```

## Deployment

### Production Deployment (Render)
- **URL**: https://accidentdetectionmlops.onrender.com
- **API Docs**: https://accidentdetectionmlops.onrender.com/docs
- **Runtime**: Python 3.9.11 with TensorFlow CPU-only mode
- **Features**: Auto-scaling, health monitoring, zero-downtime deployment

### Local Development
```bash
# Start FastAPI server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Launch Streamlit dashboard  
streamlit run streamlit_app.py

# Run load tests
locust -f src/locustfile.py --host=http://localhost:8000
```


## Dataset Information

**Source**: [Accident Detection Dataset](https://www.kaggle.com/datasets/accident-detection)

**Statistics**:
- **Training Images**: 1,500+ images per class
- **Test Images**: 300+ images per class  
- **Validation Images**: 200+ images per class
- **Format**: JPG images (various resolutions)
- **Classes**: Binary classification (Accident/No Accident)

**Preprocessing**:
- Resize to 224x224 pixels
- Normalization to [0,1] range
- Data augmentation (rotation, brightness, zoom)
- Sequence generation for temporal analysis

## Demo & Resources

### 🔗 Live Application
- **Production API**: https://accidentdetectionmlops.onrender.com
- **API Documentation**: https://accidentdetectionmlops.onrender.com/docs
- **Health Check**: https://accidentdetectionmlops.onrender.com/health

### 📺 Video Demo
[YouTube Video Demo](https://youtube.com) - *Pending creation*

### 🛠️ Development Resources
- **GitHub Repository**: https://github.com/excelasaph/AccidentDetectionMLOps
- **Jupyter Notebook**: `notebook/accident_detection.ipynb`
- **Load Testing**: `src/locustfile.py`

