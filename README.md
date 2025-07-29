# MLOps Image Classification Project

## Project Description
This project implements an end-to-end Machine Learning pipeline for classifying images (e.g., cats vs. dogs) using TensorFlow. It includes data preprocessing, model training, evaluation, API creation, a React-based UI, and cloud deployment on AWS. The system supports single image predictions, bulk data uploads for retraining, and visualizations of dataset features. Locust is used to simulate request floods and evaluate model performance.

## Setup Instructions

### Clone the Repository:
```bash
git clone <your-repo-url>
cd Project_name
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Dataset:

Place training and test images in `data/train/` and `data/test/` respectively.
Example dataset: Kaggle Cats vs Dogs.

### Run Jupyter Notebook:
```bash
jupyter notebook notebook/project_name.ipynb
```

### Run the Flask API:
```bash
python src/app.py
```

### Run the React UI:
```bash
cd ui
npm install
npm start
```

### Deploy on AWS:

Use AWS EC2 or Elastic Beanstalk for deployment.
Follow the deployment steps in the Deploy on AWS section below.

### Run Locust for Load Testing:
```bash
locust -f src/locustfile.py --host=http://<your-api-url>
```

## Video Demo
YouTube Link to Video Demo (Replace with your YouTube link)

## URL

API: http://<your-api-url>
UI: http://<your-ui-url>

## Flood Request Simulation Results

1 Docker Container: Latency ~200ms, Response Time ~250ms for 100 users.
2 Docker Containers: Latency ~150ms, Response Time ~180ms for 100 users.
(Update with actual results after running Locust)
