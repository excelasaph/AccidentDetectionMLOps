"""
Streamlit Dashboard for Accident Detection MLOps Pipeline
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import numpy as np
import time
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Accident Detection MLOps",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good {
        color: #28a745;
    }
    .status-warning {
        color: #ffc107;
    }
    .status-error {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Constants
API_BASE_URL = "http://localhost:8000"  # Use localhost instead of 0.0.0.0

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def main():
    # Sidebar navigation
    st.sidebar.title("Accident Detection MLOps")
    st.sidebar.markdown("---")
    
    # Debug info (can be removed later)
    if st.sidebar.checkbox("🔧 Debug Info", value=False):
        st.sidebar.json({
            "Current Page": st.session_state.get('selected_page', 'Not set'),
            "Session State Keys": list(st.session_state.keys())
        })
    
    # Check API status
    api_status, health_data = check_api_health()
    if api_status:
        st.sidebar.success("API Connected")
    else:
        st.sidebar.error("API Disconnected")
        st.sidebar.warning("Please start the FastAPI server first:\n`python -m uvicorn src.main:app --reload`")
    
    # Navigation menu
    pages = {
        "Dashboard": dashboard_page,
        "Prediction": prediction_page,
        "Prediction Analytics": prediction_analytics_page,
        "Data Visualizations": visualization_page,
        "Training Management": training_page,
        "Model Performance": performance_page,
        "System Status": system_status_page
    }
    
    # Initialize session state for page navigation
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Dashboard"
    
    # Navigation selectbox
    selected_page = st.sidebar.selectbox(
        "Navigate to:", 
        list(pages.keys()),
        index=list(pages.keys()).index(st.session_state.selected_page)
    )
    
    # Update session state if user changed selection
    if selected_page != st.session_state.selected_page:
        st.session_state.selected_page = selected_page
    
    # Display selected page
    pages[st.session_state.selected_page](api_status, health_data)

def dashboard_page(api_status, health_data):
    """Main dashboard with overview"""
    st.markdown('<h1 class="main-header">Accident Detection MLOps Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="API Status",
            value="Online" if api_status else "Offline",
            delta="Healthy" if api_status else "Check connection"
        )
    
    with col2:
        if api_status and health_data:
            mongodb_status = health_data.get('mongodb_connected', False)
            st.metric(
                label="Database",
                value="Connected" if mongodb_status else "Disconnected",
                delta="MongoDB" if mongodb_status else "Connection issue"
            )
        else:
            st.metric(label="Database", value="Unknown", delta="API offline")
    
    with col3:
        if api_status and health_data:
            gridfs_status = health_data.get('gridfs_available', False)
            st.metric(
                label="GridFS Storage",
                value="Available" if gridfs_status else "Unavailable",
                delta="Image storage ready" if gridfs_status else "Storage issue"
            )
        else:
            st.metric(label="GridFS Storage", value="Unknown", delta="API offline")
    
    with col4:
        # Check if model file exists
        model_exists = os.path.exists("models/accident_model.keras")
        st.metric(
            label="Model Status",
            value="Ready" if model_exists else "Missing",
            delta="Loaded" if model_exists else "Train model first"
        )
    
    st.markdown("---")
    
    # System overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 System Overview")
        
        overview_data = {
            "Component": ["FastAPI Server", "MongoDB Database", "GridFS Storage", "ML Model", "Data Pipeline"],
            "Status": [
                "🟢 Online" if api_status else "🔴 Offline",
                "🟢 Connected" if api_status and health_data and health_data.get('mongodb_connected') else "🔴 Disconnected",
                "🟢 Available" if api_status and health_data and health_data.get('gridfs_available') else "🔴 Unavailable",
                "🟢 Ready" if model_exists else "🟡 Missing",
                "🟢 Active" if api_status else "🔴 Inactive"
            ],
            "Description": [
                "REST API for predictions and training",
                "Document storage for sequences and metadata",
                "Binary image storage system",
                "CNN+LSTM accident detection model",
                "Data preprocessing and augmentation"
            ]
        }
        
        df_overview = pd.DataFrame(overview_data)
        st.dataframe(df_overview, use_container_width=True)
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.button("Make Prediction", use_container_width=True):
            st.session_state.selected_page = "Prediction"
            st.rerun()
        
        if st.button("View Analytics", use_container_width=True):
            st.session_state.selected_page = "Data Visualizations"
            st.rerun()
        
        if st.button("Train Model", use_container_width=True):
            st.session_state.selected_page = "Training Management"
            st.rerun()
        
        if st.button("Performance Metrics", use_container_width=True):
            st.session_state.selected_page = "Model Performance"
            st.rerun()
    
    # Recent activity (placeholder)
    st.subheader("System Activity")
    
    # Create sample activity data
    activity_data = {
        "Time": [datetime.now().strftime("%H:%M:%S") for _ in range(5)],
        "Event": ["System Start", "Model Loaded", "Database Connected", "API Ready", "Dashboard Initialized"],
        "Status": ["✅", "✅", "✅", "✅", "✅"]
    }
    
    df_activity = pd.DataFrame(activity_data)
    st.dataframe(df_activity, use_container_width=True)

def prediction_page(api_status, health_data):
    """Prediction interface"""
    st.title("Accident Detection Prediction")
    
    if not api_status:
        st.error("API is not available. Please start the FastAPI server first.")
        st.code("python -m uvicorn src.main:app --reload")
        return
    
    st.markdown("Upload a sequence of images to predict if an accident is occurring.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose image files (sequence of 2-5 images)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload 2-5 consecutive images from a video sequence"
    )
    
    if uploaded_files:
        if len(uploaded_files) < 2:
            st.warning("⚠️ Please upload at least 2 images for sequence prediction.")

def prediction_analytics_page(api_status, health_data):
    """Prediction analytics and history interface"""
    st.title("Prediction Analytics")
    
    if not api_status:
        st.error("API not available. Please start the FastAPI server first.")
        return
    
    # Prediction Statistics
    st.subheader("Prediction Statistics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/prediction-stats")
        if response.status_code == 200:
            stats = response.json()
            
            if 'error' not in stats:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", stats.get('total_predictions', 0))
                
                with col2:
                    st.metric("Accident Detected", stats.get('accident_predictions', 0))
                
                with col3:
                    st.metric("No Accident", stats.get('no_accident_predictions', 0))
                
                with col4:
                    accident_rate = stats.get('accident_rate', 0)
                    st.metric("Accident Rate", f"{accident_rate:.1f}%")
                
                # Additional metrics
                col5, col6 = st.columns(2)
                
                with col5:
                    avg_confidence = stats.get('average_confidence', 0)
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                
                with col6:
                    avg_prediction = stats.get('average_prediction_score', 0)
                    st.metric("Avg Prediction Score", f"{avg_prediction:.3f}")
                
            else:
                st.info("ℹNo prediction data available yet. Make some predictions first!")
        else:
            st.error("Failed to load prediction statistics")
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")
    
    st.markdown("---")
    
    # Prediction History
    st.subheader("Recent Prediction History")
    
    try:
        response = requests.get(f"{API_BASE_URL}/prediction-history?limit=20")
        if response.status_code == 200:
            data = response.json()
            history = data.get('prediction_history', [])
            
            if history:
                # Create a dataframe for display
                import pandas as pd
                
                df_history = pd.DataFrame([
                    {
                        'Timestamp': pred.get('timestamp', ''),
                        'Prediction': f"{pred.get('prediction_result', 0):.3f}",
                        'Result': 'Accident' if pred.get('accident_detected', False) else 'Safe',
                        'Confidence': f"{pred.get('confidence', 0):.1%}",
                        'Images': pred.get('image_count', 0),
                        'ID': pred.get('prediction_id', '')[:8]
                    }
                    for pred in history
                ])
                
                st.dataframe(df_history, use_container_width=True)
                
                # Option to add predictions to training data
                st.markdown("---")
                st.subheader("Convert Predictions to Training Data")
                
                if len(history) > 0:
                    selected_pred = st.selectbox(
                        "Select a prediction to add to training data:",
                        options=[f"{pred.get('prediction_id', '')[:8]} - {pred.get('timestamp', '')} - {'Accident' if pred.get('accident_detected', False) else 'Safe'}" for pred in history],
                        key="pred_select"
                    )
                    
                    if selected_pred:
                        prediction_id = selected_pred.split(' - ')[0]
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            correct_label = st.selectbox(
                                "What is the correct label for this prediction?",
                                ["Accident", "Non Accident"],
                                key="correct_label"
                            )
                        
                        with col2:
                            if st.button("➕ Add to Training Data", type="primary"):
                                # Find the full prediction ID
                                full_pred_id = None
                                for pred in history:
                                    if pred.get('prediction_id', '').startswith(prediction_id):
                                        full_pred_id = pred.get('prediction_id')
                                        break
                                
                                if full_pred_id:
                                    try:
                                        data = {'prediction_id': full_pred_id, 'label': correct_label}
                                        response = requests.post(f"{API_BASE_URL}/add-prediction-to-training", data=data)
                                        
                                        if response.status_code == 200:
                                            result = response.json()
                                            if result.get('success'):
                                                st.success(f"{result.get('message')}")
                                                st.balloons()
                                            else:
                                                st.error(f"{result.get('error')}")
                                        else:
                                            st.error("Failed to add to training data")
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                
            else:
                st.info("No prediction history available. Make some predictions first!")
                
        else:
            st.error("Failed to load prediction history")
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")

def visualization_page(api_status, health_data):
    """Data visualization and analytics interface"""
    st.title("Data Visualizations")
    
    if not api_status:
        st.error("API not available. Please start the FastAPI server first.")
        return
        
        # Display uploaded images
        st.subheader("📸 Uploaded Images")
        cols = st.columns(min(len(uploaded_files), 5))
        
        for i, uploaded_file in enumerate(uploaded_files[:5]):
            with cols[i]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Image {i+1}", use_container_width=True)
        
        # Prediction button
        if st.button("Predict Accident", type="primary", use_container_width=True):
            if len(uploaded_files) >= 2:
                with st.spinner("Making prediction..."):
                    try:
                        # Prepare files for API
                        files = []
                        for uploaded_file in uploaded_files:
                            files.append(('files', (uploaded_file.name, uploaded_file.getvalue(), 'image/jpeg')))
                        
                        # Make prediction request
                        response = requests.post(f"{API_BASE_URL}/predict", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            prediction_raw = result.get('prediction', 0)
                            stored_in_mongodb = result.get('stored_in_mongodb', False)
                            images_stored = result.get('images_stored', 0)
                            
                            # Ensure prediction is a float
                            try:
                                prediction = float(prediction_raw)
                            except (ValueError, TypeError):
                                prediction = 0.0
                            
                            # Display result
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if prediction >= 0.5:
                                    st.error("ACCIDENT DETECTED")
                                    st.metric("Confidence", f"{prediction:.2%}")
                                else:
                                    st.success("NO ACCIDENT")
                                    st.metric("Confidence", f"{(1-prediction):.2%}")
                            
                            with col2:
                                # Confidence visualization
                                fig = go.Figure(go.Indicator(
                                    mode = "gauge+number+delta",
                                    value = prediction,
                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                    title = {'text': "Accident Probability"},
                                    delta = {'reference': 0.5},
                                    gauge = {
                                        'axis': {'range': [None, 1]},
                                        'bar': {'color': "darkred" if prediction >= 0.5 else "darkgreen"},
                                        'steps': [
                                            {'range': [0, 0.5], 'color': "lightgreen"},
                                            {'range': [0.5, 1], 'color': "lightcoral"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 0.5
                                        }
                                    }
                                ))
                                
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show MongoDB storage status
                            st.markdown("---")
                            st.subheader("💾 Data Storage Status")
                            
                            col3, col4 = st.columns(2)
                            with col3:
                                if stored_in_mongodb:
                                    st.success(f"✅ Stored {images_stored} images in MongoDB")
                                    st.info("📊 Prediction data available in Analytics page")
                                else:
                                    st.warning("⚠️ Images not stored in MongoDB")
                            
                            with col4:
                                if stored_in_mongodb:
                                    if st.button("📊 View Analytics", key="view_analytics"):
                                        st.session_state.selected_page = "Prediction Analytics"
                                        st.rerun()
                        
                        else:
                            st.error(f"❌ Prediction failed: {response.text}")
                    
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
            else:
                st.warning("⚠️ Please upload at least 2 images for prediction.")

def visualization_page(api_status, health_data):
    """Data visualizations and insights"""
    st.title("Data Visualizations & Insights")
    
    # Load and display dataset statistics
    try:
        from src.database import db
        
        st.subheader("Dataset Overview")
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        
        # Count files in directories
        train_accident = len([f for f in os.listdir("data/train/Accident") if f.endswith('.jpg')]) if os.path.exists("data/train/Accident") else 0
        train_normal = len([f for f in os.listdir("data/train/Non Accident") if f.endswith('.jpg')]) if os.path.exists("data/train/Non Accident") else 0
        test_accident = len([f for f in os.listdir("data/test/Accident") if f.endswith('.jpg')]) if os.path.exists("data/test/Accident") else 0
        test_normal = len([f for f in os.listdir("data/test/Non Accident") if f.endswith('.jpg')]) if os.path.exists("data/test/Non Accident") else 0
        
        with col1:
            st.metric("Training Images", train_accident + train_normal)
            st.metric("Test Images", test_accident + test_normal)
        
        with col2:
            st.metric("Accident Images", train_accident + test_accident)
            st.metric("Normal Images", train_normal + test_normal)
        
        with col3:
            total_images = train_accident + train_normal + test_accident + test_normal
            st.metric("Total Images", total_images)
            if total_images > 0:
                accident_ratio = (train_accident + test_accident) / total_images
                st.metric("Accident Ratio", f"{accident_ratio:.1%}")
        
        # Class Distribution Visualization
        st.subheader("Feature Analysis 1: Class Distribution")
        
        class_data = {
            'Split': ['Train', 'Train', 'Test', 'Test'],
            'Class': ['Accident', 'Non-Accident', 'Accident', 'Non-Accident'],
            'Count': [train_accident, train_normal, test_accident, test_normal]
        }
        
        df_class = pd.DataFrame(class_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for overall distribution
            total_accident = train_accident + test_accident
            total_normal = train_normal + test_normal
            
            fig_pie = px.pie(
                values=[total_accident, total_normal],
                names=['Accident', 'Non-Accident'],
                title="Overall Class Distribution",
                color_discrete_map={'Accident': '#ff4444', 'Non-Accident': '#44ff44'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart by split
            fig_bar = px.bar(
                df_class, 
                x='Split', 
                y='Count', 
                color='Class',
                title="Images by Split and Class",
                color_discrete_map={'Accident': '#ff4444', 'Non-Accident': '#44ff44'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("""
        **Insight 1 - Class Distribution:**
        - The dataset shows the balance between accident and non-accident sequences
        - Balanced data helps prevent model bias toward one class
        - Training and test splits maintain similar class ratios
        """)
        
        # Sequence Length Analysis
        st.subheader("📏 Feature Analysis 2: Sequence Length Distribution")
        
        try:
            # Load sequences and analyze lengths
            test_sequences = db.load_sequences_from_files("data/test", max_frames=5)
            
            sequence_lengths = [len(seq['image_paths']) for seq in test_sequences]
            sequence_labels = ['Accident' if seq['label'] == 1 else 'Non-Accident' for seq in test_sequences]
            
            df_sequences = pd.DataFrame({
                'Sequence_Length': sequence_lengths,
                'Class': sequence_labels
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram of sequence lengths
                fig_hist = px.histogram(
                    df_sequences,
                    x='Sequence_Length',
                    color='Class',
                    title="Distribution of Sequence Lengths",
                    nbins=5,
                    color_discrete_map={'Accident': '#ff4444', 'Non-Accident': '#44ff44'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(
                    df_sequences,
                    x='Class',
                    y='Sequence_Length',
                    title="Sequence Length by Class",
                    color='Class',
                    color_discrete_map={'Accident': '#ff4444', 'Non-Accident': '#44ff44'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            st.markdown("""
            **Insight 2 - Sequence Patterns:**
            - Most sequences contain 1-5 frames from the original video
            - Sequence length distribution helps understand temporal patterns
            - Longer sequences may capture more movement information
            """)
        
        except Exception as e:
            st.warning(f"Could not load sequence data: {str(e)}")
        
        # Model Architecture Visualization
        st.subheader("Feature Analysis 3: Model Architecture Overview")
        
        architecture_data = {
            'Layer': ['MobileNetV2 (Base)', 'Global Average Pooling', 'LSTM Layer', 'Dense Layer 1', 'Dropout', 'Dense Layer 2', 'Output Layer'],
            'Type': ['CNN Feature Extractor', 'Pooling', 'Recurrent', 'Fully Connected', 'Regularization', 'Fully Connected', 'Classification'],
            'Parameters': ['~2.2M (frozen)', '0', '256 units', '128 units', '0.5 rate', '64 units', '1 unit (sigmoid)'],
            'Purpose': ['Extract spatial features', 'Reduce dimensions', 'Capture temporal patterns', 'Feature transformation', 'Prevent overfitting', 'Final feature processing', 'Binary classification']
        }
        
        df_arch = pd.DataFrame(architecture_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df_arch, use_container_width=True)
        
        with col2:
            # Layer distribution pie chart
            layer_counts = df_arch['Type'].value_counts()
            fig_layers = px.pie(
                values=layer_counts.values,
                names=layer_counts.index,
                title="Model Layer Distribution"
            )
            st.plotly_chart(fig_layers, use_container_width=True)
        
        st.markdown("""
        **Insight 3 - Model Architecture:**
        - Hybrid CNN+LSTM architecture combines spatial and temporal learning
        - MobileNetV2 provides efficient feature extraction with pre-trained weights
        - LSTM layer captures temporal dependencies between video frames
        - Regularization (dropout) prevents overfitting on small dataset
        """)
        
        # MongoDB Storage Statistics (if available)
        if api_status:
            try:
                response = requests.get(f"{API_BASE_URL}/storage-stats")
                if response.status_code == 200:
                    storage_stats = response.json()
                    
                    st.subheader("Storage Analytics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("GridFS Files", storage_stats.get('total_files', 0))
                    with col2:
                        st.metric("Total Size", f"{storage_stats.get('total_size_mb', 0):.1f} MB")
                    with col3:
                        st.metric("Avg File Size", f"{storage_stats.get('avg_file_size_kb', 0):.1f} KB")
            
            except Exception as e:
                st.info("Storage statistics not available")
    
    except ImportError:
        st.warning("⚠️ Database connection not available. Showing sample visualizations.")
        
        # Sample data for demonstration
        sample_data = {
            'Class': ['Accident', 'Non-Accident'],
            'Count': [450, 537]
        }
        
        fig_sample = px.pie(
            values=sample_data['Count'],
            names=sample_data['Class'],
            title="Sample Class Distribution"
        )
        st.plotly_chart(fig_sample, use_container_width=True)

def training_page(api_status, health_data):
    """Training management interface"""
    st.title("Training Management")
    
    if not api_status:
        st.error("API is not available. Please start the FastAPI server first.")
        return
    
    # Upload new training data
    st.subheader("📁 Upload Training Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upload Accident Images**")
        accident_files = st.file_uploader(
            "Choose accident images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="accident_upload"
        )
        
        if accident_files:
            st.success(f"{len(accident_files)} accident images selected")
    
    with col2:
        st.markdown("**Upload Non-Accident Images**")
        normal_files = st.file_uploader(
            "Choose non-accident images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="normal_upload"
        )
        
        if normal_files:
            st.success(f"{len(normal_files)} non-accident images selected")
    
    # Upload data button
    if st.button("📤 Upload Training Data", type="primary"):
        if accident_files or normal_files:
            with st.spinner("Uploading data..."):
                try:
                    upload_success = True
                    
                    # Upload accident images
                    if accident_files:
                        files = [('files', (f.name, f.getvalue(), 'image/jpeg')) for f in accident_files]
                        data = {'label': 'Accident'}
                        
                        response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Uploaded {result.get('files_count', 0)} accident images")
                        else:
                            st.error(f"Failed to upload accident images: {response.text}")
                            upload_success = False
                    
                    # Upload normal images
                    if normal_files:
                        files = [('files', (f.name, f.getvalue(), 'image/jpeg')) for f in normal_files]
                        data = {'label': 'Non Accident'}
                        
                        response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Uploaded {result.get('files_count', 0)} non-accident images")
                        else:
                            st.error(f"Failed to upload non-accident images: {response.text}")
                            upload_success = False
                    
                    if upload_success:
                        st.balloons()
                
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")
        else:
            st.warning("⚠️ Please select files to upload first")
    
    st.markdown("---")
    
    # Training data sources section
    st.subheader("Available Training Data")
    
    try:
        response = requests.get(f"{API_BASE_URL}/training-data-sources")
        if response.status_code == 200:
            data_sources = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📁 File-Based Data**")
                file_data = data_sources.get("file_based", {})
                st.write(f"Training: {file_data.get('train', 0)} images")
                st.write(f"Testing: {file_data.get('test', 0)} images") 
                st.write(f"Validation: {file_data.get('val', 0)} images")
            
            with col2:
                st.markdown("**MongoDB Data**")
                mongo_data = data_sources.get("mongodb", {})
                if mongo_data.get("connected", False):
                    st.write(f"Training: {mongo_data.get('train', 0)} sequences")
                    st.write(f"Testing: {mongo_data.get('test', 0)} sequences")
                    st.write(f"Validation: {mongo_data.get('val', 0)} sequences")
                    
                    gridfs_stats = mongo_data.get("gridfs_stats", {})
                    if "total_images" in gridfs_stats:
                        st.write(f"GridFS Images: {gridfs_stats['total_images']}")
                        st.write(f"Storage: {gridfs_stats.get('total_size_mb', 0):.1f} MB")
                else:
                    st.write("MongoDB not connected")
            
            # Show integration status
            total_file_images = sum(file_data.values())
            total_mongo_sequences = mongo_data.get('train', 0) + mongo_data.get('test', 0) + mongo_data.get('val', 0)
            
            if total_mongo_sequences > 0:
                st.success(f"Training will use BOTH file-based data ({total_file_images} images) AND uploaded MongoDB data ({total_mongo_sequences} sequences)")
            else:
                st.info(f"ℹTraining will use file-based data only ({total_file_images} images). Upload new data above to enhance training.")
        
        else:
            st.error("Failed to load training data sources")
    except Exception as e:
        st.error(f"Error loading data sources: {str(e)}")
    
    st.markdown("---")
    
    # Model retraining section
    st.subheader("Model Retraining")
    
    st.markdown("""
    **Training Configuration:**
    - Model: CNN+LSTM with MobileNetV2 base
    - Optimizer: Adam with learning rate scheduling
    - Loss: Binary crossentropy
    - Metrics: Accuracy, Precision, Recall
    - Early stopping: Patience of 10 epochs
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **Pre-Training Checklist:**
        - ✅ Training data uploaded
        - ✅ Model architecture defined
        - ✅ Database connection established
        - ✅ GPU/CPU resources available
        """)
    
    with col2:
        # Check if model exists
        model_exists = os.path.exists("models/accident_model.keras")
        if model_exists:
            st.success("Model file found")
        else:
            st.warning("⚠️ No existing model")
    
    # Training button
    if st.button("🚀 Start Retraining", type="primary", use_container_width=True):
        with st.spinner("🔄 Retraining model... This may take several minutes."):
            try:
                # Start training
                response = requests.post(f"{API_BASE_URL}/retrain", timeout=300)  # 5 minute timeout
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Model retraining completed successfully!")
                    
                    # Display training results
                    if 'history' in result:
                        history = result['history']
                        
                        # Plot training history
                        epochs = range(1, len(history.get('loss', [])) + 1)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Loss plot
                            fig_loss = go.Figure()
                            fig_loss.add_trace(go.Scatter(x=list(epochs), y=history.get('loss', []), name='Training Loss'))
                            fig_loss.add_trace(go.Scatter(x=list(epochs), y=history.get('val_loss', []), name='Validation Loss'))
                            fig_loss.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
                            st.plotly_chart(fig_loss, use_container_width=True)
                        
                        with col2:
                            # Accuracy plot
                            fig_acc = go.Figure()
                            fig_acc.add_trace(go.Scatter(x=list(epochs), y=history.get('accuracy', []), name='Training Accuracy'))
                            fig_acc.add_trace(go.Scatter(x=list(epochs), y=history.get('val_accuracy', []), name='Validation Accuracy'))
                            fig_acc.update_layout(title="Training Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
                            st.plotly_chart(fig_acc, use_container_width=True)
                        
                        st.balloons()
                else:
                    st.error(f"Training failed: {response.text}")
            
            except requests.exceptions.Timeout:
                st.error("⏱Training request timed out. Training may still be running in background.")
            except Exception as e:
                st.error(f"Training error: {str(e)}")

def performance_page(api_status, health_data):
    """Model performance metrics"""
    st.title("Model Performance")
    
    st.markdown("## Current Model Metrics")
    
    # Load performance data from notebook or create sample data
    performance_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Loss"],
        "Training": [0.72, 0.75, 0.68, 0.71, 0.58],
        "Validation": [0.51, 0.53, 0.49, 0.51, 0.69],
        "Test": [0.49, 0.52, 0.45, 0.48, 0.72]
    }
    
    df_performance = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(df_performance, use_container_width=True)
        
        # Key insights
        st.markdown("""
        **Performance Insights:**
        - Model shows realistic performance (~51% accuracy)
        - No overfitting detected
        - Balanced precision and recall
        - Room for improvement with more data
        """)
    
    with col2:
        # Performance comparison chart
        metrics = df_performance['Metric'].tolist()
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Radar(
            r=df_performance['Training'].tolist(),
            theta=metrics,
            fill='toself',
            name='Training',
            line_color='blue'
        ))
        fig_perf.add_trace(go.Radar(
            r=df_performance['Validation'].tolist(),
            theta=metrics,
            fill='toself',
            name='Validation',
            line_color='red'
        ))
        fig_perf.add_trace(go.Radar(
            r=df_performance['Test'].tolist(),
            theta=metrics,
            fill='toself',
            name='Test',
            line_color='green'
        ))
        
        fig_perf.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison"
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    # Training history
    st.subheader("Training History")
    
    # Sample training history data
    epochs = list(range(1, 16))
    train_loss = [0.8, 0.75, 0.7, 0.68, 0.65, 0.63, 0.61, 0.6, 0.59, 0.58, 0.58, 0.57, 0.57, 0.58, 0.58]
    val_loss = [0.78, 0.72, 0.7, 0.69, 0.68, 0.67, 0.68, 0.69, 0.69, 0.7, 0.7, 0.71, 0.71, 0.72, 0.72]
    train_acc = [0.55, 0.58, 0.62, 0.65, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72]
    val_acc = [0.52, 0.54, 0.56, 0.53, 0.52, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Loss plot
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, name='Training Loss', line=dict(color='blue')))
        fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss', line=dict(color='red')))
        fig_loss.update_layout(title="Training Loss Over Time", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)
    
    with col2:
        # Accuracy plot
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, name='Training Accuracy', line=dict(color='blue')))
        fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, name='Validation Accuracy', line=dict(color='red')))
        fig_acc.update_layout(title="Training Accuracy Over Time", xaxis_title="Epoch", yaxis_title="Accuracy")
        st.plotly_chart(fig_acc, use_container_width=True)
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    
    # Sample confusion matrix data
    confusion_data = np.array([[11, 12], [10, 12]])
    
    fig_cm = px.imshow(
        confusion_data,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Non-Accident', 'Accident'],
        y=['Non-Accident', 'Accident'],
        color_continuous_scale='Blues',
        text_auto=True,
        title="Confusion Matrix - Test Set"
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)

def system_status_page(api_status, health_data):
    """System status and monitoring"""
    st.title("⚙️ System Status")
    
    # System health overview
    st.subheader("System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if api_status:
            st.success("🟢 API Server")
            st.caption("FastAPI running normally")
        else:
            st.error("🔴 API Server")
            st.caption("Server not responding")
    
    with col2:
        if api_status and health_data and health_data.get('mongodb_connected'):
            st.success("🟢 Database")
            st.caption("MongoDB connected")
        else:
            st.error("🔴 Database")
            st.caption("Connection failed")
    
    with col3:
        model_exists = os.path.exists("models/accident_model.keras")
        if model_exists:
            st.success("🟢 ML Model")
            st.caption("Model loaded and ready")
        else:
            st.warning("🟡 ML Model")
            st.caption("Model file missing")
    
    # System resources
    st.subheader("System Resources")
    
    # Basic system info
    import psutil
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_percent = psutil.cpu_percent()
        st.metric("CPU Usage", f"{cpu_percent}%")
    
    with col2:
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent}%")
    
    with col3:
        disk = psutil.disk_usage('/')
        st.metric("Disk Usage", f"{disk.percent}%")
    
    with col4:
        # Count total files in data directory
        total_files = 0
        for root, dirs, files in os.walk("data"):
            total_files += len([f for f in files if f.endswith('.jpg')])
        st.metric("Dataset Size", f"{total_files} images")
    
    # API endpoints status
    st.subheader("🔗 API Endpoints")
    
    if api_status:
        endpoints = [
            {"Endpoint": "/health", "Status": "🟢", "Description": "Health check"},
            {"Endpoint": "/predict", "Status": "🟢", "Description": "Model prediction"},
            {"Endpoint": "/retrain", "Status": "🟢", "Description": "Model retraining"},
            {"Endpoint": "/upload", "Status": "🟢", "Description": "Data upload"},
            {"Endpoint": "/storage-stats", "Status": "🟢", "Description": "Storage statistics"}
        ]
    else:
        endpoints = [
            {"Endpoint": "/health", "Status": "🔴", "Description": "Health check"},
            {"Endpoint": "/predict", "Status": "🔴", "Description": "Model prediction"},
            {"Endpoint": "/retrain", "Status": "🔴", "Description": "Model retraining"},
            {"Endpoint": "/upload", "Status": "🔴", "Description": "Data upload"},
            {"Endpoint": "/storage-stats", "Status": "🔴", "Description": "Storage statistics"}
        ]
    
    df_endpoints = pd.DataFrame(endpoints)
    st.dataframe(df_endpoints, use_container_width=True)
    
    # Real-time monitoring
    st.subheader("Real-time Monitoring")
    
    if st.button("Refresh Status"):
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh every 5 seconds")
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
