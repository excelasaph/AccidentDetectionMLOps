"""
Minimal Database Implementation for Assignment Requirements
Only stores essential data without redundancy.
"""

import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
from typing import List, Dict, Optional
import uuid
from datetime import datetime

load_dotenv()

# MongoDB connection details
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = "accident_detection_minimal"

class MinimalDatabase:
    def __init__(self):
        """Initialize MongoDB client - minimal setup."""
        try:
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client[DATABASE_NAME]
            self.client.server_info()
            self.connected = True
            print("Connected to MongoDB (minimal mode)")
        except ConnectionFailure:
            print("Failed to connect to MongoDB")
            self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    # MINIMAL: Only store essential training metadata (not images)
    def store_training_session(self, model_version: str, accuracy: float, loss: float, epochs: int) -> str:
        """Store only essential training results."""
        if not self.connected:
            return None
            
        session_id = str(uuid.uuid4())
        training_record = {
            'session_id': session_id,
            'model_version': model_version,
            'accuracy': accuracy,
            'loss': loss,
            'epochs': epochs,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        collection = self.db['training_sessions']
        collection.insert_one(training_record)
        return session_id

    # MINIMAL: Store only prediction results (not images)
    def store_prediction_result(self, prediction: float, image_count: int) -> str:
        """Store minimal prediction data."""
        if not self.connected:
            return None
            
        prediction_id = str(uuid.uuid4())
        prediction_record = {
            'prediction_id': prediction_id,
            'prediction_score': prediction,
            'image_count': image_count,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        collection = self.db['predictions']
        collection.insert_one(prediction_record)
        return prediction_id

    # MINIMAL: Get basic statistics
    def get_prediction_stats(self) -> Dict:
        """Get basic prediction statistics."""
        if not self.connected:
            return {'error': 'Database not connected'}
            
        try:
            collection = self.db['predictions']
            total_predictions = collection.count_documents({})
            
            # Simple aggregation
            pipeline = [
                {'$group': {
                    '_id': None,
                    'avg_prediction': {'$avg': '$prediction_score'},
                    'total_count': {'$sum': 1}
                }}
            ]
            
            result = list(collection.aggregate(pipeline))
            avg_prediction = result[0]['avg_prediction'] if result else 0
            
            return {
                'total_predictions': total_predictions,
                'average_prediction': avg_prediction
            }
            
        except Exception as e:
            return {'error': str(e)}

    # MINIMAL: Get training history
    def get_training_history(self, limit: int = 10) -> List[Dict]:
        """Get recent training sessions."""
        if not self.connected:
            return []
            
        try:
            collection = self.db['training_sessions']
            sessions = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
            return sessions
        except Exception as e:
            print(f"Error getting training history: {e}")
            return []

# Singleton instance
minimal_db = MinimalDatabase()
