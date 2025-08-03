import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Optional
import glob
import gridfs
import base64
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# MongoDB connection details
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = "accident_detection_db"
TRAIN_COLLECTION = "train"
TEST_COLLECTION = "test"
VAL_COLLECTION = "val"

class Database:
    def __init__(self):
        """Initialize MongoDB client with GridFS support."""
        try:
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client[DATABASE_NAME]
            self.fs = gridfs.GridFS(self.db)  # GridFS for image storage
            self.client.server_info()  # Test connection
            self.connected = True
            print("Connected to MongoDB with GridFS support")
        except ConnectionFailure:
            print("Failed to connect to MongoDB. Using file-based fallback.")
            self.connected = False
            self.fs = None

    def is_connected(self) -> bool:
        """Check if the database is connected."""
        return self.connected

    def validate_sequence(self, sequence: Dict) -> None:
        """Validate sequence document structure."""
        required_fields = {"sequence_id", "image_paths", "label", "split"}
        if not all(field in sequence for field in required_fields):
            raise ValueError(f"Sequence must have fields: {required_fields}")
        if not isinstance(sequence["image_paths"], list):
            raise ValueError("image_paths must be a list")
        if not all(os.path.exists(path) for path in sequence["image_paths"]):
            raise ValueError("Some image paths do not exist")
        if sequence["label"] not in [0, 1]:
            raise ValueError("Label must be 0 or 1")
        if sequence["split"] not in ["train", "test", "val"]:
            raise ValueError("Split must be train, test, or val")

    def save_to_collection(self, sequences: List[Dict], collection_name: str) -> None:
        """Save list of sequence documents to a MongoDB collection."""
        if not self.connected:
            raise ConnectionFailure("Database not connected")
        for seq in sequences:
            self.validate_sequence(seq)
        collection = self.db[collection_name]
        if sequences:
            collection.insert_many(sequences)
            print(f"Saved {len(sequences)} sequences to {collection_name}")

    def load_from_collection(self, collection_name: str) -> List[Dict]:
        """Load sequences from a MongoDB collection."""
        if not self.connected:
            return []
        collection = self.db[collection_name]
        data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB _id
        return data if data else []

    def clear_collection(self, collection_name: str) -> None:
        """Clear all data from a collection."""
        if self.connected:
            self.db[collection_name].delete_many({})
            print(f"Cleared collection {collection_name}")

    def load_sequences_from_files(self, directory: str, max_frames: int = 5) -> List[Dict]:
        """Load sequences from file system, supporting new naming convention."""
        image_files = glob.glob(os.path.join(directory, '*/*'))
        sequences = {}
        
        for file in image_files:
            filename = os.path.basename(file)
            parent_dir = os.path.basename(os.path.dirname(file))
            
            parts = filename.split('_')
            
            if len(parts) >= 4:
                # Handle new format
                if parts[0] == 'non' and len(parts) >= 5:
                    # non_accident_train_1_1.jpg
                    sequence_num = parts[3]  # '1'
                elif parts[0] == 'accident':
                    # accident_train_1_1.jpg  
                    sequence_num = parts[2]  # '1'
                else:
                    # Fallback
                    sequence_id = f"{parent_dir}_{filename.split('.')[0]}"
                    if sequence_id not in sequences:
                        sequences[sequence_id] = []
                    sequences[sequence_id].append(file)
                    continue
                
                # Create sequence_id
                sequence_id = f"{parent_dir}_{sequence_num}"
                
            else:
                # Old format fallback
                base_id = '_'.join(filename.split('_')[:-1]) if '_' in filename else filename.split('.')[0]
                sequence_id = f"{parent_dir}_{base_id}"
            
            if sequence_id not in sequences:
                sequences[sequence_id] = []
            sequences[sequence_id].append(file)
        
        result = []
        for seq_id, files in sequences.items():
            if len(files) == 0:
                continue
                
            # Sort by frame number
            try:
                files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            except:
                files.sort()
                
            # Determine label from parent directory
            parent_dir = os.path.basename(os.path.dirname(files[0]))
            label = 1 if parent_dir == 'Accident' else 0
            
            # Handle max_frames
            if len(files) > max_frames:
                files = files[:max_frames]
            else:
                while len(files) < max_frames:
                    files.append(files[-1])
            
            # Determine split
            split = "train" if "train" in directory else "test" if "test" in directory else "val"
            
            # Clean sequence_id for storage
            clean_sequence_id = seq_id.split('_', 1)[1] if '_' in seq_id else seq_id
            
            result.append({
                "sequence_id": clean_sequence_id,
                "image_paths": files,
                "label": label,
                "split": split
            })
        print(f"Loaded {len(result)} sequences from {directory}")
        return result

    def store_image_in_gridfs(self, image_path: str, metadata: Dict = None) -> Optional[str]:
        """
        Store an image file in GridFS.
        
        Args:
            image_path: Path to the image file
            metadata: Additional metadata (sequence_id, label, etc.)
        
        Returns:
            GridFS file ID as string, or None if failed
        """
        if not self.connected or not self.fs:
            return None
            
        try:
            with open(image_path, 'rb') as img_file:
                filename = os.path.basename(image_path)
                
                file_metadata = {
                    'filename': filename,
                    'original_path': image_path,
                    'content_type': 'image/jpeg'
                }
                
                if metadata:
                    file_metadata.update(metadata)
                
                file_id = self.fs.put(
                    img_file,
                    filename=filename,
                    metadata=file_metadata
                )
                
                return str(file_id)
                
        except Exception as e:
            print(f"Error storing image {image_path}: {e}")
            return None

    def retrieve_image_from_gridfs(self, file_id: str) -> Optional[bytes]:
        """
        Retrieve an image from GridFS by file ID.
        
        Args:
            file_id: GridFS file ID
            
        Returns:
            Image data as bytes, or None if not found
        """
        if not self.connected or not self.fs:
            return None
            
        try:
            from bson import ObjectId
            grid_out = self.fs.get(ObjectId(file_id))
            return grid_out.read()
        except Exception as e:
            print(f"Error retrieving image {file_id}: {e}")
            return None

    def save_sequences_with_images(self, sequences: List[Dict], collection_name: str, store_images: bool = False) -> None:
        """
        Save sequences to MongoDB, optionally storing images in GridFS.
        
        Args:
            sequences: List of sequence dictionaries
            collection_name: Name of the collection
            store_images: Whether to store actual image files in GridFS
        """
        if not self.connected:
            raise ConnectionFailure("Database not connected")
        
        processed_sequences = []
        
        for seq in sequences:
            if store_images:
                # Store images in GridFS and replace paths with file IDs
                stored_images = []
                
                for img_path in seq['image_paths']:
                    metadata = {
                        'sequence_id': seq['sequence_id'],
                        'label': seq['label'],
                        'split': seq['split'],
                        'frame_number': len(stored_images) + 1
                    }
                    
                    file_id = self.store_image_in_gridfs(img_path, metadata)
                    if file_id:
                        stored_images.append(file_id)
                
                # Create new sequence with file IDs
                new_seq = {
                    'sequence_id': seq['sequence_id'],
                    'image_file_ids': stored_images,
                    'label': seq['label'],
                    'split': seq['split'],
                    'image_count': len(stored_images),
                    'storage_type': 'gridfs'
                }
                processed_sequences.append(new_seq)
            else:
                # Store with file paths (existing behavior)
                self.validate_sequence(seq)
                processed_sequences.append(seq)
        
        # Save to collection
        collection = self.db[collection_name]
        if processed_sequences:
            collection.insert_many(processed_sequences)
            storage_type = "with GridFS images" if store_images else "with file paths"
            print(f"Saved {len(processed_sequences)} sequences to {collection_name} {storage_type}")

    def load_sequences_with_images(self, collection_name: str) -> List[Dict]:
        """
        Load sequences from MongoDB and handle both file paths and GridFS storage.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            List of sequences with usable image data
        """
        if not self.connected:
            return []
        
        collection = self.db[collection_name]
        sequences = list(collection.find({}, {"_id": 0}))
        
        # Process sequences based on storage type
        for seq in sequences:
            if seq.get('storage_type') == 'gridfs' and 'image_file_ids' in seq:
                # Convert GridFS IDs back to usable image data
                seq['images_available'] = True
                seq['image_count'] = len(seq['image_file_ids'])
            elif 'image_paths' in seq:
                # Check if file paths still exist
                seq['images_available'] = all(os.path.exists(path) for path in seq['image_paths'])
        
        return sequences

    def get_gridfs_stats(self) -> Dict:
        """Get statistics about GridFS storage."""
        if not self.connected or not self.fs:
            return {'error': 'GridFS not available'}
            
        try:
            files = list(self.fs.find())
            total_files = len(files)
            total_size = sum(f.length for f in files)
            
            return {
                'total_images': total_files,
                'total_size_mb': total_size / (1024 * 1024),
                'average_size_kb': (total_size / total_files / 1024) if total_files > 0 else 0
            }
        except Exception as e:
            return {'error': str(e)}

    def store_prediction_record(self, image_file_ids: List[str], prediction_result: float, confidence: float, timestamp: str = None) -> Optional[str]:
        """
        Store a prediction record with associated images in MongoDB.
        
        Args:
            image_file_ids: List of GridFS file IDs for the prediction images
            prediction_result: The prediction result (0.0 to 1.0)
            confidence: The confidence score
            timestamp: Optional timestamp, uses current time if not provided
            
        Returns:
            Prediction record ID as string, or None if failed
        """
        if not self.connected:
            return None
            
        try:
            from datetime import datetime
            import uuid
            
            if timestamp is None:
                timestamp = datetime.utcnow().isoformat()
            
            prediction_record = {
                'prediction_id': str(uuid.uuid4()),
                'image_file_ids': image_file_ids,
                'prediction_result': prediction_result,
                'confidence': confidence,
                'timestamp': timestamp,
                'accident_detected': prediction_result >= 0.5,
                'image_count': len(image_file_ids),
                'storage_type': 'gridfs'
            }
            
            collection = self.db['predictions']
            result = collection.insert_one(prediction_record)
            
            print(f"Stored prediction record with {len(image_file_ids)} images")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"Error storing prediction record: {e}")
            return None

    def get_prediction_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent prediction history from MongoDB.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of prediction records
        """
        if not self.connected:
            return []
            
        try:
            collection = self.db['predictions']
            predictions = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
            return predictions
        except Exception as e:
            print(f"Error getting prediction history: {e}")
            return []

    def get_prediction_stats(self) -> Dict:
        """
        Get statistics about prediction history.
        
        Returns:
            Dictionary with prediction statistics
        """
        if not self.connected:
            return {'error': 'Database not connected'}
            
        try:
            collection = self.db['predictions']
            
            total_predictions = collection.count_documents({})
            accident_predictions = collection.count_documents({'accident_detected': True})
            no_accident_predictions = collection.count_documents({'accident_detected': False})
            
            # Get average confidence
            pipeline = [
                {'$group': {
                    '_id': None,
                    'avg_confidence': {'$avg': '$confidence'},
                    'avg_prediction': {'$avg': '$prediction_result'}
                }}
            ]
            
            avg_stats = list(collection.aggregate(pipeline))
            avg_confidence = avg_stats[0]['avg_confidence'] if avg_stats else 0
            avg_prediction = avg_stats[0]['avg_prediction'] if avg_stats else 0
            
            return {
                'total_predictions': total_predictions,
                'accident_predictions': accident_predictions,
                'no_accident_predictions': no_accident_predictions,
                'accident_rate': (accident_predictions / total_predictions * 100) if total_predictions > 0 else 0,
                'average_confidence': avg_confidence,
                'average_prediction_score': avg_prediction
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def clear_prediction_data(self) -> Dict:
        """Clear all prediction data including GridFS files with prediction metadata."""
        if not self.is_connected():
            return {'success': False, 'error': 'Database not connected'}
        
        try:
            # Clear predictions collection
            predictions_collection = self.db['predictions']
            prediction_count = predictions_collection.count_documents({})
            predictions_collection.delete_many({})
            
            # Clear GridFS files with prediction metadata
            gridfs_deleted = 0
            if hasattr(self, 'fs') and self.fs:
                # Find and delete files with prediction purpose
                for file_doc in self.fs.find({'metadata.purpose': 'prediction'}):
                    self.fs.delete(file_doc._id)
                    gridfs_deleted += 1
            
            return {
                'success': True,
                'message': f'Cleared {prediction_count} predictions and {gridfs_deleted} GridFS files',
                'predictions_deleted': prediction_count,
                'gridfs_files_deleted': gridfs_deleted
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def clear_training_data(self) -> Dict:
        """Clear all uploaded training data from MongoDB collections and GridFS."""
        if not self.is_connected():
            return {'success': False, 'error': 'Database not connected'}
        
        try:
            total_sequences = 0
            gridfs_deleted = 0
            
            # Clear training collections
            for collection_name in [TRAIN_COLLECTION, TEST_COLLECTION, VAL_COLLECTION]:
                collection = self.db[collection_name]
                count = collection.count_documents({})
                collection.delete_many({})
                total_sequences += count
            
            # Clear GridFS files with training metadata (not prediction files)
            if hasattr(self, 'fs') and self.fs:
                # Find and delete files with training/upload purpose
                for file_doc in self.fs.find({'metadata.purpose': {'$in': ['training', 'upload']}}):
                    self.fs.delete(file_doc._id)
                    gridfs_deleted += 1
            
            return {
                'success': True,
                'message': f'Cleared {total_sequences} training sequences and {gridfs_deleted} GridFS files',
                'sequences_deleted': total_sequences,
                'gridfs_files_deleted': gridfs_deleted
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Singleton instance
db = Database()