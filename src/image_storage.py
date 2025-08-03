import os
import gridfs
from pymongo import MongoClient
from dotenv import load_dotenv
from PIL import Image
import io
import base64
from typing import List, Dict, Optional
import numpy as np

# Load environment variables
load_dotenv()

class ImageStorage:
    def __init__(self):
        """Initialize MongoDB GridFS for image storage."""
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.database_name = "accident_detection_db"
        
        try:
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client[self.database_name]
            self.fs = gridfs.GridFS(self.db)  # GridFS for large file storage
            self.client.server_info()  # Test connection
            self.connected = True
            print("✅ Connected to MongoDB with GridFS support")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            self.connected = False

    def store_image(self, image_path: str, metadata: Dict = None) -> Optional[str]:
        """
        Store an image file in GridFS.
        
        Args:
            image_path: Path to the image file
            metadata: Additional metadata (sequence_id, label, split, etc.)
        
        Returns:
            GridFS file ID as string, or None if failed
        """
        if not self.connected:
            return None
            
        try:
            with open(image_path, 'rb') as img_file:
                filename = os.path.basename(image_path)
                
                # Prepare metadata
                file_metadata = {
                    'filename': filename,
                    'original_path': image_path,
                    'content_type': 'image/jpeg'
                }
                
                if metadata:
                    file_metadata.update(metadata)
                
                # Store in GridFS
                file_id = self.fs.put(
                    img_file,
                    filename=filename,
                    metadata=file_metadata
                )
                
                return str(file_id)
                
        except Exception as e:
            print(f"Error storing image {image_path}: {e}")
            return None

    def retrieve_image(self, file_id: str) -> Optional[bytes]:
        """
        Retrieve an image from GridFS by file ID.
        
        Args:
            file_id: GridFS file ID
            
        Returns:
            Image data as bytes, or None if not found
        """
        if not self.connected:
            return None
            
        try:
            from bson import ObjectId
            grid_out = self.fs.get(ObjectId(file_id))
            return grid_out.read()
        except Exception as e:
            print(f"Error retrieving image {file_id}: {e}")
            return None

    def store_sequence_with_images(self, sequence_data: Dict) -> Dict:
        """
        Store a complete sequence with all its images in MongoDB.
        
        Args:
            sequence_data: Dictionary with sequence_id, image_paths, label, split
            
        Returns:
            Modified sequence data with GridFS file IDs instead of paths
        """
        if not self.connected:
            return sequence_data
            
        stored_images = []
        
        for img_path in sequence_data['image_paths']:
            metadata = {
                'sequence_id': sequence_data['sequence_id'],
                'label': sequence_data['label'],
                'split': sequence_data['split'],
                'frame_number': len(stored_images) + 1
            }
            
            file_id = self.store_image(img_path, metadata)
            if file_id:
                stored_images.append(file_id)
            else:
                print(f"Failed to store image: {img_path}")
        
        # Return modified sequence data
        return {
            'sequence_id': sequence_data['sequence_id'],
            'image_file_ids': stored_images,  # GridFS file IDs instead of paths
            'label': sequence_data['label'],
            'split': sequence_data['split'],
            'image_count': len(stored_images)
        }

    def load_sequence_images(self, sequence_data: Dict) -> List[np.ndarray]:
        """
        Load sequence images from GridFS and return as numpy arrays.
        
        Args:
            sequence_data: Sequence with image_file_ids
            
        Returns:
            List of image arrays ready for model input
        """
        if not self.connected or 'image_file_ids' not in sequence_data:
            return []
            
        images = []
        
        for file_id in sequence_data['image_file_ids']:
            img_bytes = self.retrieve_image(file_id)
            if img_bytes:
                # Convert bytes to PIL Image, then to numpy array
                img = Image.open(io.BytesIO(img_bytes))
                img = img.resize((224, 224))  # Resize for model
                img_array = np.array(img) / 255.0  # Normalize
                images.append(img_array)
        
        return images

    def migrate_existing_dataset(self, directory: str, max_frames: int = 5) -> List[Dict]:
        """
        Migrate existing file-based dataset to MongoDB with GridFS.
        
        Args:
            directory: Directory containing the dataset
            max_frames: Maximum frames per sequence
            
        Returns:
            List of sequences with GridFS file IDs
        """
        # Use existing database logic to get sequences
        from src.database import db
        sequences = db.load_sequences_from_files(directory, max_frames)
        
        migrated_sequences = []
        
        for seq in sequences:
            print(f"Migrating sequence: {seq['sequence_id']}")
            migrated_seq = self.store_sequence_with_images(seq)
            migrated_sequences.append(migrated_seq)
        
        print(f"✅ Migrated {len(migrated_sequences)} sequences to GridFS")
        return migrated_sequences

    def get_storage_stats(self) -> Dict:
        """Get statistics about stored images."""
        if not self.connected:
            return {}
            
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
            print(f"Error getting stats: {e}")
            return {}

# Singleton instance
image_storage = ImageStorage()
