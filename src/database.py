import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Optional
import glob

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
        """Initialize MongoDB client."""
        try:
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client[DATABASE_NAME]
            self.client.server_info()  # Test connection
            self.connected = True
            print("Connected to MongoDB")
        except ConnectionFailure:
            print("Failed to connect to MongoDB. Using file-based fallback.")
            self.connected = False

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
        """Load sequences from file system, mimicking group_images_by_sequence."""
        image_files = glob.glob(os.path.join(directory, '*/*'))
        sequences = {}
        for file in image_files:
            filename = os.path.basename(file)
            parent_dir = os.path.basename(os.path.dirname(file))
            
            if '_' in filename:
                base_sequence_id = '_'.join(filename.split('_')[:-1])
            else:
                base_sequence_id = filename.split('.')[0]
            
            # Include parent directory to avoid conflicts between Accident/Non-Accident
            sequence_id = f"{parent_dir}_{base_sequence_id}"
            
            if sequence_id not in sequences:
                sequences[sequence_id] = []
            sequences[sequence_id].append(file)

        result = []
        for seq_id, files in sequences.items():
            try:
                files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            except:
                files.sort()
            if len(files) == 0:
                continue
            parent_dir = os.path.basename(os.path.dirname(files[0]))
            label = 1 if parent_dir == 'Accident' else 0
            if len(files) >= 1:
                if len(files) > max_frames:
                    files = files[:max_frames]
                else:
                    while len(files) < max_frames:
                        files.append(files[-1])
                split = "train" if "train" in directory else "test" if "test" in directory else "val"
                
                # Remove parent directory prefix from sequence_id for cleaner storage
                clean_sequence_id = seq_id.split('_', 1)[1] if '_' in seq_id else seq_id
                
                result.append({
                    "sequence_id": clean_sequence_id,
                    "image_paths": files,
                    "label": label,
                    "split": split
                })
        print(f"Loaded {len(result)} sequences from {directory}")
        return result

# Singleton instance
db = Database()