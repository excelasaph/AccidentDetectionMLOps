"""
Examples of using MongoDB GridFS for image storage in the accident detection MLOps pipeline.
"""

import sys
import os
# Add parent directory to Python path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import db
from src.image_storage import ImageStorage

def example_1_basic_image_storage():
    """Example 1: Store and retrieve a single image"""
    print("=== Example 1: Basic Image Storage ===")
    
    # Store an image (using the new renamed format)
    image_path = "data/test/Accident/accident_test_1_1.jpg"
    if os.path.exists(image_path):
        file_id = db.store_image_in_gridfs(image_path)
        print(f"Stored image with ID: {file_id}")
        
        # Retrieve the image
        image_data = db.retrieve_image_from_gridfs(file_id)
        print(f"Retrieved image data: {len(image_data)} bytes")
        
        # Save retrieved image to disk
        with open("retrieved_image.jpg", "wb") as f:
            f.write(image_data)
        print("Saved retrieved image as 'retrieved_image.jpg'")
    else:
        print(f"Image not found: {image_path}")

def example_2_sequence_storage():
    """Example 2: Store sequences with images in GridFS"""
    print("\n=== Example 2: Sequence Storage ===")
    
    # Load sequences from test directory
    sequences = db.load_sequences_from_files("data/test", max_frames=5)
    print(f"Loaded {len(sequences)} sequences")
    
    # Store sequences with images in GridFS
    db.save_sequences_with_images(sequences, "test_gridfs", store_images=True)
    print("Stored sequences with images in MongoDB")
    
    # Load sequences back from GridFS
    loaded_sequences = db.load_sequences_with_images("test_gridfs")
    print(f"Loaded {len(loaded_sequences)} sequences from GridFS")
    
    for i, seq in enumerate(loaded_sequences):
        # Use the correct key structure from GridFS storage
        if 'image_count' in seq:
            images_count = seq['image_count']
        elif 'image_file_ids' in seq:
            images_count = len(seq['image_file_ids'])
        elif 'images' in seq:
            images_count = len(seq['images'])
        elif 'image_paths' in seq:
            images_count = len(seq['image_paths'])
        else:
            images_count = "unknown"
        
        label_text = "Accident" if seq.get('label', 0) == 1 else "Non-Accident"
        print(f"Sequence {i}: {images_count} images, label: {label_text}")
        
        # Only show first few for brevity
        if i >= 4:
            print(f"... and {len(loaded_sequences) - 5} more sequences")
            break

def example_3_migration():
    """Example 3: Migrate existing dataset to GridFS"""
    print("\n=== Example 3: Dataset Migration ===")
    
    # Using ImageStorage class for migration
    image_storage = ImageStorage()
    
    # Get storage stats before migration
    stats_before = image_storage.get_storage_stats()
    print(f"GridFS files before: {stats_before.get('total_files', 0)}")
    
    # Migrate the entire test dataset
    migrated_sequences = image_storage.migrate_existing_dataset("data/test", max_frames=5)
    print(f"Migrated {len(migrated_sequences)} sequences")
    
    # Get storage stats after migration
    stats_after = image_storage.get_storage_stats()
    print(f"GridFS files after: {stats_after.get('total_files', 0)}")

def example_4_hybrid_loading():
    """Example 4: Load from both files and GridFS"""
    print("\n=== Example 4: Hybrid Loading ===")
    
    # Load from files (traditional way)
    file_sequences = db.load_sequences_from_files("data/test")
    print(f"Loaded {len(file_sequences)} sequences from files")
    
    # Load from GridFS (new way)
    try:
        gridfs_sequences = db.load_sequences_with_images("test")
        print(f"Loaded {len(gridfs_sequences)} sequences from GridFS")
    except Exception as e:
        print(f"GridFS loading failed (maybe no data stored yet): {e}")

def example_5_storage_statistics():
    """Example 5: Get storage statistics"""
    print("\n=== Example 5: Storage Statistics ===")
    
    stats = db.get_gridfs_stats()
    print("GridFS Storage Statistics:")
    print(f"  Total files: {stats.get('total_files', 0)}")
    print(f"  Total size: {stats.get('total_size_mb', 0):.2f} MB")
    print(f"  Average file size: {stats.get('avg_file_size_kb', 0):.2f} KB")
    
    # Collection statistics
    for collection in ['train', 'test', 'val']:
        count = db.get_collection_stats(collection)
        print(f"  {collection} collection: {count} documents")

if __name__ == "__main__":
    # Make sure MongoDB is connected
    if not db.is_connected():
        print("MongoDB is not connected. Please check your connection.")
        exit(1)
    
    print("Starting image storage examples...")
    
    # Run examples
    example_1_basic_image_storage()
    example_2_sequence_storage()
    example_3_migration()
    example_4_hybrid_loading()
    example_5_storage_statistics()
    
    print("\n=== All examples completed! ===")
