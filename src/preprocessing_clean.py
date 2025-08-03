import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.database import db
import glob
import os
from collections import defaultdict

def group_images_by_sequence(directory, max_frames):
    """Load sequences from MongoDB or file system."""
    if db.is_connected():
        split = "train" if "train" in directory else "test" if "test" in directory else "val"
        sequences = db.load_sequences_with_images(split)
        if sequences:
            print(f"Loaded {len(sequences)} sequences from MongoDB {split} collection")
            
            # Extract sequences and labels from MongoDB data
            sequence_data = []
            labels = []
            
            for seq in sequences:
                if seq.get('storage_type') == 'gridfs' and 'image_file_ids' in seq:
                    # For GridFS stored images, we'll handle them later in the processing pipeline
                    sequence_data.append(seq['image_file_ids'])
                    labels.append(seq['label'])
                elif 'image_paths' in seq and seq.get('images_available', True):
                    # For file-based images
                    sequence_data.append(seq['image_paths'])
                    labels.append(seq['label'])
            
            return sequence_data, labels
    
    # Load from file system
    file_sequences = db.load_sequences_from_files(directory, max_frames)
    if file_sequences:
        sequence_data = [seq["image_paths"] for seq in file_sequences]
        labels = [seq["label"] for seq in file_sequences]
        return sequence_data, labels
    
    return [], []

def load_sequence_data(train_dir, test_dir, val_dir, img_size=(224, 224), max_frames=5, batch_size=16):
    """Load and preprocess sequence data for CNN-LSTM model.
    
    Args:
        train_dir, test_dir, val_dir (str): Paths to dataset directories.
        img_size (tuple): Image size (height, width).
        max_frames (int): Number of frames per sequence.
        batch_size (int): Batch size for generators.
    
    Returns:
        Tuple of (generator, steps) for train, test, and validation sets.
    """
    train_sequences, train_labels = group_images_by_sequence(train_dir, max_frames)
    test_sequences, test_labels = group_images_by_sequence(test_dir, max_frames)
    val_sequences, val_labels = group_images_by_sequence(val_dir, max_frames)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,        
        width_shift_range=0.15,   
        height_shift_range=0.15,  
        shear_range=0.15,         
        zoom_range=0.15,          
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  
        channel_shift_range=0.1,       
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    def generator(sequences, labels, datagen, batch_size, is_training=True):
        while True:
            indices = np.arange(len(sequences))
            if is_training:
                np.random.shuffle(indices)
            
            # Process in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_sequences = []
                batch_labels = []
                
                for i in batch_indices:
                    sequence = sequences[i]
                    seq_images = []
                    
                    # Ensure we have exactly max_frames images
                    processed_sequence = list(sequence)
                    
                    # If we have fewer frames than max_frames, duplicate the last frame
                    while len(processed_sequence) < max_frames:
                        if processed_sequence:
                            processed_sequence.append(processed_sequence[-1])
                        else:
                            # If no frames at all, create a dummy frame
                            processed_sequence.append(None)
                    
                    # If we have more frames than max_frames, take the first max_frames
                    if len(processed_sequence) > max_frames:
                        processed_sequence = processed_sequence[:max_frames]
                    
                    for item in processed_sequence:
                        if item is None:
                            # Create a black image
                            img_array = np.zeros((*img_size, 3))
                        elif isinstance(item, str) and (item.startswith('/') or item.startswith('\\') or '.' in item):
                            # File path
                            try:
                                if os.path.exists(item):
                                    img = tf.keras.preprocessing.image.load_img(item, target_size=img_size)
                                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                                else:
                                    # Create a black image if file doesn't exist
                                    img_array = np.zeros((*img_size, 3))
                            except Exception as e:
                                print(f"Error loading image {item}: {e}")
                                img_array = np.zeros((*img_size, 3))
                        else:
                            # GridFS file ID
                            try:
                                image_data = db.retrieve_image_from_gridfs(item)
                                if image_data:
                                    # Save to temporary file and load
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                                        temp_file.write(image_data)
                                        temp_file.flush()
                                        img = tf.keras.preprocessing.image.load_img(temp_file.name, target_size=img_size)
                                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                                    # Clean up temp file
                                    os.unlink(temp_file.name)
                                else:
                                    # Create a black image if GridFS retrieval fails
                                    img_array = np.zeros((*img_size, 3))
                            except Exception as e:
                                print(f"Error loading image from GridFS: {e}")
                                img_array = np.zeros((*img_size, 3))
                        
                        # Apply augmentation to individual frames if training
                        if is_training:
                            # Reshape to add batch dimension for datagen
                            img_array = img_array[np.newaxis, ...]
                            # Apply augmentation
                            try:
                                img_array = datagen.flow(img_array, batch_size=1)[0][0]
                            except:
                                img_array = img_array[0] / 255.0
                        else:
                            # Just normalize for validation/test
                            img_array = img_array / 255.0
                        
                        seq_images.append(img_array)
                    
                    # Ensure seq_images has exactly max_frames elements with consistent shape
                    if len(seq_images) != max_frames:
                        print(f"Warning: sequence has {len(seq_images)} frames, expected {max_frames}")
                        # Pad or truncate to max_frames
                        while len(seq_images) < max_frames:
                            seq_images.append(np.zeros((*img_size, 3)))
                        seq_images = seq_images[:max_frames]
                    
                    # Convert to numpy array and ensure consistent shape
                    seq_images = np.array(seq_images)  # Shape: (max_frames, height, width, channels)
                    batch_sequences.append(seq_images)
                    batch_labels.append(labels[i])
                
                # Convert to proper batch format: (batch_size, max_frames, height, width, channels)
                if batch_sequences:
                    batch_sequences = np.array(batch_sequences)
                    batch_labels = np.array(batch_labels)
                    yield batch_sequences, batch_labels
    
    train_generator = generator(train_sequences, train_labels, train_datagen, batch_size, is_training=True)
    test_generator = generator(test_sequences, test_labels, val_test_datagen, batch_size, is_training=False)
    val_generator = generator(val_sequences, val_labels, val_test_datagen, batch_size, is_training=False)
    
    steps_per_epoch = max(1, len(train_sequences) // batch_size) if train_sequences else 1
    validation_steps = max(1, len(val_sequences) // batch_size) if val_sequences else 1
    test_steps = max(1, len(test_sequences) // batch_size) if test_sequences else 1
    
    return (train_generator, steps_per_epoch), (test_generator, test_steps), (val_generator, validation_steps)

def preprocess_for_prediction(image_paths, img_size=(224, 224), max_frames=5):
    """Preprocess a sequence of images for prediction.
    
    Args:
        image_paths (list): List of image file paths.
        img_size (tuple): Image size (height, width).
        max_frames (int): Maximum number of frames to process.
    
    Returns:
        np.array: Preprocessed sequence of shape (1, max_frames, height, width, 3).
    """
    seq_images = []
    for img_path in image_paths:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        seq_images.append(img_array)
    
    # Ensure exactly max_frames
    if len(seq_images) < max_frames:
        seq_images.extend([seq_images[-1]] * (max_frames - len(seq_images)))
    elif len(seq_images) > max_frames:
        seq_images = seq_images[:max_frames]
    
    return np.array([seq_images])

def load_sequence_data_with_mongodb(train_dir=None, test_dir=None, val_dir=None, img_size=(224, 224), max_frames=5, batch_size=16, use_mongodb=True):
    """Enhanced version that combines file-based and MongoDB data for training."""
    import tempfile
    
    # Combine file-based and MongoDB data
    def load_combined_sequences(file_dir, collection_name):
        all_sequences = []
        all_labels = []
        
        # Load from files if directory provided
        if file_dir and os.path.exists(file_dir):
            file_sequences, file_labels = group_images_by_sequence(file_dir, max_frames)
            all_sequences.extend(file_sequences)
            all_labels.extend(file_labels)
        
        # Load from MongoDB if connected and use_mongodb is True
        if use_mongodb and db.is_connected():
            try:
                mongo_sequences = db.load_sequences_with_images(collection_name)
                
                for seq in mongo_sequences:
                    if seq.get('storage_type') == 'gridfs' and 'image_file_ids' in seq:
                        # We'll keep the GridFS IDs and handle them in the generator
                        file_ids = seq['image_file_ids']
                        
                        # Ensure we have exactly max_frames
                        if len(file_ids) < max_frames:
                            file_ids.extend([file_ids[-1]] * (max_frames - len(file_ids)))
                        elif len(file_ids) > max_frames:
                            file_ids = file_ids[:max_frames]
                        
                        all_sequences.append(file_ids)
                        all_labels.append(seq['label'])
                    
                    elif 'image_paths' in seq and seq.get('images_available', True):
                        # Use existing file paths
                        paths = seq['image_paths']
                        if len(paths) < max_frames:
                            paths.extend([paths[-1]] * (max_frames - len(paths)))
                        elif len(paths) > max_frames:
                            paths = paths[:max_frames]
                        
                        all_sequences.append(paths)
                        all_labels.append(seq['label'])
                
                print(f"Loaded {len(mongo_sequences)} sequences from MongoDB collection '{collection_name}'")
            except Exception as e:
                print(f"Error loading from MongoDB: {e}")
        
        return all_sequences, all_labels
    
    # Load combined data
    train_sequences, train_labels = load_combined_sequences(train_dir, "train")
    test_sequences, test_labels = load_combined_sequences(test_dir, "test") 
    val_sequences, val_labels = load_combined_sequences(val_dir, "val")
    
    print(f"Total training sequences: {len(train_sequences)}")
    print(f"Total test sequences: {len(test_sequences)}")
    print(f"Total validation sequences: {len(val_sequences)}")
    
    # Use the same generator as load_sequence_data
    return load_sequence_data(None, None, None, img_size, max_frames, batch_size)
