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
        sequences = db.load_from_collection(split)
        if sequences:
            print(f"Loaded {len(sequences)} sequences from MongoDB {split} collection")
            return [seq["image_paths"] for seq in sequences], [seq["label"] for seq in sequences]
    return db.load_sequences_from_files(directory, max_frames)

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
                    for img_path in sequence:
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        
                        # Apply augmentation to individual frames if training
                        if is_training:
                            # Reshape to add batch dimension for datagen
                            img_array = img_array[np.newaxis, ...]
                            # Apply augmentation
                            img_array = datagen.flow(img_array, batch_size=1)[0][0]
                        else:
                            # Just normalize for validation/test
                            img_array = img_array / 255.0
                        
                        seq_images.append(img_array)
                    
                    seq_images = np.array(seq_images)
                    batch_sequences.append(seq_images)
                    batch_labels.append(labels[i])
                
                # Convert to proper batch format: (batch_size, max_frames, height, width, channels)
                batch_sequences = np.array(batch_sequences)
                batch_labels = np.array(batch_labels)
                
                yield batch_sequences, batch_labels
    
    train_generator = generator(train_sequences, train_labels, train_datagen, batch_size, is_training=True)
    test_generator = generator(test_sequences, test_labels, val_test_datagen, batch_size, is_training=False)
    val_generator = generator(val_sequences, val_labels, val_test_datagen, batch_size, is_training=False)
    
    steps_per_epoch = len(train_sequences) // batch_size
    validation_steps = len(val_sequences) // batch_size
    test_steps = len(test_sequences) // batch_size
    
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
    if len(seq_images) < max_frames:
        seq_images += [seq_images[-1]] * (max_frames - len(seq_images))
    else:
        seq_images = seq_images[:max_frames] 
    return np.array([seq_images])

def load_sequence_data_with_mongodb(train_dir=None, test_dir=None, val_dir=None, img_size=(224, 224), max_frames=5, batch_size=16, use_mongodb=True):
    """Load and preprocess sequence data for CNN-LSTM model with MongoDB support.
    
    Args:
        train_dir, test_dir, val_dir (str): Paths to dataset directories (optional if using MongoDB).
        img_size (tuple): Image size (height, width).
        max_frames (int): Number of frames per sequence.
        batch_size (int): Batch size for generators.
        use_mongodb (bool): Whether to load from MongoDB GridFS in addition to files.
    
    Returns:
        Tuple of (generator, steps) for train, test, and validation sets.
    """
    from src.database import db
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
                        # Extract images from GridFS to temporary files
                        temp_paths = []
                        for file_id in seq['image_file_ids']:
                            image_data = db.retrieve_image_from_gridfs(file_id)
                            if image_data:
                                # Create temporary file
                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                                temp_file.write(image_data)
                                temp_file.close()
                                temp_paths.append(temp_file.name)
                        
                        if temp_paths:
                            # Ensure we have exactly max_frames
                            if len(temp_paths) < max_frames:
                                temp_paths.extend([temp_paths[-1]] * (max_frames - len(temp_paths)))
                            elif len(temp_paths) > max_frames:
                                temp_paths = temp_paths[:max_frames]
                            
                            all_sequences.append(temp_paths)
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
    
    # Use existing data generators
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
        indices = np.arange(len(sequences))
        while True:
            if is_training:
                np.random.shuffle(indices)
            
            for start in range(0, len(sequences), batch_size):
                end = min(start + batch_size, len(sequences))
                batch_indices = indices[start:end]
                
                batch_sequences = []
                batch_labels = []
                
                for idx in batch_indices:
                    sequence_images = []
                    
                    for img_path in sequences[idx]:
                        try:
                            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                            img_array = tf.keras.preprocessing.image.img_to_array(img)
                            
                            if is_training:
                                img_array = datagen.random_transform(img_array)
                            
                            img_array = datagen.standardize(img_array)
                            sequence_images.append(img_array)
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
                            # Use a black image as fallback
                            img_array = np.zeros((*img_size, 3))
                            sequence_images.append(img_array)
                    
                    batch_sequences.append(sequence_images)
                    batch_labels.append(labels[idx])
                
                yield np.array(batch_sequences), np.array(batch_labels)
    
    # Calculate steps
    train_steps = max(1, len(train_sequences) // batch_size)
    test_steps = max(1, len(test_sequences) // batch_size) if test_sequences else 1
    val_steps = max(1, len(val_sequences) // batch_size) if val_sequences else 1
    
    # Create generators
    train_gen = generator(train_sequences, train_labels, train_datagen, batch_size, True)
    test_gen = generator(test_sequences, test_labels, val_test_datagen, batch_size, False) if test_sequences else None
    val_gen = generator(val_sequences, val_labels, val_test_datagen, batch_size, False) if val_sequences else None
    
    return (train_gen, train_steps), (test_gen, test_steps), (val_gen, val_steps)