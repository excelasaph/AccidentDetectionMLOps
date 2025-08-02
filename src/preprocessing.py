import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.database import db
import glob
import os
from collections import defaultdict

# def group_images_by_sequence(directory, max_frames):
#     """Group images into sequences based on naming convention (e.g., test10_*).
    
#     Args:
#         directory (str): Path to directory (e.g., 'data/train').
#         max_frames (int): Maximum number of frames per sequence (truncates if > max_frames, pads if < max_frames).
    
#     Returns:
#         sequences (list): List of sequences, each containing up to max_frames image paths.
#         labels (list): Binary labels (1 for Accident, 0 for Non_Accident).
#     """
#     image_files = glob.glob(os.path.join(directory, '*/*'))
#     sequences = defaultdict(list)
    
#     for file in image_files:
#         filename = os.path.basename(file)
        
#         # Handle different filename patterns
#         if '_' in filename:
#             # Standard pattern: test10_33.jpg
#             sequence_id = '_'.join(filename.split('_')[:-1])
#         else:
#             # Handle files like acc1 (1).jpg or other patterns
#             sequence_id = filename.split('.')[0] 
        
#         sequences[sequence_id].append(file)
    
#     for seq in sequences:
#         try:
#             # Try standard pattern first (test10_33.jpg -> 33)
#             sequences[seq].sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
#         except (ValueError, IndexError):
#             try:
#                 # Try alternative pattern (acc1 (2).jpg -> 2)
#                 sequences[seq].sort(key=lambda x: int(os.path.basename(x).split('(')[-1].split(')')[0]))
#             except (ValueError, IndexError):
#                 # If all else fails, sort alphabetically
#                 sequences[seq].sort()
    
#     padded_sequences = []
#     labels = []
    
#     for seq, files in sequences.items():
#         if len(files) == 0:
#             continue
            
#         # LABELING LOGIC
#         parent_dir = os.path.basename(os.path.dirname(files[0]))
#         label = 1 if parent_dir == 'Accident' else 0
        
#         # Only process sequences with at least 1 frame
#         if len(files) >= 1:
#             if len(files) > max_frames:
#                 files = files[:max_frames] 
#             else:
#                 # Pad with last frame if needed
#                 while len(files) < max_frames:
#                     files = files + [files[-1]]
            
#             padded_sequences.append(files)
#             labels.append(label)
    
#     print(f"Found {len(padded_sequences)} sequences in {directory}")
    
#     return padded_sequences, labels
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
        rotation_range=20, 
        width_shift_range=0.1, 
        height_shift_range=0.1,  
        shear_range=0.1,  
        zoom_range=0.1,  
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
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