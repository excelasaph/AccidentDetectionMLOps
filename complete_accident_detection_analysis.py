#!/usr/bin/env python3
"""
Complete Accident Detection Analysis - CNN+LSTM Temporal Model
==============================================================
This file contains all the code from the accident_detection.ipynb notebook
with expected outputs and analysis results.

Project: AccidentDetectionMLOps
Author: Excel
Date: August 1, 2025
"""

# =============================================================================
# CELL 1: Import Required Libraries
# =============================================================================
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import glob
import cv2
import os
import sys

sys.path.append('../')
from src.preprocessing import load_sequence_data
from src.model import create_model, train_model

print("✅ All libraries imported successfully")

# =============================================================================
# CELL 2: Ensure Output Directory Exists
# =============================================================================
os.makedirs('../static/images', exist_ok=True)
print("✅ Output directory created/verified")

# =============================================================================
# CELL 3: Display Sample Images (One Sequence)
# =============================================================================
print("📸 Loading sample images for visualization...")

# Load sample accident and non-accident sequences
accident_files = sorted(
    glob.glob('../data/train/Accident/test12_*'), 
    key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])
)[:3]
non_accident_files = sorted(
    glob.glob('../data/train/Non Accident/test26_*'), 
    key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])
)[:3]

images = accident_files + non_accident_files
titles = ['Accident Frame 1', 'Accident Frame 2', 'Accident Frame 3',
          'Non Accident Frame 1', 'Non Accident Frame 2', 'Non Accident Frame 3']

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for i, ax in enumerate(axes.flatten()):
    img = tf.keras.preprocessing.image.load_img(images[i])
    ax.imshow(img)
    ax.set_title(titles[i])
    ax.axis('off')
plt.tight_layout()
plt.savefig('../static/images/sample_images.png')
plt.show()
plt.close()

print("✅ Sample images visualization saved to ../static/images/sample_images.png")

# =============================================================================
# CELL 4: Visualize Augmentation Examples
# =============================================================================
print("🔄 Creating augmentation examples...")

img = tf.keras.preprocessing.image.load_img(accident_files[0])
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = img_array.reshape((1,) + img_array.shape)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30, horizontal_flip=True, brightness_range=[0.8, 1.2]
)
it = datagen.flow(img_array, batch_size=1)

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    batch = next(it)
    plt.imshow(batch[0].astype('uint8'))
    plt.title(f'Augmented Image {i+1}')
    plt.axis('off')
plt.savefig('../static/images/augmentation.png')
plt.show()
plt.close()

print("✅ Augmentation examples saved to ../static/images/augmentation.png")

# =============================================================================
# CELL 5: Test Model Creation Function
# =============================================================================
print("🏗️ Testing model creation...")

test_model = create_model(max_frames=5)
print("Model created successfully!")
print(f"Model input shape: {test_model.input_shape}")
print(f"Model output shape: {test_model.output_shape}")
test_model.summary()

# Expected Output:
# Model input shape: (None, 5, 224, 224, 3)
# Model output shape: (None, 1)
# [Model architecture summary with layer details]

# =============================================================================
# CELL 6: Load and Preprocess Sequence Data
# =============================================================================
print("📊 Loading and preprocessing sequence data...")

(train_generator, train_steps), (test_generator, test_steps), (val_generator, val_steps) = load_sequence_data(
    '../data/train', '../data/test', '../data/val'
)

# Expected Output:
# Found X sequences in ../data/train
#    Accident sequences: Y
#    Non-accident sequences: Z
# Found X sequences in ../data/test
# Found X sequences in ../data/val

# =============================================================================
# CELL 7: Test Generator Output Shapes
# =============================================================================
print("🔍 Testing generator output shapes...")

test_batch_x, test_batch_y = next(train_generator)
print(f"Generator batch X shape: {test_batch_x.shape}")
print(f"Generator batch Y shape: {test_batch_y.shape}")
print(f"Expected model input shape: (None, 5, 224, 224, 3)")
print(f"Actual generator shape: {test_batch_x.shape}")
print(f"Shape match: {'Match' if len(test_batch_x.shape) == 5 and test_batch_x.shape[1:] == (5, 224, 224, 3) else 'No Match'}")

test_model = create_model(max_frames=5)
print(f"Model input shape: {test_model.input_shape}")
print(f"Model expects: {test_model.input_shape}")
print(f"Generator provides: {test_batch_x.shape}")

# Expected Output:
# Generator batch X shape: (16, 5, 224, 224, 3)
# Generator batch Y shape: (16,)
# Shape match: Match

# =============================================================================
# CELL 8: Reload Preprocessing Module
# =============================================================================
print("🔄 Reloading preprocessing module...")

import importlib
import src.preprocessing
importlib.reload(src.preprocessing)
from src.preprocessing import load_sequence_data

print("Reloading sequence data with batch support...")
(train_generator, train_steps), (test_generator, test_steps), (val_generator, val_steps) = load_sequence_data(
    '../data/train', '../data/test', '../data/val', batch_size=8
)

test_batch_x, test_batch_y = next(train_generator)
print(f"Generator batch X shape: {test_batch_x.shape}")
print(f"Generator batch Y shape: {test_batch_y.shape}")
print(f"Expected model input shape: (batch_size, 5, 224, 224, 3)")
print(f"Actual generator shape: {test_batch_x.shape}")
print(f"Shape match: {'Match' if len(test_batch_x.shape) == 5 and test_batch_x.shape[1:] == (5, 224, 224, 3) else 'No Match'}")

print(f"Batch dimension: {test_batch_x.shape[0]}")
print(f"Sequence length: {test_batch_x.shape[1]}")
print(f"Image dimensions: {test_batch_x.shape[2:5]}")

# =============================================================================
# CELL 9: Create and Train Temporal Model
# =============================================================================
print("🚀 Creating temporal CNN+LSTM model...")

model = create_model(max_frames=5)
print("Model Summary:")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# =============================================================================
# CELL 10: Training Temporal Model
# =============================================================================
print("⏳ Training temporal model...")

history = train_model(model, train_generator, val_generator, train_steps, val_steps, epochs=50)

# Expected Output:
# Training progress with epochs
# Early stopping if validation loss doesn't improve
# Model saved to ../models/accident_model.keras

# =============================================================================
# CELL 11: Evaluate the Temporal Model
# =============================================================================
print("📊 Evaluating temporal CNN+LSTM model...")

# Get test predictions
test_batch_x, test_batch_y = next(test_generator)
predictions = model.predict(test_batch_x)
predictions_binary = (predictions > 0.5).astype(int).flatten()

# Calculate metrics
accuracy = accuracy_score(test_batch_y, predictions_binary)
precision = precision_score(test_batch_y, predictions_binary)
recall = recall_score(test_batch_y, predictions_binary)
f1 = f1_score(test_batch_y, predictions_binary)

print(f"Overfitting Check:")
print(f"Training Accuracy: ~{history.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: ~{history.history['val_accuracy'][-1]:.4f}")
print(f"Gap: {abs(history.history['accuracy'][-1] - history.history['val_accuracy'][-1]):.4f}")
print(f"Status: {'Good generalization' if abs(history.history['accuracy'][-1] - history.history['val_accuracy'][-1]) < 0.15 else 'Some overfitting'}")

print(f"Model Architecture:")
print(f"Input: Sequences of {test_batch_x.shape[1]} frames")
print(f"Frame size: {test_batch_x.shape[2:4]}")
print(f"Model type: CNN+LSTM Temporal")

# =============================================================================
# CELL 12: Training History Visualization
# =============================================================================
print("📈 Creating training history visualization...")

plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
plt.title('CNN+LSTM Temporal Model - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
plt.title('CNN+LSTM Temporal Model - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Overfitting gap analysis
plt.subplot(1, 3, 3)
accuracy_gap = [abs(t - v) for t, v in zip(history.history['accuracy'], history.history['val_accuracy'])]
plt.plot(accuracy_gap, label='Train-Val Gap', color='red', linewidth=2)
plt.title('Overfitting Analysis')
plt.xlabel('Epoch')
plt.ylabel('|Train - Val| Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../static/images/cnn+lstm temporal_training_history.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Training Summary:")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Average Overfitting Gap: {np.mean(accuracy_gap):.4f}")
print(f"Model saved as: ../models/accident_model.keras")

# =============================================================================
# CELL 13: Comprehensive Overfitting Analysis
# =============================================================================
print("🔍 Comprehensive Overfitting Analysis...")

# Extract key metrics from training history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Calculate gaps
final_acc_gap = abs(train_acc[-1] - val_acc[-1])
final_loss_gap = abs(train_loss[-1] - val_loss[-1])
max_acc_gap = max([abs(t - v) for t, v in zip(train_acc, val_acc)])
avg_acc_gap = np.mean([abs(t - v) for t, v in zip(train_acc, val_acc)])

print("Overfitting Indicators:")
print(f"   Final Accuracy Gap: {final_acc_gap:.4f}")
print(f"   Maximum Accuracy Gap: {max_acc_gap:.4f}")
print(f"   Average Accuracy Gap: {avg_acc_gap:.4f}")
print(f"   Final Loss Gap: {final_loss_gap:.4f}")

val_loss_std = np.std(val_loss[-5:])  # Last 5 epochs
print(f"   Validation Loss Stability: {'Stable' if val_loss_std < 0.1 else 'Unstable/Oscillating'}")

print("Overfitting Assessment:")
if final_acc_gap > 0.15:
    status = "Severe Overfitting"
    color = "🔴"
elif final_acc_gap > 0.08:
    status = "Moderate Overfitting"
    color = "🟡"
elif final_acc_gap > 0.05:
    status = "Mild Overfitting"
    color = "🟠"
else:
    status = "Good Generalization"
    color = "🟢"

print(f"{color} Status: {status}")

if max(val_acc) < 0.7 and avg_acc_gap < 0.05:
    print("Potential Underfitting Detected:")
    print("   - Validation accuracy is low despite good generalization")

# =============================================================================
# CELL 14: Fix Generator Issue and Evaluate Model
# =============================================================================
print("🔧 Fixing generator issue and evaluating model...")

# Reload data generators to fix StopIteration error
(train_gen_new, train_steps_new), (test_gen_new, test_steps_new), (val_gen_new, val_steps_new) = load_sequence_data(
    '../data/train', '../data/test', '../data/val', batch_size=8
)

print(f"New generators created successfully!")
print(f"Test steps available: {test_steps_new}")

if test_steps_new > 0:
    test_images, test_labels = next(test_gen_new)
    print(f"Successfully got test batch:")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Test labels: {test_labels}")
    
    # Make predictions
    predictions = model.predict(test_images)
    predictions_binary = (predictions > 0.5).astype(int)
    
    print(f"Predictions completed:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Binary predictions: {predictions_binary.flatten()}")
else:
    print("No test data available - test_steps is 0")
    print("This means there are no test sequences in your test directory")

# =============================================================================
# CELL 15: Model Metrics Calculation
# =============================================================================
print("📊 Calculating model metrics...")

# Metrics
accuracy = accuracy_score(test_labels, predictions_binary)
precision = precision_score(test_labels, predictions_binary)
recall = recall_score(test_labels, predictions_binary)
f1 = f1_score(test_labels, predictions_binary)

print("Model Evaluation:")
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))

# Expected Output:
# Model Evaluation:
# Accuracy: 0.xxxx
# Precision: 0.xxxx
# Recall: 0.xxxx
# F1 Score: 0.xxxx

# =============================================================================
# CELL 16: Confusion Matrix Visualization
# =============================================================================
print("📊 Creating comprehensive confusion matrix visualization...")

# Create confusion matrix
cm = confusion_matrix(test_labels, predictions_binary.flatten())

plt.figure(figsize=(15, 5))

# Standard Confusion Matrix
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Accident', 'Accident'], 
            yticklabels=['Non-Accident', 'Accident'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix\n(Raw Counts)', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Normalized Confusion Matrix (by true class)
plt.subplot(1, 3, 2)
cm_normalized = confusion_matrix(test_labels, predictions_binary.flatten(), normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=['Non-Accident', 'Accident'], 
            yticklabels=['Non-Accident', 'Accident'],
            cbar_kws={'label': 'Percentage'})
plt.title('Confusion Matrix\n(Normalized By True Class)', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Detailed Metrics Visualization
plt.subplot(1, 3, 3)
metrics_data = {
    'Accuracy': [accuracy],
    'Precision': [precision], 
    'Recall': [recall],
    'F1-Score': [f1]
}

metrics_names = list(metrics_data.keys())
metrics_values = [v[0] for v in metrics_data.values()]

bars = plt.bar(metrics_names, metrics_values, 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
               alpha=0.8, edgecolor='black', linewidth=1)

for bar, value in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.title('Model Performance Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../static/images/confusion_matrix_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Detailed confusion matrix analysis
tn, fp, fn, tp = cm.ravel()

print(f"Confusion Matrix Breakdown:")
print(f"   True Negatives (TN):  {tn:2d} (Correctly Predicted Non-Accident)")
print(f"   False Positives (FP): {fp:2d} (Incorrectly Predicted Accident)")
print(f"   False Negatives (FN): {fn:2d} (Missed Accidents)")
print(f"   True Positives (TP):  {tp:2d} (Correctly Predicted Accident)")

print(f"Error Analysis:")
if fp > 0:
    print(f"   False Positives: {fp} Cases Where Model Incorrectly Predicted Accidents")
    print(f"   → Could Lead To False Alarms In Real Deployment")
if fn > 0:
    print(f"   False Negatives: {fn} Cases Where Model Missed Real Accidents")
    print(f"   → Missing Real Accidents!")

# Class-wise performance
if tp + fn > 0:  
    accident_recall = tp / (tp + fn)
    print(f"   Accident Detection Rate: {accident_recall:.3f} ({tp}/{tp + fn})")
else:
    print(f"   No Actual Accidents In Test Set")

if tn + fp > 0: 
    non_accident_recall = tn / (tn + fp)
    print(f"   Non-Accident Detection Rate: {non_accident_recall:.3f} ({tn}/{tn + fp})")
else:
    print(f"   No Actual Non-Accidents In Test Set")

print(f"Classification Report:")
class_names = ['Non-Accident', 'Accident']
print(classification_report(test_labels, predictions_binary.flatten(), 
                          target_names=class_names, digits=3))

# =============================================================================
# CELL 17: Class Distribution Visualization
# =============================================================================
print("📊 Creating class distribution visualization...")

class_counts = {
    'Accident': len(glob.glob('../data/train/Accident/*')), 
    'Non Accident': len(glob.glob('../data/train/Non Accident/*'))
}
plt.figure(figsize=(6, 4))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), 
           palette=['#1f77b4', '#ff7f0e'])
plt.title('Class Distribution')
plt.ylabel('Number of Images')
plt.savefig('../static/images/class_distribution.png')
plt.show()
plt.close()

print(f"✅ Class distribution: Accident={class_counts['Accident']}, Non-Accident={class_counts['Non Accident']}")

# =============================================================================
# CELL 18: Image Brightness Distribution
# =============================================================================
print("🌟 Analyzing image brightness distribution...")

def calculate_brightness(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return np.mean(img)

brightness_accident = [calculate_brightness(img) for img in glob.glob('../data/train/Accident/*')[:100]]
brightness_non_accident = [calculate_brightness(img) for img in glob.glob('../data/train/Non Accident/*')[:100]]

plt.figure(figsize=(8, 5))
sns.histplot(brightness_accident, label='Accident', color='#1f77b4', alpha=0.5, bins=30)
sns.histplot(brightness_non_accident, label='Non_Accident', color='#ff7f0e', alpha=0.5, bins=30)
plt.title('Image Brightness Distribution')
plt.xlabel('Mean Pixel Brightness')
plt.legend()
plt.savefig('../static/images/brightness_distribution.png')
plt.show()
plt.close()

print("✅ Brightness distribution analysis saved")

# =============================================================================
# CELL 19: Edge Density Distribution
# =============================================================================
print("🔍 Analyzing edge density distribution...")

def calculate_edge_density(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    return np.sum(edges) / (img.shape[0] * img.shape[1])

edge_accident = [calculate_edge_density(img) for img in glob.glob('../data/train/Accident/*')[:100]]
edge_non_accident = [calculate_edge_density(img) for img in glob.glob('../data/train/Non Accident/*')[:100]]

plt.figure(figsize=(8, 5))
sns.histplot(edge_accident, label='Accident', color='#1f77b4', alpha=0.5, bins=30)
sns.histplot(edge_non_accident, label='Non_Accident', color='#ff7f0e', alpha=0.5, bins=30)
plt.title('Edge Density Distribution')
plt.xlabel('Edge Density')
plt.legend()
plt.savefig('../static/images/edge_density.png')
plt.show()
plt.close()

print("✅ Edge density distribution analysis saved")

# =============================================================================
# CELL 20: Testing Model with Specific Training Sequences
# =============================================================================
print("🧪 Testing model with specific training sequences...")

from src.preprocessing import preprocess_for_prediction

def get_sequence_examples():
    """Get specific accident and non-accident sequences for testing"""
    
    # Find accident sequences
    accident_sequences = {}
    accident_files = glob.glob('../data/train/Accident/*')
    for file in accident_files:
        filename = os.path.basename(file)
        if '_' in filename:
            seq_id = '_'.join(filename.split('_')[:-1])
            if seq_id not in accident_sequences:
                accident_sequences[seq_id] = []
            accident_sequences[seq_id].append(file)
    
    # Find non-accident sequences  
    non_accident_sequences = {}
    non_accident_files = glob.glob('../data/train/Non Accident/*')
    for file in non_accident_files:
        filename = os.path.basename(file)
        if '_' in filename:
            seq_id = '_'.join(filename.split('_')[:-1])
            if seq_id not in non_accident_sequences:
                non_accident_sequences[seq_id] = []
            non_accident_sequences[seq_id].append(file)
    
    # Filter and sort sequences
    valid_accident_seqs = {}
    for seq_id, files in accident_sequences.items():
        if len(files) >= 3:
            try:
                sorted_files = sorted(files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
                valid_accident_seqs[seq_id] = sorted_files
            except:
                continue
    
    valid_non_accident_seqs = {}
    for seq_id, files in non_accident_sequences.items():
        if len(files) >= 3:
            try:
                sorted_files = sorted(files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
                valid_non_accident_seqs[seq_id] = sorted_files
            except:
                continue
    
    return valid_accident_seqs, valid_non_accident_seqs

# Get sequence examples
accident_seqs, non_accident_seqs = get_sequence_examples()

print(f"Found {len(accident_seqs)} accident sequences")
print(f"Found {len(non_accident_seqs)} non-accident sequences")

test_accident_seqs = dict(list(accident_seqs.items())[:2])
test_non_accident_seqs = dict(list(non_accident_seqs.items())[:2])

print(f"\nTesting with:")
for seq_id in test_accident_seqs.keys():
    print(f"  Accident: {seq_id} ({len(test_accident_seqs[seq_id])} frames)")
for seq_id in test_non_accident_seqs.keys():
    print(f"  Non-Accident: {seq_id} ({len(test_non_accident_seqs[seq_id])} frames)")

# =============================================================================
# CELL 21: Test Predictions on Selected Sequences
# =============================================================================
print("🎯 Testing predictions on selected sequences...")

def test_sequence_prediction(sequence_files, seq_id, true_label, max_frames=5):
    """Test model prediction on a specific sequence"""
    
    processed_sequence = preprocess_for_prediction(sequence_files[:max_frames], max_frames=max_frames)
    
    prediction = model.predict(processed_sequence, verbose=0)
    prediction_prob = prediction[0][0]
    predicted_class = "Accident" if prediction_prob > 0.5 else "Non-Accident"
    confidence = prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob)
    
    true_class = "Accident" if true_label == 1 else "Non-Accident"
    correct = (prediction_prob > 0.5) == true_label
    
    print(f"Sequence: {seq_id}")
    print(f"   True Label: {true_class}")
    print(f"   Predicted: {predicted_class}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Raw Score: {prediction_prob:.3f}")
    print(f"   Result: {'Correct' if correct else 'Incorrect'}")
    
    return prediction_prob, correct, sequence_files[:max_frames]

# Test all selected sequences
all_results = []
all_sequences_for_viz = []

print("Accident Sequences:")
for seq_id, files in test_accident_seqs.items():
    prob, correct, viz_files = test_sequence_prediction(files, seq_id, true_label=1)
    all_results.append({
        'seq_id': seq_id,
        'true_label': 'Accident',
        'prediction': prob,
        'correct': correct,
        'files': viz_files
    })
    all_sequences_for_viz.extend(viz_files[:3]) 

print("Non-Accident Sequences:")
for seq_id, files in test_non_accident_seqs.items():
    prob, correct, viz_files = test_sequence_prediction(files, seq_id, true_label=0)
    all_results.append({
        'seq_id': seq_id,
        'true_label': 'Non-Accident',
        'prediction': prob,
        'correct': correct,
        'files': viz_files
    })
    all_sequences_for_viz.extend(viz_files[:3])  

# Calculate accuracy on test sequences
correct_predictions = sum([r['correct'] for r in all_results])
total_predictions = len(all_results)
test_accuracy = correct_predictions / total_predictions

print(f"Summary:")
print(f"   Sequences Tested: {total_predictions}")
print(f"   Correct Predictions: {correct_predictions}")
print(f"   Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")

# =============================================================================
# CELL 22: Visualize Tested Sequences with Predictions
# =============================================================================
print("🖼️ Visualizing tested sequences with predictions...")

fig, axes = plt.subplots(len(all_results), 3, figsize=(15, 4*len(all_results)))
if len(all_results) == 1:
    axes = axes.reshape(1, -1)

for idx, result in enumerate(all_results):
    sequence_files = result['files'][:3]
    
    for frame_idx, img_path in enumerate(sequence_files):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        
        ax = axes[idx, frame_idx]
        ax.imshow(img)
        ax.axis('off')
        
        if frame_idx == 1: 
            status_emoji = "✅" if result['correct'] else "❌"
            title = f"{status_emoji} {result['seq_id']}\nTrue: {result['true_label']}\nPred: {result['prediction']:.3f}"
            ax.set_title(title, fontsize=10, fontweight='bold')
        else:
            ax.set_title(f"Frame {frame_idx + 1}", fontsize=9)

plt.tight_layout()
plt.savefig('../static/images/sequence_predictions_test.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Detailed prediction analysis
for result in all_results:
    prediction_prob = result['prediction']
    true_label = result['true_label']
    
    print(f"{result['seq_id']} ({true_label}):")
    print(f"   Raw Prediction Score: {prediction_prob:.4f}")
    print(f"   Decision Threshold: 0.5")
    print(f"   Predicted Class: {'Accident' if prediction_prob > 0.5 else 'Non-Accident'}")
    
    # Confidence analysis
    if prediction_prob > 0.5:
        confidence = prediction_prob
        certainty_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
    else:
        confidence = 1 - prediction_prob
        certainty_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
    
    print(f"   Confidence: {confidence:.3f} ({certainty_level})")
    
    # Error analysis for incorrect predictions
    if not result['correct']:
        if true_label == "Accident" and prediction_prob < 0.5:
            print(f"False Negative: Model missed an accident (score: {prediction_prob:.3f})")
        elif true_label == "Non-Accident" and prediction_prob > 0.5:
            print(f"False Positive: Model incorrectly detected accident (score: {prediction_prob:.3f})")

# =============================================================================
# CELL 23: Investigate Model Bias
# =============================================================================
print("🔍 Investigating model bias...")

def quick_test_multiple_sequences(num_test=10):
    """Test model with multiple random sequences to check for bias"""
    
    accident_seqs, non_accident_seqs = get_sequence_examples()
    
    sample_accident = dict(list(accident_seqs.items())[:num_test//2])
    sample_non_accident = dict(list(non_accident_seqs.items())[:num_test//2])
    
    results = []
    
    print(f"Testing {len(sample_accident)} accident and {len(sample_non_accident)} non-accident sequences...")
    
    # Test accident sequences
    for seq_id, files in sample_accident.items():
        processed_seq = preprocess_for_prediction(files[:5], max_frames=5)
        prediction = model.predict(processed_seq, verbose=0)[0][0]
        results.append({
            'seq_id': seq_id,
            'true_class': 'Accident',
            'prediction': prediction,
            'predicted_class': 'Accident' if prediction > 0.5 else 'Non-Accident'
        })
    
    # Test non-accident sequences
    for seq_id, files in sample_non_accident.items():
        processed_seq = preprocess_for_prediction(files[:5], max_frames=5)
        prediction = model.predict(processed_seq, verbose=0)[0][0]
        results.append({
            'seq_id': seq_id,
            'true_class': 'Non-Accident',
            'prediction': prediction,
            'predicted_class': 'Accident' if prediction > 0.5 else 'Non-Accident'
        })
    
    return results

# Run bias test
bias_test_results = quick_test_multiple_sequences(num_test=10)

# Analyze results
accident_predictions = [r['prediction'] for r in bias_test_results]
true_accidents = [r for r in bias_test_results if r['true_class'] == 'Accident']
true_non_accidents = [r for r in bias_test_results if r['true_class'] == 'Non-Accident']

print(f"Bias Analysis Results:")
print(f"   Total sequences tested: {len(bias_test_results)}")
print(f"   Average prediction score: {np.mean(accident_predictions):.4f}")
print(f"   Min prediction: {np.min(accident_predictions):.4f}")
print(f"   Max prediction: {np.max(accident_predictions):.4f}")
print(f"   Standard deviation: {np.std(accident_predictions):.4f}")

# Check if model always predicts accident
always_accident = all(p > 0.5 for p in accident_predictions)
print(f"Model Always Predicts Accident: {'YES' if always_accident else 'NO'}")

accident_correct = sum(1 for r in true_accidents if r['prediction'] > 0.5)
non_accident_correct = sum(1 for r in true_non_accidents if r['prediction'] <= 0.5)

print(f"   Accident sequences correctly identified: {accident_correct}/{len(true_accidents)}")
print(f"   Non-accident sequences correctly identified: {non_accident_correct}/{len(true_non_accidents)}")
print(f"   Overall accuracy: {(accident_correct + non_accident_correct)/len(bias_test_results):.3f}")

if always_accident:
    print(f"Critical Issue Confirmed:")
    print(f"   The model has learned to ALWAYS predict 'Accident'")

# =============================================================================
# CELL 24: Data Distribution Analysis
# =============================================================================
print("📊 Final data distribution analysis...")

# Class distribution
accident_count = len(glob.glob('../data/train/Accident/*'))
non_accident_count = len(glob.glob('../data/train/Non Accident/*'))
total_count = accident_count + non_accident_count

print(f"   Accident samples: {accident_count}")
print(f"   Non-accident samples: {non_accident_count}")
print(f"   Total samples: {total_count}")
print(f"   Class ratio: {accident_count/non_accident_count:.2f}:1 (Accident:Non-Accident)")

imbalance_ratio = max(accident_count, non_accident_count) / min(accident_count, non_accident_count)
print(f"   Imbalance ratio: {imbalance_ratio:.2f}")
if imbalance_ratio > 3:
    print(f"Severe Class Imbalance Detected!")
else:
    print(f"Class Distribution Is Acceptable")

# Sequence distribution
print(f"Sequence Distribution:")
print(f"   Accident sequences found: {len(accident_seqs)}")
print(f"   Non-accident sequences found: {len(non_accident_seqs)}")
sequence_ratio = len(accident_seqs) / len(non_accident_seqs) if len(non_accident_seqs) > 0 else float('inf')
print(f"   Sequence ratio: {sequence_ratio:.2f}:1")

# Training generator labels
print(f"Training Generator Analysis:")
label_samples = []
for i in range(5): 
    batch_x, batch_y = next(train_generator)
    label_samples.extend(batch_y.tolist())

accident_labels = sum(label_samples)
non_accident_labels = len(label_samples) - accident_labels
print(f"   Sample batches (5 batches = {len(label_samples)} samples):")
print(f"   Accident labels: {accident_labels}")
print(f"   Non-accident labels: {non_accident_labels}")
if len(label_samples) > 0:
    print(f"   Label ratio in batches: {accident_labels/len(label_samples):.2f} accident rate")

unique_labels = set(label_samples)
print(f"   Unique labels in sample: {unique_labels}")
if len(unique_labels) == 1:
    print(f"   All Labels Are The Same! ({list(unique_labels)[0]})")
    print(f"   This Explains Why The Model Always Predicts One Class!")

# =============================================================================
# SUMMARY OF RESULTS
# =============================================================================
print("\n" + "="*80)
print("🎯 FINAL ANALYSIS SUMMARY")
print("="*80)

print(f"📊 MODEL PERFORMANCE:")
print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"   Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"   Overfitting Gap: {abs(history.history['accuracy'][-1] - history.history['val_accuracy'][-1]):.4f}")
print(f"   Test Accuracy: {accuracy:.4f}")
print(f"   Test Precision: {precision:.4f}")
print(f"   Test Recall: {recall:.4f}")
print(f"   Test F1-Score: {f1:.4f}")

print(f"\n🔍 CONFUSION MATRIX:")
print(f"   True Positives: {tp}")
print(f"   True Negatives: {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")

print(f"\n📈 DATA INSIGHTS:")
print(f"   Total Training Images: {accident_count + non_accident_count}")
print(f"   Class Balance: {accident_count}:{non_accident_count}")
print(f"   Sequence Count: {len(accident_seqs) + len(non_accident_seqs)}")

print(f"\n🚨 CRITICAL FINDINGS:")
if always_accident:
    print(f"   ⚠️  Model shows severe bias - always predicts accident")
    print(f"   ⚠️  This indicates a fundamental training issue")
else:
    print(f"   ✅ Model can distinguish between classes")

if final_acc_gap > 0.15:
    print(f"   ⚠️  Severe overfitting detected (gap: {final_acc_gap:.4f})")
else:
    print(f"   ✅ Overfitting is within acceptable range")

print(f"\n💾 SAVED ARTIFACTS:")
print(f"   Model: ../models/accident_model.keras")
print(f"   Training History: ../static/images/cnn+lstm temporal_training_history.png")
print(f"   Confusion Matrix: ../static/images/confusion_matrix_analysis.png")
print(f"   Sample Images: ../static/images/sample_images.png")
print(f"   Class Distribution: ../static/images/class_distribution.png")
print(f"   Brightness Analysis: ../static/images/brightness_distribution.png")
print(f"   Edge Analysis: ../static/images/edge_density.png")
print(f"   Sequence Predictions: ../static/images/sequence_predictions_test.png")

print("="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
