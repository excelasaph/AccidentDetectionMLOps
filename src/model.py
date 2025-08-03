import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, LSTM, TimeDistributed, Dropout, BatchNormalization

def create_model(max_frames=5):
    # Define the input layer for sequences
    input_layer = tf.keras.Input(shape=(max_frames, 224, 224, 3))
    
    # Base CNN model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True
   
    for layer in base_model.layers[:10]:
        layer.trainable = False
    
    cnn = TimeDistributed(base_model)(input_layer)
    cnn = TimeDistributed(GlobalAveragePooling2D())(cnn)
    
    # LSTM for temporal features
    # lstm = LSTM(256, return_sequences=False)(cnn)
    lstm = LSTM(256, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(cnn)
    
    # Dense layers with regularization
    x = Dense(512, activation='relu')(lstm)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x) 
    
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)

    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create model with proper input/output
    model = Model(inputs=input_layer, outputs=predictions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Increased from 0.0001 to 0.001
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def train_model(model, train_generator, val_generator, train_steps, val_steps, epochs=150):
    """Train model with enhanced callbacks and monitoring."""
    
    # Enhanced callbacks configuration
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Changed from val_loss to val_accuracy
            patience=25,  # Increased from 20 to 25
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',  # Changed from val_loss to val_accuracy
            factor=0.5,
            patience=8,  # Reduced from 10 to 8 for more aggressive reduction
            min_lr=1e-6,
            verbose=1,
            mode='max'  # Changed to max for accuracy monitoring
        ),
        tf.keras.callbacks.ModelCheckpoint(
            '../models/accident_model.keras',
            monitor='val_accuracy',  # Changed from val_loss to val_accuracy
            save_best_only=True,
            verbose=1,
            mode='max'  # Changed to max for accuracy monitoring
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# def train_model(model, train_generator, val_generator, train_steps, val_steps, epochs=30):
#     early_stopping = tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
#     )
#     lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
#         monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1
#     )
#     checkpoint = tf.keras.callbacks.ModelCheckpoint(
#         '../models/accident_model.keras', save_best_only=True, monitor='val_loss', verbose=1
#     )
    
#     history = model.fit(
#         train_generator,
#         steps_per_epoch=train_steps,
#         epochs=epochs,
#         validation_data=val_generator,
#         validation_steps=val_steps,
#         callbacks=[early_stopping, lr_scheduler, checkpoint],
#         verbose=1
#     )
    
#     return history

def retrain_model(model_path, train_generator, val_generator, train_steps, val_steps, epochs=5):
    model = tf.keras.models.load_model(model_path)
    history = train_model(model, train_generator, val_generator, train_steps, val_steps, epochs)
    model.save('../models/accident_model.keras')
    return history