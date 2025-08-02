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
   
    for layer in base_model.layers[:30]:
        layer.trainable = False
    
    cnn = TimeDistributed(base_model)(input_layer)
    cnn = TimeDistributed(GlobalAveragePooling2D())(cnn)
    
    # LSTM for temporal features
    lstm = LSTM(128, return_sequences=False)(cnn)
    
    # Dense layers with regularization
    x = Dense(256, activation='relu')(lstm)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x) 
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x) 
    
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
    x = Dropout(0.5)(x)

    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create model with proper input/output
    model = Model(inputs=input_layer, outputs=predictions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_generator, val_generator, train_steps, val_steps, epochs=30):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        '../models/accident_model.keras', save_best_only=True, monitor='val_loss', verbose=1
    )
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=[early_stopping, lr_scheduler, checkpoint],
        verbose=1
    )
    
    return history

def retrain_model(model_path, train_generator, val_generator, train_steps, val_steps, epochs=5):
    model = tf.keras.models.load_model(model_path)
    history = train_model(model, train_generator, val_generator, train_steps, val_steps, epochs)
    model.save('../models/accident_model.keras')
    return history