from tensorflow import keras
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import os

def load_data():
    # Use data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load training and validation data
    train_generator = train_datagen.flow_from_directory(
        './data/coco_dataset/train_classified',  # Replace with your train data directory
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_directory(
        './data/coco_dataset/valid_classified',  # Replace with your validation data directory
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical"
    )

    return train_generator, val_generator

def build_model(unfreeze_layers=50):
    # Load MobileNetV2 with pre-trained weights
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Unfreeze specified number of layers
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    # Freeze the rest of the layers
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)  # Adjusted dropout
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)  # Additional dropout layer
    outputs = Dense(26, activation='softmax')(x)  # 26 classes for ASL alphabet

    model = Model(inputs=base_model.input, outputs=outputs)

    # Define an Exponential Decay learning rate schedule
    initial_learning_rate = 1e-4
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )

    # Compile model with AdamW optimizer and learning rate schedule
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model():
    # Load data
    train_generator, val_generator = load_data()

    # Build the CNN model
    model = build_model(unfreeze_layers=50)

    # Callbacks: removed ReduceLROnPlateau since we're using ExponentialDecay
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir='./logs_optimized', histogram_freq=1)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,  # Increase as necessary
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=[early_stopping, tensorboard]
    )

    # Save the fine-tuned model
    model.save('asl_optimized_model.h5')
    print("Model saved as asl_optimized_model.h5")


if __name__ == "__main__":
    train_model()
