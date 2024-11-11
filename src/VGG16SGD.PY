from tensorflow import keras
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam, SGD
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from data import load_data  # Ensure this file exists and works properly

def build_model(unfreeze=False):
    # Load the VGG16 model with pre-trained weights from ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    if unfreeze:
        # Unfreeze more layers for fine-tuning
        for layer in base_model.layers[-12:]:  # Unfreeze last 12 layers for better fine-tuning
            layer.trainable = True
    else:
        # Freeze the entire base model initially
        for layer in base_model.layers:
            layer.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add another Dropout layer to prevent overfitting
    outputs = Dense(26, activation='softmax')(x)  # 26 classes for sign language alphabet

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model with SGD optimizer for fine-tuning stability
    optimizer = SGD(learning_rate=1e-4, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Train the model
def train_model():
    # Load the data
    train_generator, val_generator = load_data()

    # Build the CNN model with unfreezing enabled
    model = build_model(unfreeze=True)

    # Callbacks
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=1e-7)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,  # Increase epochs to allow better convergence
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=[lr_reduction, early_stopping, tensorboard]
    )

    # Save the fine-tuned model
    model.save('asl_finetuned_vgg16_model_SGD.h5')
    print("Model saved as asl_finetuned_vgg16_model.h5")


if __name__ == "__main__":
    train_model()
