from tensorflow import keras
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from data import load_data  # Ensure this file exists and works properly

def build_model():
    # Load the pretrained MobileNetV2 model, without the top (final classification layer)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers to retain the pre-trained weights
    base_model.trainable = False

    # Add custom layers for sign language classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
    x = Dense(512, activation='relu')(x)  # Fully connected layer
    predictions = Dense(26, activation='softmax')(x)  # Output layer for 26 classes

    # Define the complete model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # Load the data (train_generator and val_generator must be returned by load_data)
    train_generator, val_generator = load_data()

    # Build the pretrained MobileNetV2-based model
    model = build_model()

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,  # Adjust based on your performance and hardware
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=val_generator.samples // val_generator.batch_size,
    )

    # Save the trained model to a file
    model.save('asl_pretrained_model.h5')
    print("Model saved as asl_pretrained_model.h5")

if __name__ == "__main__":
    train_model()
