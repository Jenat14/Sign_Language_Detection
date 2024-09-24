from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from train import load_data  # Ensure this file exists and works properly

def build_model():
    # Simple CNN model
    model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(26, activation='softmax')  # 26 classes
])

    
    # Compile the model with categorical crossentropy for multi-class classification
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # Load the data (train_generator and val_generator must be returned by load_data)
    train_generator, val_generator = load_data()

    # Build the CNN model
    model = build_model()

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,  # Adjust as needed
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator)
    )

    # Save the trained model to a file
    model.save('asl_model.h5')
    print("Model saved as asl_model.h5")

if __name__ == "__main__":
    train_model()
