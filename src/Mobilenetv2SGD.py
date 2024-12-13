from tensorflow import keras
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import SGD
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard

from data import load_data  

def build_model(unfreeze=False):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    if unfreeze:
        for layer in base_model.layers[-30:]: 
            layer.trainable = True
    else:
        
        for layer in base_model.layers:
            layer.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)  
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  
    outputs = Dense(26, activation='softmax')(x)  

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model with SGD optimizer with momentum
    model.compile(
        optimizer=SGD(learning_rate=1e-4, momentum=0.9), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Train the model
def train_model():
    train_generator, val_generator = load_data()

    # Build the CNN model
    model = build_model()

    # Callbacks
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=1e-7)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,  # Adjust as needed
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=[lr_reduction, early_stopping, tensorboard],  # Add learning rate reduction and early stopping callbacks
    )

    # Save the fine-tuned model
    model.save('asl_finetuned_model.h5')
    print("Model saved as asl_finetuned_model.h5")

if __name__ == "__main__":
    train_model()
