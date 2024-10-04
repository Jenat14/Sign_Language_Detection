from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        './data/coco_dataset/train_classified',  # Adjust path to your training data
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'  # Set as training data
    )

    val_generator = datagen.flow_from_directory(
        './data/coco_dataset/valid_classified',  # Use the same directory for validation
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'  # Set as validation data
    )

    return train_generator, val_generator
