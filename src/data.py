from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

def load_data(batch_size=32, img_size=(224, 224)):
    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize pixel values
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation data (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Load training data from directory
    train_generator = train_datagen.flow_from_directory(
        './data/coco_dataset/train_classified',  # Path to training data
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load validation data from directory
    val_generator = val_datagen.flow_from_directory(
        './data/coco_dataset/valid_classified',  # Path to validation data
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator
