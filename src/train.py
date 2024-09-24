from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    # Create an instance of ImageDataGenerator for real-time augmentation and validation split
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        './data/coco_dataset/train_classified',  # Adjust this to match your dataset
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        './data/coco_dataset/train_classified',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator

# Only run this if the script is executed directly
if __name__ == "__main__":
    train_generator, val_generator = load_data()
