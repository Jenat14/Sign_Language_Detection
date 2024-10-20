import os
import numpy as np  
import shutil
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def augment_and_save_images(class_dir, target_count, datagen):
    # Get the current number of images in the class
    current_images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(current_images)
    
    # Calculate how many more images we need
    images_needed = target_count - current_count
    
    print(f"Augmenting {class_dir} by {images_needed} images...")
    
    # Load the images and augment them
    for i in range(images_needed):
        # Pick a random image to augment
        image_path = os.path.join(class_dir, current_images[i % current_count])
        image = Image.open(image_path)
        image = image.resize((224, 224))  # Resize to target size if necessary
        image = np.expand_dims(np.array(image), axis=0)  # Add batch dimension

        # Apply random transformation
        aug_iter = datagen.flow(image, batch_size=1)
        aug_image = next(aug_iter)[0].astype('uint8')

        # Save the augmented image
        aug_image_path = os.path.join(class_dir, f"augmented_{i}.jpg")
        aug_image = Image.fromarray(aug_image)
        aug_image.save(aug_image_path)

# Set up your directories
train_directory = 'data/coco_dataset/valid_classified'

# Find the largest class
# Find the largest class
class_sizes = {}
for subdir, _, files in os.walk(train_directory):
    if subdir != train_directory:
        class_name = os.path.basename(subdir)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Class: {class_name}, Images found: {len(image_files)}")  # Debugging line
        class_sizes[class_name] = len(image_files)

if not class_sizes:
    print("No classes or images were found.")
else:
    # Determine the target size (largest class size)
    max_size = max(class_sizes.values())
    print(f"Largest class size: {max_size}")


# Create an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augment classes with fewer images
for class_name, size in class_sizes.items():
    if size < max_size:
        class_dir = os.path.join(train_directory, class_name)
        augment_and_save_images(class_dir, max_size, datagen)

print("Data augmentation complete.")
