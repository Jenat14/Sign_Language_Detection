import os
import cv2
import numpy as np
from pycocotools.coco import COCO

# Load COCO dataset
train_path = os.path.join('.', 'data', 'coco_dataset', 'train', '_annotations.coco.json')
coco_train = COCO(train_path)

def preprocess_image(image_path, input_size):
    # Read and resize the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_size, input_size))
    
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    return img

# Example: Preprocess the first image
image_id = coco_train.getImgIds()[0]  # Get the first image ID from the COCO dataset
img_info = coco_train.loadImgs(image_id)[0]  # Load the image metadata

# Construct the full image path (assuming the images are in the 'data/coco_dataset/train' folder)
img_path = os.path.join('.', 'data', 'coco_dataset', 'train', img_info['file_name'])

# Preprocess the image
processed_image = preprocess_image(img_path, 224)  # Resize images to 224x224 for model input
print(processed_image.shape)  # Should print (224, 224, 3)
