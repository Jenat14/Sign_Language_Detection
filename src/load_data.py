import os
from pycocotools.coco import COCO

# Define paths for training, validation, and test datasets
train_path = os.path.join('.', 'data', 'coco_dataset', 'train', '_annotations.coco.json')
val_path = os.path.join('.','data', 'coco_dataset', 'valid', '_annotations.coco.json')
test_path = os.path.join('.', 'data', 'coco_dataset', 'test', '_annotations.coco.json')

coco_train = COCO(train_path)
coco_val = COCO(val_path)
coco_test = COCO(test_path)

# Example: Print the number of images and annotations for each dataset
print(f"Training set - Number of images: {len(coco_train.imgs)}, Number of annotations: {len(coco_train.anns)}")
print(f"Validation set - Number of images: {len(coco_val.imgs)}, Number of annotations: {len(coco_val.anns)}")
print(f"Test set - Number of images: {len(coco_test.imgs)}, Number of annotations: {len(coco_test.anns)}")