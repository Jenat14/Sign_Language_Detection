import os
import shutil
from pycocotools.coco import COCO

# Load COCO dataset
train_path = './data/coco_dataset/train/_annotations.coco.json'
coco = COCO(train_path)

# Directory where images are currently stored
images_dir = './data/coco_dataset/train/'

# Directory where we'll store images in subdirectories by class
output_dir = './data/coco_dataset/train_classified/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get category information
categories = coco.loadCats(coco.getCatIds())
category_names = {cat['id']: cat['name'] for cat in categories}

# For each image, move it to the appropriate class directory
for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    img_file = img_info['file_name']

    # Get the category ID for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    if anns:
        cat_id = anns[0]['category_id']
        class_name = category_names[cat_id]

        # Create a directory for this class if it doesn't exist
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Move or copy the image to the appropriate class directory
        shutil.move(os.path.join(images_dir, img_file), os.path.join(class_dir, img_file))

print("Images have been moved to class-specific directories.")
