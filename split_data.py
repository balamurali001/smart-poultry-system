import os
import shutil
import random
from pathlib import Path

# Set these paths
input_dir = 'dataset'  # Folder with 4 class folders inside
output_dir = 'split_dataset'
train_ratio = 0.8

# Create output dirs
for split in ['train', 'test']:
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(output_dir, split, class_name)
        os.makedirs(class_path, exist_ok=True)

# Go through each class folder
for class_name in os.listdir(input_dir):
    class_dir = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    images = list(Path(class_dir).glob('*.*'))
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    for img_path in train_images:
        shutil.copy(img_path, os.path.join(output_dir, 'train', class_name, img_path.name))
    
    for img_path in test_images:
        shutil.copy(img_path, os.path.join(output_dir, 'test', class_name, img_path.name))

print("✅ Split complete. Train/Test folders ready in 'split_dataset/'")
