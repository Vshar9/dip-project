import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def is_valid_image(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) and not filename.startswith(".")

def load_dataset(dataset_path, as_rgb=False):
    images = []
    labels = []

    for class_name in sorted(os.listdir(dataset_path)):
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        image_files = os.listdir(class_folder)
        for img_file in tqdm(image_files, desc=f"Loading {class_name}"):
            if not is_valid_image(img_file):
                continue
            img_path = os.path.join(class_folder, img_file)
            
            if as_rgb:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load image in RGB
            else:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in Grayscale
            
            if img is None:
                continue
            images.append(img)
            labels.append(class_name)

    images = np.array(images)
    labels = np.array(labels)

    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    return images, labels_encoded, encoder
