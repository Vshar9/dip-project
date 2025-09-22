import pickle
import numpy as np
import os
import cv2

PATH = "./cifar-10-batches-py"
OUT_DIR = "dataset"
IMG_SIZE = (64,64)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convert_and_save(data_batch, labels, label_names, prefix):
    for idx, (img_flat, label) in enumerate(zip(data_batch, labels)):
        img = np.reshape(img_flat,(3,32,32)).transpose(1,2,0)
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), IMG_SIZE)
        class_name = label_names[label]
        class_dir = os.path.join(OUT_DIR,class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_filename = f"{prefix}_{idx}.png"
        img_path = os.path.join(class_dir, img_filename)
        cv2.imwrite(img_path, gray)

def main():
    meta = unpickle(os.path.join(PATH,"batches.meta"))
    label_names = [label.decode('utf-8') for label in meta[b'label_names']]
    for i in range(1,6):
        batch = unpickle(os.path.join(PATH,f"data_batch_{i}"))
        convert_and_save(batch[b'data'],batch[b'labels'],label_names, prefix=f"train_{i}")
    test = unpickle(os.path.join(PATH,"test_batch"))
    convert_and_save(test[b'data'],test[b'labels'],label_names, prefix="test")
    print("Success")

if __name__ == "__main__":
    main()

        