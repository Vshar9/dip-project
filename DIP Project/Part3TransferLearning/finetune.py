import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import load_dataset 

# --- Paths ---
dataset_path = os.path.join("..", "dataset")
model_save_dir = "trained_models"
os.makedirs(model_save_dir, exist_ok=True)

pretrained_model_path = os.path.join(model_save_dir, "MobileNetV2_transfer_best.keras")
finetuned_model_path = os.path.join(model_save_dir, "MobileNetV2_transfer_finetuned.keras")

if len(tf.config.list_physical_devices('GPU')) > 0:
    print("Using GPU for fine-tuning")
else:
    print("No GPU detected, using CPU.")

print("Loading dataset for fine-tuning...")
images, labels, encoder = load_dataset(dataset_path, as_rgb=True)

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.1, stratify=labels, random_state=42
)
del images, labels 
train_gen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    preprocessing_function=lambda x: x / 255.0
)
val_gen = ImageDataGenerator(
    preprocessing_function=lambda x: x / 255.0
)

train_data = train_gen.flow(X_train, y_train, batch_size=32, shuffle=True)
val_data = val_gen.flow(X_val, y_val, batch_size=32, shuffle=False)

print(f"Loading model from {pretrained_model_path}...")
model = load_model(pretrained_model_path)

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower learning rate
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    finetuned_model_path,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

fine_tune_epochs = 100
print(f"Fine-tuning for {fine_tune_epochs} more epochs...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=fine_tune_epochs,
    callbacks=[checkpoint]
)

print(f"\n✅ Fine-tuning complete. Model saved to: {finetuned_model_path}")
