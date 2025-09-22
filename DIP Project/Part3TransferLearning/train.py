import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from utils import load_dataset  
dataset_path = os.path.join("..", "dataset")
model_save_dir = "trained_models"
os.makedirs(model_save_dir, exist_ok=True)

best_model_path = os.path.join(model_save_dir, "MobileNetV2_transfer_best.keras")
final_model_path = os.path.join(model_save_dir, "MobileNetV2_transfer_final.keras")
model_json_path = os.path.join(model_save_dir, "MobileNetV2_transfer_model.json")
history_path = os.path.join(model_save_dir, "MobileNetV2_transfer_history.npy")

if len(tf.config.list_physical_devices('GPU')) > 0:
    print("Using GPU for training")
else:
    print("No GPU detected, using CPU.")

print("Loading dataset...")
images, labels, encoder = load_dataset(dataset_path, as_rgb=True)
print(f"Image shape: {images.shape}, Label shape: {labels.shape}")

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

input_shape = (64, 64, 3)
inputs = Input(shape=input_shape)
x = Resizing(96, 96)(inputs)  
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=x)
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(len(encoder.classes_), activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()


checkpoint = ModelCheckpoint(
    best_model_path, save_best_only=True,
    monitor="val_accuracy", mode="max", verbose=1
)

print("Training model...")
epochs = 30 
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint]
)


print(f"Saving final model to {final_model_path}...")
model.save(final_model_path)

print(f"Saving model architecture to {model_json_path}...")
with open(model_json_path, "w") as f:
    f.write(model.to_json())

print(f"Saving training history to {history_path}...")
np.save(history_path, history.history)

print("All artifacts saved successfully.")
