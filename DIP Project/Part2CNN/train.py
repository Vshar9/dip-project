import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model.tresnet import build_tresnet_cifar10
import os

DATASET_DIR = os.path.join("..","dataset")
BATCH_SIZE = 64
IMG_SIZE = (32,32)
EPOCHS = 50
MODEL_SAVE_PATH = "tresnet_cifar10_full_model.keras"  
WEIGHTS_SAVE_PATH= "tresnet_cifar10_best.h5"

def load_datasets():
    train_ds= tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.1,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size= BATCH_SIZE
    )

    val_ds= tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.1,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size= BATCH_SIZE
    )

    normalization_layer =tf.keras.layers.Rescaling(1.0/255)
    train_ds= train_ds.map(lambda x, y: (normalization_layer(x),y))
    val_ds =val_ds.map(lambda x, y:(normalization_layer(x),y))

    return train_ds, val_ds

def main():
    train_ds, val_ds =load_datasets()
    model =build_tresnet_cifar10()

    model.compile(
        optimizer =Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

    callbacks=[
    ModelCheckpoint(WEIGHTS_SAVE_PATH,save_best_only=True,monitor='val_accuracy',mode='max'),
    EarlyStopping(patience=10, restore_best_weights=True)   
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    test_loss,test_acc = model.evaluate(val_ds)
    print(f"\nValidation accuracy: {test_acc:.4f}")

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved")

if __name__=="__main__":
    main()