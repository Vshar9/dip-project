import os
import sys
import cv2
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Fix import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'features')))
from features.utils import load_dataset
from features.glcm_features import extract_features_from_dataset

def train_and_save_models_glcm():
    print("Loading dataset...")
    images, labels, encoder = load_dataset("../dataset")

    # Convert images to grayscale
    
    print("Extracting GLCM features...")
    features = extract_features_from_dataset(images)

    # Check shapes of features and labels
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Split the dataset into training and test sets (90/10 split)
    print("Splitting dataset (90/10)...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, stratify=labels, random_state=42
    )

    # Ensure the splits have the correct shapes
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # Define models to train
    models = {
        "random_forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42)),
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))
    }

    # Create directory to save models
    os.makedirs("trained_models", exist_ok=True)

    # Loop over models and train each one
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Save the trained model
        model_path = f"trained_models/{name}_glcm.joblib"
        print(f"Saving model to {model_path}")
        joblib.dump(model, model_path)

        # Evaluate the model on the test set
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=encoder.classes_))

if __name__ == "__main__":
    train_and_save_models_glcm()
