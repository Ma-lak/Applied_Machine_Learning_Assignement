import os
import numpy as np

def load_data():
  

    DATASET_DIR = "datasets"
    npz_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".npz")]
    if len(npz_files) == 0:
      raise FileNotFoundError("No .npz file found in Datasets folder.")
    npz_path = os.path.join(DATASET_DIR, npz_files[0])
    print(f"Loading dataset from: {npz_path}")
    data = np.load(npz_path)
    print("Keys in npz file:", data.files)

    Xtrain = data["train_images"]
    ytrain = data["train_labels"]

    Xtest = data["test_images"]
    ytest = data["test_labels"]

    Xval  = data["val_images"]
    yval  = data["val_labels"]

    print("Train shape:", Xtrain.shape, ytrain.shape)
    print("Test shape:", Xtest.shape, ytest.shape)

    return Xtrain, ytrain, Xtest, ytest, Xval, yval