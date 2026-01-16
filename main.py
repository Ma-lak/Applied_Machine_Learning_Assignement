import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

#from ModelA.acquire_data import load_data
from ModelA.feature_pipeline import feature_pipeline
from ModelA.data_augmentation import augment_img
from ModelA.model_training import train_evaluate_visualise
from ModelB.resnet18_train_evaluate import resnet18_train_evaluate
from ModelB.augment import augment_resnet_images



def main():
    '''
    Load data from NPZ file, perform feature extraction, data augmentation, and execute the training and evaluation pipeline.
    Args:
        None
    Returns:
        None
    '''
    DATASET_DIR = "datasets"
    npz_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".npz")]
    if len(npz_files) == 0:
      raise FileNotFoundError("No .npz file found in Datasets folder.")
    npz_path = os.path.join(DATASET_DIR, npz_files[0])
    print(f"Loading dataset from: {npz_path}")
    data = np.load(npz_path)

    # Print the keys in the npz file to understand its structure.
    print("Keys in npz file:", data.files)

    Xtrain = data["train_images"]
    ytrain = data["train_labels"]

    Xtest = data["test_images"]
    ytest = data["test_labels"]

    Xval  = data["val_images"]
    yval  = data["val_labels"]

    # Print the shapes of the loaded image arrays
    print("Train shape:", Xtrain.shape, ytrain.shape)
    print("Test shape:", Xtest.shape, ytest.shape)
    print("Val shape:", Xval.shape, yval.shape)

    # MODEL A
    # Flatten the images for SVM - keeping raw features
    X_train = Xtrain.reshape(len(Xtrain), -1)  # (N, 784)
    X_test = Xtest.reshape(len(Xtest), -1)  # (N, 784)
    y_train = ytrain.ravel()
    y_test = ytest.ravel()

    # Feature Preprocessing and Extraction including Normalization, HOG method, and PCA
    processed_data = feature_pipeline(Xtrain, Xtest, ytrain, ytest)

    # Data Augmentation
    # Arrays to hold augmented data
    X_aug = []              
    Y_aug = []

    # Loop to augment each image
    for img, label in zip(Xtrain, ytrain):
        augmented_imgs = augment_img(img)
        X_aug.extend(augmented_imgs)
        Y_aug.extend([label] * len(augmented_imgs))

    # Convert to numpy arrays
    X_aug = np.array(X_aug)
    Y_aug = np.array(Y_aug).ravel()  # SVM expects 1D labels

    # Flattening augmented training images to fit within SVM
    X_aug_flat = X_aug.reshape(X_aug.shape[0], -1)


    # Train, evaluate and visualise model performance
    # For raw features
    train_evaluate_visualise(X_train, y_train, X_test, y_test, C = 100, gamma = 1e-06, feature_type="Raw")
    print("Train shape is: ", X_train.shape)
    print("Test shape is: ", X_test.shape)

    # For HOG features
    train_evaluate_visualise(processed_data['X_train_hog'], y_train, processed_data['X_test_hog'], y_test, C = 10, gamma = 0.1, feature_type="HOG")
    print("HOG Train and test shapes are: ", processed_data['X_train_hog'].shape, processed_data['X_test_hog'].shape)

    # For Augmented Features (Flipped, Gaussian Noise Removal and Flattened)
    train_evaluate_visualise(X_aug_flat, Y_aug, X_test, y_test, C = 100, gamma = 1e-06, feature_type="Augmented")
    print("Augmented train shape is:" , X_aug_flat.shape)


    # Model B : Resnet18

    # Train Model with raw features
    # Convert numpy to torch tensors
    # Convert to 4D tensor with 3 channels as Resnet requires 3 channels
    X_train_tensor = torch.tensor(Xtrain, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)
    X_val_tensor   = torch.tensor(Xval, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)
    X_test_tensor  = torch.tensor(Xtest, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)

    # Labels
    y_train_tensor = torch.tensor(ytrain, dtype=torch.long).squeeze()
    y_val_tensor   = torch.tensor(yval, dtype=torch.long).squeeze()
    y_test_tensor  = torch.tensor(ytest, dtype=torch.long).squeeze()
 


    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

    trainloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
    )

    valloader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
    )

    testloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

    resnet18_train_evaluate(trainloader, testloader, valloader)

    # Augment images for ResNet18
    # Training data: augmentation + normalization
    X_train_aug = augment_resnet_images(X_train_tensor, train=True)

    # Validation and test: normalization only
    X_val_norm  = augment_resnet_images(X_val_tensor, train=False)
    X_test_norm = augment_resnet_images(X_test_tensor, train=False)

    train_dataset_aug = TensorDataset(X_train_aug, y_train_tensor)
    val_dataset_aug   = TensorDataset(X_val_norm, y_val_tensor)
    test_dataset_aug  = TensorDataset(X_test_norm, y_test_tensor)

    trainloader_aug = DataLoader(
        train_dataset_aug,
        batch_size=32,
        shuffle=True
    )

    valloader_aug = DataLoader(
        val_dataset_aug,
        batch_size=32,
        shuffle=False
    )

    testloader_aug = DataLoader(
        test_dataset_aug,
        batch_size=32,
        shuffle=False
    )

    resnet18_train_evaluate(trainloader_aug, testloader_aug, valloader_aug)

if __name__ == "__main__":
    main()