import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from ModelA.acquire_data import load_data
from ModelA.feature_pipeline import feature_pipeline
from ModelA.data_augmentation import augment_img
from ModelA.model_training import train_evaluate_visualise
from ModelB.train import train_evaluate_resnet18
from ModelB.augment import augment_resnet_images

def main():

    Xtrain, ytrain, Xtest, ytest, Xval, yval = load_data()

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

    # Flattening training and testing images to fit within SVM
    X_aug_flat = X_aug.reshape(X_aug.shape[0], -1)
    X_test = Xtest.reshape(Xtest.shape[0], -1)
    y_test = ytest.ravel()

    # Train, evaluate and visualise model performance
    # For raw features
    train_evaluate_visualise(X_train, y_train, X_test, y_test, gamma = 1e-06, feature_type="Raw")

    # For PCA features
    train_evaluate_visualise(processed_data['X_train_pca'], y_train, processed_data['X_test_pca'], y_test, gamma = 1e-06, feature_type="PCA")

    # For PCA + Normalization features
    train_evaluate_visualise(processed_data['X_train_pca_normalized'], y_train, processed_data['X_test_pca_normalized'], y_test, gamma = 0.1, feature_type="PCA + Normalization")  # check if changing gamma fixes error

    # For HOG features
    train_evaluate_visualise(processed_data['X_train_hog'], y_train, processed_data['X_test_hog'], y_test, gamma = 0.1, feature_type="HOG")

    # For HOG + Normalization features
    train_evaluate_visualise(processed_data['X_train_hog_normalized'], y_train, processed_data['X_test_hog_normalized'], y_test, gamma = 0.1, feature_type="HOG + Normalization")

    # For HOG + PCA + Normalization features
    train_evaluate_visualise(processed_data['X_train_pca_hog_normalized'], y_train, processed_data['X_test_pca_hog_normalized'], y_test, gamma = 0.1, feature_type="HOG + PCA + Normalization")

    # For Augmented Features (Flipped, Gaussian Noise Removal and Flattened)
    train_evaluate_visualise(X_aug_flat, Y_aug, X_test, y_test, gamma = 1e-06, feature_type="Augmented")



    # Model B : Resnet18

    # Train Model with raw features
    # Convert numpy to torch tensors
    # Convert to 4D tensor with 3 channels
    X_train_tensor = torch.tensor(Xtrain, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)
    X_val_tensor   = torch.tensor(Xval, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)
    X_test_tensor  = torch.tensor(Xtest, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)

    # Labels
    y_train_tensor = torch.tensor(ytrain, dtype=torch.long).squeeze()
    y_val_tensor   = torch.tensor(yval, dtype=torch.long).squeeze()
    y_test_tensor  = torch.tensor(ytest, dtype=torch.long).squeeze()

    # Augment images for ResNet18
    # Training data: augmentation + normalization
    X_train_aug = augment_resnet_images(X_train_tensor, train=True)

    # Validation & test: normalization only
    X_val_norm  = augment_resnet_images(X_val_tensor, train=False)
    X_test_norm = augment_resnet_images(X_test_tensor, train=False)

    # Model config Resnet18 
    RESNET_CONFIG = {
    "num_classes": 2,           # BreastMNIST is binary               # Or whatever you want
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "print_every": 5}

    # Training on Raw features
    resnet_results = train_evaluate_resnet18(
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor,
    X_test_tensor,
    y_test_tensor,
    RESNET_CONFIG)

    # Evaluate Model
    print("Final test accuracy:", resnet_results["test_accuracy"])
    print("Validation accuracies per epoch:", resnet_results["val_accuracies"])
    print("Training losses per epoch:", resnet_results["losses"])


    # Training on augmented data
    resnet__augmented_results = train_evaluate_resnet18(
    X_train_aug,
    y_train_tensor,
    X_val_norm,
    y_val_tensor,
    X_test_norm,
    y_test_tensor,
    RESNET_CONFIG
    )

    # Evaluate Model
    print("Test accuracy on augmented data:", resnet__augmented_results["test_accuracy"])
    print("Validation accuracies per epoch:", resnet__augmented_results["val_accuracies"])
    print("Training losses per epoch:", resnet__augmented_results["losses"])

    



if __name__ == "__main__":
    main()



