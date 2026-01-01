
import numpy as np

# Load the dataset
data = np.load("breastmnist.npz")

# Inspect available keys
print(data.files)

Xtrain = data["train_images"]
ytrain = data["train_labels"]

Xtest = data["test_images"]
ytest = data["test_labels"]

print("Train shape:", Xtrain.shape, ytrain.shape)
print("Test shape:", Xtest.shape, ytest.shape)


X_train = Xtrain.reshape(len(Xtrain), -1)  # (N, 784)
X_test = Xtest.reshape(len(Xtest), -1)  # (N, 784)
y_train = ytrain.squeeze()
y_test = ytest.squeeze()


