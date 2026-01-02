
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
from typing import Dict

def feature_pipeline(Xtrain, Xtest, ytrain, ytest) -> Dict[str, np.ndarray]:
    
    # Flatten the images for SVM - keeping raw features
    X_train = Xtrain.reshape(len(Xtrain), -1)  # (N, 784)
    X_test = Xtest.reshape(len(Xtest), -1)  # (N, 784)
    y_train = ytrain.ravel()
    y_test = ytest.ravel()

    # PCA (keep 80% variance)
    pca = PCA(n_components=0.8, random_state=42) #0.8
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    # Normalize PCA features
    X_train_pca_normalized = preprocessing.normalize(X_train_pca, norm='l2')
    X_test_pca_normalized = preprocessing.normalize(X_test_pca, norm='l2')

    # Extract HOG features
    features_train = []
    for img in Xtrain:
        hog_feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(7, 7), # explain these!!!!!!!!!
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features_train.append(hog_feat)

    X_train_hog = np.array(features_train)

    features_test = []
    for img in Xtest:
        hog_feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features_test.append(hog_feat)

    X_test_hog = np.array(features_test)


    # Normalise HOG features
    X_train_hog_normalized = preprocessing.normalize(X_train_hog, norm='l2')
    X_test_hog_normalized = preprocessing.normalize(X_test_hog, norm='l2')


    # Apply PCA to HOG features.   
    pca = PCA(n_components=0.8, random_state=42) #0.8
    X_train_pca_hog = pca.fit_transform(X_train_hog)
    X_test_pca_hog  = pca.transform(X_test_hog)

    # Normalize HOG PCA features
    X_train_pca_hog_normalized = preprocessing.normalize(X_train_pca_hog, norm='l2')
    X_test_pca_hog_normalized = preprocessing.normalize(X_test_pca_hog, norm='l2')



    return {
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "X_train_hog": X_train_hog,
        "X_test_hog": X_test_hog,
        "X_train_pca_normalized" : X_train_pca_normalized,
        "X_test_pca_normalized": X_test_pca_normalized,
        "X_train_hog_normalized": X_train_hog_normalized,
        "X_test_hog_normalized": X_test_hog_normalized,
        "X_train_pca_hog": X_train_pca_hog, #surement a enlever !!!!! does not make sense
        "X_test_pca_hog": X_test_pca_hog,
        "X_train_pca_hog_normalized" : X_train_pca_hog_normalized,
        "X_test_pca_hog_normalized" : X_test_pca_hog_normalized
    }
