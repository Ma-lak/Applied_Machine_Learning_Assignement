
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

def train_evaluate_visualise(X_train, y_train, X_test, y_test, C, gamma, feature_type):
    """
    Train and evaluate SVM model.

    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Testing data and labels
        C: Regularization parameter
        gamma: Kernel coefficient
        feature_type: Type of features used (e.g., "Raw", "HOG", "PCA")

    Returns:
        model: Trained SVM model
    """
   # Defining model complexity and gamma
    SVM_model = svm.SVC(kernel='rbf',gamma = gamma, C=C, probability=True) # class_weight='balanced' : reduces accuracy to 0.75

   # training the model using the train dataset
    SVM_model.fit(X_train, y_train)

   # Predicting on the test and train datasets
    y_pred = SVM_model.predict(X_test)
    y_pred_train = SVM_model.predict(X_train)

   # Print classification report including accuracy, F1 score, Recall and Precision
    print("Model Accuracy with " + feature_type + " features: " + str(accuracy_score(y_test, y_pred)))
    print("Model Train Accuracy with " + feature_type + " features: " + str(accuracy_score(y_train, y_pred_train)))
    print("Model F1 Score with " + feature_type + " features: " + str(f1_score(y_test, y_pred)))
    print("Model Precision Score with " + feature_type + " features: " + str(precision_score(y_test, y_pred)))
    print("Model Recall Score with " + feature_type + " features: " + str(recall_score(y_test, y_pred)))


   # Visualizing the data based on the two classes
    if feature_type== "Raw":
        print("Visualizing " + feature_type + " features...")
        fig, ax = plt.subplots(figsize=(5, 4))
        scatter = ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            s=150,
            c=y_train,
            edgecolors="k"
           )
        ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax.set_title(feature_type +" features in two-dimensional feature space")
        plt.show()


# Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    Cs = [0.01, 0.1, 1, 10, 100]
    train_acc = []
    test_acc = []

    for C in Cs:
        svmplot = SVC(C=C, gamma=gamma, kernel='rbf')
        svmplot.fit(X_train, y_train)

        train_acc.append(svmplot.score(X_train, y_train))
        test_acc.append(svmplot.score(X_test, y_test))

    plt.plot(Cs, train_acc, marker='o', label='Training accuracy')
    plt.plot(Cs, test_acc, marker='o', label='Test accuracy')
    plt.xscale('log')
    plt.xlabel('C (model complexity)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Overfitting analysis for ' + feature_type + ' features')
    plt.show()

    return SVM_model




