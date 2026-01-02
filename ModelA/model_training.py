
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def train_evaluate_visualise(X_train, y_train, X_test, y_test, gamma, feature_type):

   # Defining model complexity and gamma
    SVM_model = svm.SVC(kernel='rbf',gamma = gamma, C=10.0, probability=True) # class_weight='balanced' : reduces accuracy to 0.75

   # training the model using the train dataset
    SVM_model.fit(X_train, y_train)

   # Predicting on the test and train datasets
    y_pred = SVM_model.predict(X_test)
    y_pred_train = SVM_model.predict(X_train)

   # Print classification report including accuracy, F1 score, Recall and Precision
    print("Model Accuracy with " + feature_type + " features: " + str(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

   # Visualizing the data based on the two classes
    print("Visualizing " + feature_type + " features:")

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


# Learning curve plot
    Cs = [0.01, 0.1, 1, 10, 100]
    train_acc = []
    test_acc = []

    for C in Cs:
     svm_plot = SVC(C=C, gamma='scale')
     svm_plot.fit(X_train, y_train)

     train_acc.append(svm_plot.score(X_train, y_train))
     test_acc.append(svm_plot.score(X_test, y_test))

    plt.figure(figsize=(4, 3))
    plt.plot(Cs, train_acc, marker='o', label='Training accuracy')
    plt.plot(Cs, test_acc, marker='o', label='Test accuracy')
    plt.xscale('log')
    plt.xlabel('C (model complexity)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Overfitting analysis for ' + feature_type + ' features')

    plt.show()

    return "Training, Evaluation and Visualisation Complete"

