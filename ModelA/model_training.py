
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

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

# Learning curve plot



# Create cross-validation strategy
    cv = ShuffleSplit(
       n_splits=50,
       test_size=0.2,
       random_state=0
   )

# Plot
    fig, ax = plt.subplots(figsize=(6, 5))

    LearningCurveDisplay.from_estimator(
    estimator=SVM_model,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=cv,
    score_type="both",
    n_jobs=4,
    line_kw={"marker": "o"},
    std_display_style="fill_between",
    score_name="Accuracy",
    ax=ax
   )

# Clean legend and labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Validation Score"])
    ax.set_title("Learning Curve for SVM with " + feature_type + " features")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy")
    ax.grid(True)

    plt.show()




    return "Training, Evaluation and Visualisation Complete"

