from sklearn.metrics import classification_report, accuracy_score


def evaluate(y_test, y_pred):

    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    return accuracy_score(y_test, y_pred)#, accuracy_score(y_train, y_pred_train)
