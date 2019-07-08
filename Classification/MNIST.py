import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Import MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X = mnist["data"]
y = mnist["target"].astype(int)


# Function plots a digit from array of pixels
def plot_digit(digit_array):
    digit_image = digit_array.reshape(28, 28)
    plt.matshow(digit_image, cmap=matplotlib.cm.binary)
    plt.axis("off")
    plt.show()

# plot_digit(X[0])


# Split into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)


# Scales X train and test data
def scale_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = scale_data(X_train, X_test)


# Create classification models
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
sgd_clf = SGDClassifier(random_state=42)
forest_clf = RandomForestClassifier(random_state=42)


from sklearn.model_selection import cross_val_predict
# Predict scores and plot precision-recall curve for one digit classifier models SGD and Random Forest
# Compute scores for one digit classifiers
def predict_scores_digit(X, y, digit):
    y_true = (y_train == digit)
    y_scores_sgd = cross_val_predict(sgd_clf, X_train, y_true,
                                     cv=3, method='decision_function')
    y_scores_forest = cross_val_predict(forest_clf, X_train, y_true,
                                        cv=3, method='predict_proba')[:,1]
    return y_true, y_scores_sgd, y_scores_forest

# Plot precision-recall curve function
from sklearn.metrics import precision_recall_curve
def plot_precision_recall_curve(y_true, y_scores, title=None):
    plt.figure()
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    plt.plot(thresholds, precisions[:-1], 'b-', label="Precision")
    plt.plot(thresholds, recalls[:-1], 'g--', label="Recall")
    plt.legend()
    plt.xlabel("Threshold")
    plt.ylim([0, 1])
    plt.xlim([min(thresholds), max(thresholds)])
    if title:
        plt.title(title)
    plt.show()

# Call previous functions to plot precision-recall curves for selected digit
def SGD_Forest_digit_plot_precision_recall_curves(digit):
    y_train_digit, y_scores_sgd_digit, y_scores_forest_digit = predict_scores_digit(X_train_scaled, y_train, digit)
    plot_precision_recall_curve(y_train_digit, y_scores_sgd_digit, ("SGD for digit " + str(digit)))
    plot_precision_recall_curve(y_train_digit, y_scores_forest_digit, ("Random forest for digit " + str(digit)))

# Plot precision-recall curves for number 4 one digit classifiers SGD and Random forest
# SGD_Forest_digit_plot_precision_recall_curves(4)

from sklearn.metrics import precision_score, recall_score, f1_score
# Compute precision, recall and F1 score for one digit classifiers
def digit_clf_precision_recall_F1_score(digit, thresholds=[0, 0.5]):
    y_train_digit, y_scores_sgd_digit, y_scores_forest_digit = predict_scores_digit(X_train_scaled, y_train, digit)
    y_scores_array = [y_scores_sgd_digit, y_scores_forest_digit]
    clf_names = ["SGD", "Random forest"]
    for i in range(2):
        y_predict = (y_scores_array[i] > thresholds[i])
        print(clf_names[i] + "\n",
              "Precision - Recall - F1 score\n",
              precision_score(y_train_digit, y_predict),
              recall_score(y_train_digit, y_predict),
              f1_score(y_train_digit, y_predict))

# Print precision, recall and F1 score for number 4 one digit classifiers SGD and Random forest
# precision_recall_F1_score(4)

# FIXME function below does'nt work propertly
# Plot ROC curve function
from sklearn.metrics import roc_curve, roc_auc_score
def plot_roc_curve(digit, thresholds=[0, 0.5]):
    y_train_digit, y_scores_sgd_digit, y_scores_forest_digit = predict_scores_digit(X_train_scaled, y_train, digit)
    y_scores_array = [y_scores_sgd_digit, y_scores_forest_digit]
    clf_names = ["SGD", "Random forest"]
    plt.figure()
    for i in range(2):
        print(thresholds[i])
        y_predict = (y_scores_array[i] > thresholds[i])
        fpr, tpr, _ = roc_curve(y_train_digit, y_predict)
        plt.plot(fpr, tpr, label=clf_names[i])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.axis([-0.05, 1, 0, 1.05])
    plt.legend()
    plt.show()

# plot_roc_curve(4)


# Predict y_train for SGD and Random forest classifiers
# y_pred_sgd = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# y_pred_forest = cross_val_predict(forest_clf, X_train_scaled, y_train, cv=3)


# Plot normalized confusion matrix of classifier model
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred, title=None):
    conf_mx = confusion_matrix(y_true, y_pred)
    norm_conf_mx = conf_mx / conf_mx.sum(axis=1, keepdims=True)
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    if title:
        plt.title(title)
    plt.show()

# plot_confusion_matrix(y_train, y_pred_sgd, "SGD")
# plot_confusion_matrix(y_train, y_pred_forest, "Forest")


# Print cross-validate precision, recall and F1 score for classifier
from sklearn.model_selection import cross_validate
def print_precision_recall_f1_score(clf, title=None):
    scoring = ["precision_macro", "recall_macro", "f1_macro"]
    scores = cross_validate(clf, X_train_scaled, y_train,
                            cv=3, scoring=scoring)
    test_precision = np.mean(scores["test_precision_macro"])
    test_recall = np.mean(scores["test_recall_macro"])
    test_f1 = np.mean(scores["test_f1_macro"])
    if title:
        print(title)
    print(f"Precision = {test_precision:.4f}")
    print(f"Recall = {test_recall:.4f}")
    print(f"F1 score = {test_f1:.4f}")

# print_precision_recall_f1_score(sgd_clf, "SGD")
# print_precision_recall_f1_score(forest_clf, "Random forest")
# # SGD
# # Precision = 0.9085
# # Recall = 0.9083
# # F1 score = 0.9082
# # Random forest
# # Precision = 0.9410
# # Recall = 0.9407
# # F1 score = 0.9407


# Use grid search to find best max_features value for Random forest classifier and save a DataFrame with the results
from sklearn.model_selection import GridSearchCV

def hyper_parameter_tuning(clf, param_grid, scoring, df_path, refit_parameter=None):
    if refit_parameter:
        grid_search = GridSearchCV(clf, param_grid, scoring,
                               cv=3, refit=refit_parameter)
    else:
        grid_search = GridSearchCV(clf, param_grid, scoring,
                                   cv=3)
    grid_search.fit(X_train_scaled, y_train)
    results_grid_search = pd.DataFrame(grid_search.cv_results_)
    results_grid_search.to_pickle(df_path)
    return grid_search.best_estimator_, results_grid_search

def grid_search_forest_max_features():
    clf_forest = RandomForestClassifier(random_state=42)
    param_grid_forest = {"max_features": ["auto", 0.8, "log2", None]}
    scoring = ["precision_macro", "recall_macro", "f1_macro"]
    df_path = "./DataFrames/MNIST_forest_max_features_grid_search.pkl"
    return hyper_parameter_tuning(clf_forest, param_grid_forest, scoring, df_path,
                                  "precision_macro")

def grid_search_forest_n_estimators():
    clf_forest = RandomForestClassifier(random_state=42)
    param_grid_forest = {"n_estimators": [3, 10, 30, 100]}
    scoring = ["precision_macro", "recall_macro", "f1_macro"]
    df_path = "./DataFrames/MNIST_forest_n_estimators_grid_search.pkl"
    return hyper_parameter_tuning(clf_forest, param_grid_forest, scoring, df_path,
                                  "precision_macro")

# forest_clf_best, _ = grid_search_forest_max_features()
# Best max_features = 'auto'
forest_clf_best, _ = grid_search_forest_n_estimators()
# Best n_estimators = 100
# print_precision_recall_f1_score(forest_clf_best)
# Precision = 0.9659
# Recall = 0.9658
# F1 score = 0.9658

print("Finished running")