from itertools import cycle
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
import seaborn as sns
import numpy as np


def SVM_Model(X_train, X_test, y_train, y_test, classes):
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': [2, 3]  # Only used for 'poly' kernel
    }

    print("Starting Grid Search for Hyperparameter Tuning...")
    grid_search = GridSearchCV(
        SVC(probability=True),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\nBest Parameters Found:")
    print(grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    y_proba = best_model.predict_proba(X_test)

    n_classes = y_proba.shape[1]
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink'])

    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        avg_precision = average_precision_score(y_test_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'Class {classes[i]} (AP = {avg_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (One-vs-Rest)')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    if X_train.shape[1] == 2:
        plot_decision_boundary(best_model, X_test, y_test, classes)

    return y_pred


def plot_decision_boundary(model, X, y, classes):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.show()
