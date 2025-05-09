import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def KNN_Model(X_train, X_test, y_train, y_test, classes):
    # Some validation before running the model
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError("Missing values detected in feature data.")
    if np.isnan(y_train).any() or np.isnan(y_test).any():
        raise ValueError("Missing values detected in target labels.")
    if np.isinf(X_train).any().any() or np.isinf(X_test).any().any():
        raise ValueError("Infinite values detected in feature data.")
    
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_knn = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)

    cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5)
    print(f"CV Accuracy scores: {cv_scores}")
    print(f"Average CV accuracy: {cv_scores.mean():.4f}")


    predictions = best_knn.predict(X_test)

    print("\nTest Accuracy: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return predictions
