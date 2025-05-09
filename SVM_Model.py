import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def initialize_parameters(n_features):
    weights = np.random.randn(n_features)
    bias = 0
    return weights, bias

def predict_svm(X, weights, bias):
    linear_output = np.dot(X, weights) - bias
    return np.sign(linear_output)

def train_and_evaluate_svm(X_train, Y_train, X_test, Y_test, learning_rate=0.001, lambda_param=0.01, iterations=1000):
    Y_train = np.where(Y_train <= 0, -1, 1)
    Y_test = np.where(Y_test <= 0, -1, 1)

    n_samples, n_features = X_train.shape
    weights, bias = initialize_parameters(n_features)

    for _ in range(iterations):
        for idx, x_i in enumerate(X_train):
            condition = Y_train[idx] * (np.dot(x_i, weights) - bias) >= 1
            if condition:
                weights -= learning_rate * (2 * lambda_param * weights)
            else:
                weights -= learning_rate * (2 * lambda_param * weights - np.dot(x_i, Y_train[idx]))
                bias -= learning_rate * Y_train[idx]

    predictions = predict_svm(X_test, weights, bias)

    accuracy = accuracy_score(Y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(Y_test, predictions, target_names=["Not Obese", "Obese"]))
    return weights, bias
