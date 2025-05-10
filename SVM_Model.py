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

def train_multi_class_svm(X_train, Y_train, learning_rate=0.001, lambda_param=0.01, iterations=1000):
    label_encoder = LabelEncoder()
    Y_train_encoded = label_encoder.fit_transform(Y_train)
    classes = label_encoder.classes_

    classifiers = {}
    for class_label in classes:
        binary_labels = np.where(Y_train_encoded == label_encoder.transform([class_label])[0], 1, -1)
        weights, bias = train_svm(X_train, binary_labels, learning_rate, lambda_param, iterations)
        classifiers[class_label] = (weights, bias)

    return classifiers, label_encoder

def predict_multi_class_svm(X_test, classifiers, label_encoder):
    class_confidences = {cls: predict_svm(X_test, weights, bias) for cls, (weights, bias) in classifiers.items()}
    predictions = np.array([max(class_confidences, key=lambda k: class_confidences[k][i]) for i in range(X_test.shape[0])])
    return label_encoder.transform(predictions)

def predict_and_evaluate(X_test, Y_test, classifiers, label_encoder):
    Y_pred = predict_multi_class_svm(X_test, classifiers, label_encoder)
    Y_test_decoded = label_encoder.inverse_transform(Y_test)  # Ensure Y_test is properly decoded

    accuracy = accuracy_score(Y_test_decoded, label_encoder.inverse_transform(Y_pred))
    print(f"Multi-Class SVM Accuracy: {accuracy:.2f}\n")
    print("Classification Report:")
    print(classification_report(Y_test_decoded, label_encoder.inverse_transform(Y_pred), target_names=label_encoder.classes_))
    return Y_pred