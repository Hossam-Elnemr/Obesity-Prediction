import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def sigmoid(z):
    return 1/(1+np.exp(-z))


def compute_cost_logistic(X, y, w, b):
    cost = 0
    m = X.shape[0]
    for i in range(m):
        z_i = np.dot(X[i], w)+b
        y_pred = sigmoid(z_i)
        cost += y[i]*np.log(y_pred)+(1-y[i])*np.log(1-y_pred)

    return -cost/m


def compute_gradient_logistic(X, y, w, b):

    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        z_i = np.dot(X[i], w)+b
        y_pred = sigmoid(z_i)
        dj_dw += (y_pred-y[i])*X[i]
        dj_db += (y_pred-y[i])
    return dj_dw/m, dj_db/m


def gradient_descent(X, y, w_initial, b_initial, alpha, iterations):
    w = w_initial
    b = b_initial
    J_history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        if i<100000:
            J_history.append( compute_cost_logistic(X, y, w, b) )
        if i% math.ceil(iterations / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b

def one_vs_rest_train(X, y, classes, alpha=0.01, iterations=1000):
    classifiers = {}
    m, n = X.shape
    encoder = joblib.load('label_encoder.pkl')
    for cls in classes:
        class_name = encoder.inverse_transform([cls])[0]
        print(f"\nTraining classifier for class: {class_name}")
        # Convert y to binary: 1 if current class, 0 otherwise
        y_binary =[]
        for label in y:
            if label == cls:
                y_binary.append(1)
            else:
                y_binary.append(0)

        w_init = np.zeros(n)
        b_init = 0

        # Train logistic regression
        w_trained, b_trained = gradient_descent(X, y_binary, w_init, b_init, alpha, iterations)

        # Save model
        classifiers[cls] = (w_trained, b_trained)
    
    return classifiers

def one_vs_rest_predict(X, y, classifiers):
    predictions = []
    m = X.shape[0]
    for i in range(m):
        class_probs = {}
        for cls, (w, b) in classifiers.items():
            prob = sigmoid(np.dot(X[i], w) + b)
            class_probs[cls] = prob
        # Choose the class with the highest probability
        predicted_class = max(class_probs, key=class_probs.get)
        predictions.append(predicted_class)
    encoder = joblib.load('label_encoder.pkl')
    print("Accuracy Score: {}".format(accuracy_score(y, predictions)))
    print("Classification Report\n{}".format(classification_report(y, predictions, target_names=encoder.classes_)))
    return predictions

def confusion_Matrix(y_test, predictions):
    encoder = joblib.load('label_encoder.pkl')
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
