import pandas as pd
import numpy as np
import math


def sigmoid(z):
    return 1/(1+np.exp(-z))


def compute_cost_logistic(X, y, w, b):
    cost = 0
    m = X.shape[0]
    for i in range(m):
        z_i = np.dot(X[i], w)+b
        y_pred = sigmoid(z_i)
        cost += y[i]*np.log2(y_pred)+(1-y[i])*np.log2(1-y_pred)

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


def gradient_descent(X,y , w_initial, b_initial, alpha, iterations):
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


def predict(X, w, b):
    return sigmoid(np.dot(X, w) + b) >= 0.5