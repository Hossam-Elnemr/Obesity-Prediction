import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sklearn as sk
from Logistic_regression_Model import gradient_descent, predict
train_data = pd.read_csv("Obesity-Prediction/train_dataset.csv")
test_data = pd.read_csv("Obesity-Prediction/test_dataset.csv")
st.write(train_data)
