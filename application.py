import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sklearn as sk
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')
st.write(train_data)
