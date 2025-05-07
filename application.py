import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sklearn as sk
from SVM import SVM
from Logistic_regression_Model import gradient_descent, predict
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')
st.write(train_data)

def get_column_types(df):
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

    return numerical_cols, categorical_cols

def handle_outliers(df, columns):
    df_clean = df.copy()

    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)

    return df_clean

def scale_features(train_data, test_data, columns):
    scaler = StandardScaler()

    train_data_scaled = train_data.copy()
    test_data_scaled = test_data.copy()

    train_data_scaled[columns] = scaler.fit_transform(train_data[columns])
    test_data_scaled[columns] = scaler.transform(test_data[columns])

    return train_data_scaled, test_data_scaled, scaler

def encode_categorical(train_data, test_data, columns):
    train_encoded = pd.get_dummies(train_data, columns=columns)
    test_encoded = pd.get_dummies(test_data, columns=columns)

    missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
    for col in missing_cols:
        test_encoded[col] = 0

    test_encoded = test_encoded[train_encoded.columns]

    return train_encoded, test_encoded

def add_bmi_feature(df):
    height_in_meters = df['Height'] / 100
    df['BMI'] = df['Weight'] / (height_in_meters ** 2)

    return df


def testing():
    # Feature engineering [ BMI ]
    print("BMI Statistics in Training Data:")
    print(train_final['BMI'].describe())
    print("_" * 20)

    # test columns types
    print("Numerical columns:", numerical_cols)
    print("\nCategorical columns:", categorical_cols)
    print("_" * 20)

    # test data cleaning [ outliers ONLY ]
    for col in numerical_cols:
        print(f"\nColumn: {col}")
        print("Before:")
        print(train_df[col].describe())
        print("\nAfter:")
        print(train_clean[col].describe())
    print("_" * 20)

    # test numerical data scaling
    for col in numerical_cols:
        print(f"\nColumn: {col}")
        print("Before scaling:")
        print(train_clean[col].describe())
        print("\nAfter scaling:")
        print(train_scaled[col].describe())
    print("_" * 20)

    # test categorical data encoding
    print("Shape before encoding:")
    print("Train:", train_scaled.shape)
    print("Test:", test_scaled.shape)
    print("Shape after encoding:")
    print("Train:", train_final.shape)
    print("Test:", test_final.shape)
    print(train_final.info())


#main
#########################

#Add BMI feature
train_with_BMI_feature = add_bmi_feature(train_data)
test_with_BMI_feature = add_bmi_feature(train_data)

#Define types of columns
numerical_cols, categorical_cols = get_column_types(train_with_BMI_feature)

#Handle outliers
train_clean = handle_outliers(train_with_BMI_feature, numerical_cols)
test_clean = handle_outliers(train_with_BMI_feature, numerical_cols)

#numerical data scaling
train_scaled, test_scaled, scaler = scale_features(train_clean, test_clean, numerical_cols)

#Categorical data encoding
train_final, test_final = encode_categorical(train_scaled, test_scaled, categorical_cols)

#testing()
