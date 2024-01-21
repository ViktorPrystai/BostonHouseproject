import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from custom_model import CustomLinearRegression


# Load the preprocessed data
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
try:
    raw_df1 = pd.read_csv('../input/boston-house-prices/housing.csv', header=None, delimiter=r"\s+", names=column_names)
except:
    raw_df1 = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

# Feature columns
feature_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Data preprocessing function
def preprocess_data(inputs):
    inputs = [float(inputs[col]) for col in feature_columns]
    return np.array(inputs).reshape(1, -1)

# Load the pre-trained model
lm = joblib.load('linear_regression_model.joblib')

# Streamlit app
st.title('Boston Housing Price Prediction')

# User input for features
user_inputs = {}
for col in feature_columns:
    user_inputs[col] = st.number_input(f'Enter {col}', value=0.0)

# Preprocess user inputs
input_features = preprocess_data(user_inputs)

# Make predictions
prediction = lm.predict(input_features)

# Display the prediction
st.subheader('Predicted Housing Price:')
st.write(prediction[0])
