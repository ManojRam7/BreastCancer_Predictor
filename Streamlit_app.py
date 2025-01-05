import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load the trained model
MODEL_FILE = 'finalized_model.pkl'
with open(MODEL_FILE, 'rb') as file:
    model = pickle.load(file)

# App title
st.title("Breast Cancer Prediction App")

# Description
st.write("""
This app predicts whether a breast cancer cell is malignant or benign based on user inputs.
""")

# Sidebar for user inputs
st.sidebar.header("Input Features")

# Load dataset to provide feature information
data = load_breast_cancer()
feature_names = data['feature_names']

# Dynamic sliders for features
def user_input_features():
    input_data = {}
    for feature in feature_names:
        min_val = float(data['data'][:, list(feature_names).index(feature)].min())
        max_val = float(data['data'][:, list(feature_names).index(feature)].max())
        mean_val = float(data['data'][:, list(feature_names).index(feature)].mean())
        input_data[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val)
    return np.array([list(input_data.values())])

input_features = user_input_features()

# Display user inputs
st.subheader("User Input Features")
st.write(pd.DataFrame(input_features, columns=feature_names))

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)
    
    # Display results
    if prediction[0] == 0:
        st.success("The model predicts the cell is **Benign**.")
    else:
        st.error("The model predicts the cell is **Malignant**.")
    
    st.subheader("Prediction Probabilities")
    st.write(f"Benign: {prediction_proba[0][0]:.2f}")
    st.write(f"Malignant: {prediction_proba[0][1]:.2f}")
