import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Title ---
st.title("🔍 Sonar Object Classification: Rock vs Mine")
st.markdown("This model predicts whether an object is a **Rock** or a **Mine** based on Sonar data.")

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    data = pd.read_csv('sonar data.csv', header=None)
    X = data.drop(columns=60, axis=1)
    Y = data[60]
    return X, Y

X, Y = load_data()

# --- Split Data ---
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# --- Train Model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# --- Accuracy Scores ---
train_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)

st.sidebar.success(f"✅ Training Accuracy: {train_accuracy:.2f}")
st.sidebar.success(f"✅ Testing Accuracy: {test_accuracy:.2f}")

# --- Input Data from User ---
st.subheader("Enter 60 Sonar Features:")
user_input = []
for i in range(60):
    val = st.number_input(f"Feature {i+1}", step=0.01, format="%.4f")
    user_input.append(val)

# --- Prediction ---
if st.button("Classify"):
    input_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    if prediction == 'R':
        st.success("🎯 The object is classified as a **Rock**")
    else:
        st.success("🎯 The object is classified as a **Mine**")
