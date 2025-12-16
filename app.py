# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Predict whether a patient has heart disease using Machine Learning")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"✅ Accuracy: **{accuracy:.2f}**")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("🧑‍⚕️ Patient Information")

def user_input_features():
    age = st.slider("Age", 20, 80, 45)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.slider("Chest Pain Type (0–3)", 0, 3, 1)
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [1, 0])
    restecg = st.slider("Rest ECG Results (0–2)", 0, 2, 1)
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.slider("Slope (0–2)", 0, 2, 1)
    ca = st.slider("Major Vessels Colored (0–4)", 0, 4, 0)
    thal = st.slider("Thalassemia (1–3)", 1, 3, 2)

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# -----------------------------
# Prediction
# -----------------------------
st.subheader("🔍 Prediction Result")

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
prediction_prob = model.predict_proba(input_scaled)[0][1]

if prediction == 1:
    st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {prediction_prob:.2f}")
else:
    st.success(f"✅ Low Risk of Heart Disease\n\nProbability: {prediction_prob:.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("📌 Built with Streamlit & Scikit-Learn | Heart Disease Prediction Project")
