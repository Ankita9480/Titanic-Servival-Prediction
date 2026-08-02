import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="centered"
)

# --- Dynamic File Path Resolution ---
# Yeh line automatically current directory ka path nikal leti hai
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "titanic.csv")

# --- Model Training Function with Caching ---
@st.cache_resource
def load_and_train_model():
    # Dataset Load
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset missing! Path checked: {DATASET_PATH}")
        st.stop()
        
    df = pd.read_csv(DATASET_PATH)

    # Data Preprocessing
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Convert Categorical variables to Numeric
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Select Features & Target
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features]
    y = df['Survived']

    # Train-Test Split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train Model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler, features

# Load model and scaler
model, scaler, features = load_and_train_model()

# --- Streamlit User Interface ---
st.title("Titanic Survival Prediction")
st.write("Enter passenger details below to predict their chance of survival.")

st.markdown("---")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2, help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=1, max_value=80, value=25)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=500.0, value=32.2)

# Convert Gender to numeric format matching training logic
sex_numeric = 1 if sex == "Female" else 0

st.markdown("---")

# Prediction Trigger Button
if st.button("Predict Survival", type="primary"):
    # Input DataFrame
    input_data = pd.DataFrame([[pclass, sex_numeric, age, sibsp, parch, fare]], columns=features)
    
    # Scale inputs & Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    # Display Result
    if prediction == 1:
        st.success(f"**Survived!** (Confidence: {prediction_proba[1]*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"**Did Not Survive** (Confidence: {prediction_proba[0]*100:.1f}%)")