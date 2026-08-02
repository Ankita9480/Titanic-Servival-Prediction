import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("Titanic Survival Prediction System")
st.write("Predict whether a passenger would survive the Titanic disaster based on demographic and ticket info.")

@st.cache_data
def load_and_train_model():
   
    df = pd.read_csv("/home/ankita/Documents/AIML/Titanic Survival Prediction/dataset/try.csv")
    
    # Preprocessing
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['has_cabin'] = df['Cabin'].notnull().astype(int)
    
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    dfCleaned = df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'])
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(dfCleaned, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    
    X = df_encoded.drop(columns=['Survived'])
    y = df_encoded['Survived']
    
    # Train Logistic Regression Model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    return model, X.columns

model, feature_columns = load_and_train_model()

# Sidebar User Inputs
st.sidebar.header("Enter Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Gender", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.number_input("Siblings / Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.sidebar.number_input("Parents / Children Aboard (Parch)", 0, 10, 0)
fare = st.sidebar.slider("Fare Paid ($)", 0.0, 500.0, 15.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])
has_cabin = st.sidebar.selectbox("Has Cabin Assigned?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
title = st.sidebar.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

# Predict Button
if st.button("Predict Survival Odds"):
    input_dict = {
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'has_cabin': has_cabin,
        'family_size': sibsp + parch + 1,
        'is_alone': 1 if (sibsp + parch) == 0 else 0,
        f'Sex_{sex}': 1,
        f'Embarked_{embarked}': 1,
        f'Title_{title}': 1
    }
    
    input_df = pd.DataFrame([input_dict])
    input_encoded = input_df.reindex(columns=feature_columns, fill_value=0)
    
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0]
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"**Survived!** (Probability: {prob[1]*100:.2f}%)")
    else:
        st.error(f"**Did Not Survive** (Probability: {prob[0]*100:.2f}%)")