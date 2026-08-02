import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic ML Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean Dark UI CSS ---
st.markdown("""
    <style>
    /* App Background */
    .stApp {
        background-color: #121824;
        color: #CBD5E1;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Compact Layout Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #1E2640;
        border-radius: 10px;
        padding: 10px 14px;
        border: 1px solid #2E3A59;
        text-align: center;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 2px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 20px;
        font-weight: 600;
    }

    /* Reduce Vertical Gaps */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem !important;
    }

    /* Primary Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #FF5A5F 0%, #FF3366 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF3366 0%, #E02850 100%);
        box-shadow: 0 4px 12px rgba(255, 90, 95, 0.3);
    }

    /* Headings */
    h1 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #F8FAFC !important;
        margin-bottom: 0px !important;
    }
    .section-header {
        font-size: 12px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Dynamic Path & Model Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "titanic.csv")

@st.cache_resource
def load_and_train():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset missing at: {DATASET_PATH}")
        st.stop()
        
    try:
        df = pd.read_csv(DATASET_PATH, on_bad_lines='skip', engine='python')
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()

    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median())
    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce').fillna(df['Fare'].median())
    
    if 'Sex' in df.columns:
        df['Sex_Code'] = df['Sex'].astype(str).str.lower().map({'male': 0, 'female': 1}).fillna(0)
    else:
        df['Sex_Code'] = 0

    features = ['Pclass', 'Sex_Code', 'Age', 'SibSp', 'Parch', 'Fare']
    
    for col in features:
        if col not in df.columns:
            df[col] = 0

    X = df[features]
    y = df['Survived']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler, features, df

model, scaler, features, df_raw = load_and_train()

# --- SIDEBAR (INPUT CONTROLS) ---
st.sidebar.title("Passenger Config")
st.sidebar.markdown("Adjust parameters to run prediction model.")

pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"Class {x}")
sex = st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True)
age = st.sidebar.slider("Age (Years)", 1, 80, 28)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
with col_s2:
    parch = st.number_input("Parents/Children", 0, 10, 0)
    
fare = st.sidebar.slider("Fare Paid ($)", 0.0, 500.0, 32.5)
predict_btn = st.sidebar.button("RUN PREDICTION 🚀")

# --- MAIN DASHBOARD AREA ---
st.title("🚢 Titanic ML Dashboard")
st.markdown("<p style='color:#64748B; font-size:13px; font-weight:400; margin-top:-6px;'>Real-time Machine Learning prediction and dataset analytics</p>", unsafe_allow_html=True)

# 1. Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Passengers</div>
            <div class="metric-value">{len(df_raw)}</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    surv_rate = (df_raw['Survived'].mean() * 100) if 'Survived' in df_raw.columns else 0
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Survival Rate</div>
            <div class="metric-value">{surv_rate:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    avg_fare = df_raw['Fare'].mean()
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Fare</div>
            <div class="metric-value">${avg_fare:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    avg_age = df_raw['Age'].mean()
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Passenger Age</div>
            <div class="metric-value">{avg_age:.1f} yrs</div>
        </div>
    """, unsafe_allow_html=True)

# 2. Prediction & Analytics Section
col_left, col_right = st.columns([1, 1])

sex_num = 1 if sex == "Female" else 0
input_df = pd.DataFrame([[pclass, sex_num, age, sibsp, parch, fare]], columns=features)
input_scaled = scaler.transform(input_df)
prob_survive = model.predict_proba(input_scaled)[0][1] * 100

with col_left:
    st.markdown('<div class="section-header">Prediction Analysis</div>', unsafe_allow_html=True)
    
    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob_survive,
        number = {'suffix': "%", 'font': {'color': '#F8FAFC', 'size': 26, 'weight': 'bold'}},
        title = {'text': "Survival Probability", 'font': {'color': '#94A3B8', 'size': 13, 'weight': 'normal'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2E3A59"},
            'bar': {'color': "#FF5A5F"},
            'bgcolor': "#1A2138",
            'borderwidth': 1,
            'bordercolor': "#2E3A59",
            'steps': [
                {'range': [0, 40], 'color': '#2A1A24'},
                {'range': [40, 70], 'color': '#2E2D3A'},
                {'range': [70, 100], 'color': '#1D3534'}
            ],
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=185,
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    if prob_survive >= 50:
        st.success(f"High chance of survival ({prob_survive:.1f}%)")
    else:
        st.error(f"Low chance of survival ({prob_survive:.1f}%)")

with col_right:
    st.markdown('<div class="section-header">Age & Survival Demographics</div>', unsafe_allow_html=True)
    if 'Age' in df_raw.columns and 'Survived' in df_raw.columns:
        fig_hist = px.histogram(
            df_raw, 
            x="Age", 
            color="Survived", 
            barmode="overlay",
            color_discrete_map={0: "#FF5A5F", 1: "#20C997"},
            template="plotly_dark"
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=185,
            margin=dict(l=5, r=5, t=10, b=5),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, use_container_width=True)