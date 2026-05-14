import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Executive Salary AI", page_icon="👔", layout="wide")

# --- CUSTOM CSS (NEW DARK THEME) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    div[data-testid="stMetricValue"] {
        color: #4facfe;
    }
    /* Style inputs */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #1c2128;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        with open('salary_model.pkl', 'rb') as f:
            data = pickle.load(f)
        df = pd.read_csv('employee_data.csv')
        return data, df
    except FileNotFoundError:
        st.error("Missing files! Please run train_model.py first.")
        return None, None

saved_data, df = load_resources()

if saved_data and df is not None:
    model = saved_data["model"]
    encoders = saved_data["encoders"]
    feature_names = saved_data["features"]

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Admin Panel")
        choice = st.radio("Navigation", ["Predict Salary", "Workforce Analytics"])
        st.divider()
        st.write("System Status: **Online**")

    # --- PAGE 1: PREDICTION ---
    if choice == "Predict Salary":
        st.title("👔 Professional Salary Estimator")
        st.write("Enter data to calculate expected compensation.")

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 18, 65, 30)
                gender = st.selectbox("Gender", encoders['Gender'].classes_)
                edu = st.selectbox("Education", encoders['Education'].classes_)
                dept = st.selectbox("Department", encoders['Department'].classes_)
            with col2:
                exp = st.number_input("Years of Experience", 0, 45, 5)
                hours = st.slider("Weekly Hours", 20, 60, 40)
                perf = st.slider("Performance Score", 1, 100, 50)
                proj = st.number_input("Projects Completed", 1, 20, 5)

            if st.button("Calculate Prediction"):
                # 1. Transform inputs
                g_enc = encoders['Gender'].transform([gender])[0]
                e_enc = encoders['Education'].transform([edu])[0]
                d_enc = encoders['Department'].transform([dept])[0]
                
                # 2. Create DataFrame with EXACT column names from training
                input_df = pd.DataFrame([[
                    age, g_enc, e_enc, d_enc, exp, hours, perf, proj
                ]], columns=feature_names)
                
                # 3. Predict
                prediction = model.predict(input_df)[0]
                
                st.balloons()
                st.markdown(f"""
                <div style="background-color:#1c2128; padding:20px; border-radius:10px; border-left: 5px solid #00f2fe;">
                    <h2 style="color:white; margin:0;">Estimated Salary: ${prediction:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)

    # --- PAGE 2: ANALYTICS ---
    elif choice == "Workforce Analytics":
        st.title("📊 Workforce Insights")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Staff Count", len(df))
        c2.metric("Median Salary", f"${df['Salary'].median():,.0f}")
        c3.metric("Avg Performance", f"{df['Performance_Score'].mean():.1f}")

        st.divider()
        
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig = px.histogram(df, x="Salary", nbins=20, title="Salary Distribution", 
                               template="plotly_dark", color_discrete_sequence=['#4facfe'])
            st.plotly_chart(fig, use_container_width=True)
            
        with chart_col2:
            fig = px.scatter(df, x="Experience", y="Salary", color="Education", 
                             title="Experience vs Salary", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)