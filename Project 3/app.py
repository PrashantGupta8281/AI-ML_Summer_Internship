import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Canada Income Predictor",
    page_icon="🇨🇦",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e3a8a,#dc2626);
    color:white;
}

.main-title{
    text-align:center;
    font-size:52px;
    font-weight:700;
    color:white;
    margin-bottom:0px;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#d1d5db;
    margin-bottom:35px;
}

.glass{
    background:rgba(255,255,255,0.12);
    backdrop-filter:blur(18px);
    padding:30px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,.2);
    box-shadow:0 8px 32px rgba(0,0,0,.3);
}

.result{
    background:linear-gradient(90deg,#16a34a,#22c55e);
    padding:25px;
    border-radius:20px;
    text-align:center;
    color:white;
    font-size:30px;
    font-weight:bold;
    box-shadow:0 10px 25px rgba(0,0,0,.3);
}

.metric{
    background:rgba(255,255,255,.12);
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
}

div.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#ef4444,#dc2626);
    color:white;
    border:none;
    border-radius:12px;
    height:55px;
    font-size:20px;
    font-weight:bold;
}

div.stButton>button:hover{
    background:linear-gradient(90deg,#dc2626,#991b1b);
    transform:scale(1.03);
}

section[data-testid="stSidebar"]{
    background:#111827;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Dataset ----------------
df = pd.read_csv(Path(__file__).parent / "canada_per_capita_income.csv")

X = df[['year']]
y = df['per capita income (US$)']

model = LinearRegression()
model.fit(X, y)

# ---------------- Header ----------------
st.markdown("<h1 class='main-title'>🇨🇦 Canada Per Capita Income Predictor</h1>", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Machine Learning • Linear Regression • Streamlit Dashboard</p>", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.title("📊 Dashboard")

st.sidebar.info("""
### About

This project predicts Canada's Per Capita Income using a Linear Regression model.

**Model:** Linear Regression

**Dataset:** Canada Per Capita Income

**Framework:** Streamlit
""")

st.sidebar.success("Made with ❤️ using Python")

# ---------------- Statistics ----------------
col1,col2,col3=st.columns(3)

with col1:
    st.markdown(f"""
    <div class='metric'>
    <h3>📅 First Year</h3>
    <h2>{df.year.min()}</h2>
    </div>
    """,unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric'>
    <h3>📅 Last Year</h3>
    <h2>{df.year.max()}</h2>
    </div>
    """,unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric'>
    <h3>📈 Records</h3>
    <h2>{len(df)}</h2>
    </div>
    """,unsafe_allow_html=True)

st.write("")

# ---------------- Prediction Card ----------------
st.markdown("<div class='glass'>",unsafe_allow_html=True)

year=st.slider(
    "Select Year",
    min_value=1960,
    max_value=2100,
    value=2020
)

if st.button("🚀 Predict Income"):

    prediction=model.predict(np.array([[year]]))[0]

    st.markdown(f"""
    <div class='result'>
        Estimated Per Capita Income<br><br>
        <span style="font-size:45px;">${prediction:,.2f}</span><br>
        Year: {year}
    </div>
    """,unsafe_allow_html=True)

st.markdown("</div>",unsafe_allow_html=True)

# ---------------- Data Preview ----------------
st.write("")
st.subheader("📋 Dataset Preview")
st.dataframe(df, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<center><h4>🇨🇦 Canada Income Prediction</h4></center>",
    unsafe_allow_html=True
)
