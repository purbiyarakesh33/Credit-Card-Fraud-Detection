# =============================================================
# CREDIT CARD FRAUD DETECTION - STREAMLIT FRONTEND
# =============================================================

import streamlit as st
import requests
import pandas as pd
import kagglehub
import numpy as np

st.set_page_config(
    page_title="FraudShield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0a0e1a; color: #e2e8f0; }
    .stApp { background-color: #0a0e1a; }
    section[data-testid="stSidebar"] { background-color: #0f1420; border-right: 1px solid #1e2740; }

    .sidebar-logo { font-size: 1.4rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
    .sidebar-sub { font-size: 0.72rem; color: #4a5568; font-family: 'JetBrains Mono', monospace; margin-bottom: 1.5rem; }
    .sidebar-label { font-size: 0.65rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 1.2rem; margin-bottom: 0.4rem; }
    .sidebar-value { font-size: 0.8rem; color: #a0aec0; font-family: 'JetBrains Mono', monospace; }
    .metric-badge { display: inline-block; background: #1a2035; border: 1px solid #2d3748; border-radius: 4px; padding: 3px 10px; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; color: #63b3ed; margin-bottom: 5px; }

    .page-title { font-size: 2rem; font-weight: 800; color: #fff; letter-spacing: -0.03em; margin-bottom: 0.3rem; }
    .page-sub { font-size: 0.85rem; color: #4a5568; margin-bottom: 1.5rem; }
    .section-header { font-size: 0.7rem; font-weight: 700; color: #4a5568; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.8rem; padding-bottom: 0.5rem; border-bottom: 1px solid #1e2740; }

    .result-fraud { background: linear-gradient(135deg, #2d0a0a, #450f0f); border: 1px solid #e53e3e; border-radius: 10px; padding: 1.2rem 1.5rem; font-size: 1.4rem; font-weight: 800; color: #fc8181; margin-bottom: 1rem; }
    .result-legit { background: linear-gradient(135deg, #0a1f0a, #0f2d1a); border: 1px solid #38a169; border-radius: 10px; padding: 1.2rem 1.5rem; font-size: 1.4rem; font-weight: 800; color: #68d391; margin-bottom: 1rem; }

    .metric-card { background: #0f1420; border: 1px solid #1e2740; border-radius: 8px; padding: 1rem; text-align: center; }
    .metric-card-label { font-size: 0.65rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace; }
    .metric-card-value { font-size: 1.3rem; font-weight: 800; color: #e2e8f0; }
    .metric-card-value-fraud { font-size: 1.3rem; font-weight: 800; color: #fc8181; }
    .metric-card-value-legit { font-size: 1.3rem; font-weight: 800; color: #68d391; }

    .tx-info { background: #0f1420; border: 1px solid #1e2740; border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #718096; }
    .tx-info span { color: #63b3ed; font-weight: 600; }
    .tx-fraud { color: #fc8181 !important; }
    .tx-legit { color: #68d391 !important; }

    .stButton > button { background: #0f1420 !important; border: 1px solid #63b3ed !important; color: #63b3ed !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.9rem !important; border-radius: 6px !important; width: 100% !important; transition: all 0.2s !important; }
    .stButton > button:hover { background: #63b3ed !important; color: #0a0e1a !important; }

    hr { border-color: #1e2740 !important; }
    label { color: #718096 !important; font-size: 0.75rem !important; font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_sample_data():
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    df = pd.read_csv(path + r'\fraudTrain.csv')
    fraud = df[df['is_fraud'] == 1].sample(500, random_state=42)
    legit = df[df['is_fraud'] == 0].sample(500, random_state=42)
    return pd.concat([fraud, legit]).reset_index(drop=True)

@st.cache_data
def get_categories():
    try:
        resp = requests.get("http://127.0.0.1:8000/categories")
        return resp.json()
    except:
        return None

sample_df = load_sample_data()
categories = get_categories()

# -------------------------
# SESSION STATE
# -------------------------
if 'loaded_tx' not in st.session_state:
    st.session_state.loaded_tx = None
if 'actual_label' not in st.session_state:
    st.session_state.actual_label = None

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-logo">FraudShield</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Semi-Supervised Detection System</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-badge">Autoencoder (TF)</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-badge">XGBoost Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-badge">SMOTE Balancing</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">kartik2112/fraud-detection</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Training Samples</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">1,296,675</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Fraud Rate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">0.58% (7,506 cases)</div>', unsafe_allow_html=True)

# -------------------------
# MAIN
# -------------------------
st.markdown('<div class="page-title">Transaction Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Load a sample transaction or enter details manually</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Random Transaction"):
        row = sample_df.sample(1).iloc[0]
        st.session_state.loaded_tx = row
        st.session_state.actual_label = int(row['is_fraud'])
with c2:
    if st.button("Load Fraud Case"):
        fraud_df = sample_df[sample_df['is_fraud'] == 1]
        row = fraud_df.sample(1).iloc[0]
        st.session_state.loaded_tx = row
        st.session_state.actual_label = 1
with c3:
    if st.button("Load Legit Case"):
        legit_df = sample_df[sample_df['is_fraud'] == 0]
        row = legit_df.sample(1).iloc[0]
        st.session_state.loaded_tx = row
        st.session_state.actual_label = 0

if st.session_state.loaded_tx is not None:
    row = st.session_state.loaded_tx
    actual = "FRAUD" if st.session_state.actual_label == 1 else "LEGITIMATE"
    tag = "tx-fraud" if st.session_state.actual_label == 1 else "tx-legit"
    st.markdown(f"""
<div class="tx-info">
    Merchant: <span>{str(row['merchant'])[:30]}</span> &nbsp;|&nbsp;
    Amount: <span>${row['amt']:.2f}</span> &nbsp;|&nbsp;
    Category: <span>{row['category']}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

tx = st.session_state.loaded_tx
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-header">Financial Info</div>', unsafe_allow_html=True)
    amt = st.number_input("Amount ($)", value=float(tx['amt']) if tx is not None else 0.0, min_value=0.0, format="%.2f")
    cat_options = categories['categories'] if categories else ['grocery_pos', 'shopping_net', 'entertainment']
    default_cat = tx['category'] if tx is not None and tx['category'] in cat_options else cat_options[0]
    category = st.selectbox("Category", cat_options, index=cat_options.index(default_cat))
    merchant_options = categories['merchants'][:100] if categories else ['merchant_1']
    default_merch = tx['merchant'] if tx is not None and tx['merchant'] in merchant_options else merchant_options[0]
    merchant = st.selectbox("Merchant", merchant_options, index=merchant_options.index(default_merch) if default_merch in merchant_options else 0)

with col2:
    st.markdown('<div class="section-header">Customer Info</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ['M', 'F'], index=0 if tx is None or tx['gender'] == 'M' else 1)
    state_options = categories['states'] if categories else ['NY', 'CA', 'TX']
    default_state = tx['state'] if tx is not None and tx['state'] in state_options else state_options[0]
    state = st.selectbox("State", state_options, index=state_options.index(default_state))
    job_options = categories['jobs'][:100] if categories else ['engineer']
    job = st.selectbox("Job", job_options, index=0)
    city_pop = st.number_input("City Population", value=int(tx['city_pop']) if tx is not None else 50000, min_value=0)

with col3:
    st.markdown('<div class="section-header">Behavioral Info</div>', unsafe_allow_html=True)
    from datetime import datetime
    now = datetime.now()
    tx_time = str(tx['trans_date_trans_time']).split(' ')[1] if tx is not None else '12:00:00'
    tx_hour = int(tx_time.split(':')[0]) if tx is not None else now.hour
    hour = st.slider("Transaction Hour", 0, 23, tx_hour)
    day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, now.weekday())
    dob = pd.to_datetime(tx['dob']) if tx is not None else pd.Timestamp('1990-01-01')
    age = int((pd.Timestamp('2020-12-31') - dob).days // 365) if tx is not None else 35
    age = st.number_input("Customer Age", value=age, min_value=18, max_value=100)
    if tx is not None:
        dist = float(((tx['lat'] - tx['merch_lat'])**2 + (tx['long'] - tx['merch_long'])**2)**0.5)
    else:
        dist = 0.0
    distance = st.number_input("Distance to Merchant", value=dist, min_value=0.0, format="%.4f")
    tx_velocity = st.number_input("Total Transactions (this card)", value=10, min_value=1)
    amt_deviation = st.number_input("Amount Deviation", value=0.0, min_value=0.0, format="%.4f")
    tx_per_day = st.number_input("Transactions Today", value=1, min_value=1)

st.markdown("---")

if st.button("Analyze Transaction"):
    with st.spinner("Running models..."):
        try:
            payload = {
                "amt": amt, "category": category, "merchant": merchant,
                "gender": gender, "state": state, "city_pop": city_pop,
                "job": job, "hour": hour, "day_of_week": day_of_week,
                "age": age, "distance": distance, "tx_velocity": tx_velocity,
                "amt_deviation": amt_deviation, "tx_per_day": tx_per_day
            }

            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            result = response.json()

            final = result["prediction"]
            prob = result["fraud_probability"]
            recon = result["reconstruction_error"]

            if final == "Fraud":
                st.markdown('<div class="result-fraud">FRAUDULENT TRANSACTION DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-legit">TRANSACTION IS LEGITIMATE</div>', unsafe_allow_html=True)

           

            c1, c2, c3 = st.columns(3)
            with c1:
                color = "metric-card-value-fraud" if final == "Fraud" else "metric-card-value-legit"
                st.markdown(f'<div class="metric-card"><div class="metric-card-label">Prediction</div><div class="{color}">{final}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-card-label">Fraud Probability</div><div class="metric-card-value">{prob}%</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-card-label">Reconstruction Error</div><div class="metric-card-value">{recon}</div></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Cannot connect to API. Make sure backend is running! Error: {e}")