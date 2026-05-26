# =============================================================
# CREDIT CARD FRAUD DETECTION - STREAMLIT APP (NO BACKEND)
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import kagglehub
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

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

    .stButton > button { background: #0f1420 !important; border: 1px solid #63b3ed !important; color: #63b3ed !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.9rem !important; border-radius: 6px !important; width: 100% !important; transition: all 0.2s !important; }
    .stButton > button:hover { background: #63b3ed !important; color: #0a0e1a !important; }

    hr { border-color: #1e2740 !important; }
    label { color: #718096 !important; font-size: 0.75rem !important; font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_models():
    autoencoder = tf.keras.models.load_model('autoencoder_fraud.keras')
    
    xgb = XGBClassifier()
    xgb.load_model('xgb_model.json')
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('target_encoders.pkl', 'rb') as f:
        target_encoders = pickle.load(f)
    
    with open('best_threshold.pkl', 'rb') as f:
        best_threshold = pickle.load(f)
    
    return autoencoder, xgb, scaler, target_encoders, best_threshold

autoencoder, xgb_model, scaler, target_encoders, best_threshold = load_models()

FEATURE_COLS = ['merchant', 'category', 'amt', 'gender', 'state', 'city_pop',
                'job', 'hour', 'day_of_week', 'age', 'distance',
                'tx_velocity', 'amt_deviation', 'tx_per_day']

# -------------------------
# LOAD SAMPLE DATA
# -------------------------
@st.cache_data
def load_sample_data():
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    df = pd.read_csv(path + '/fraudTrain.csv')
    fraud = df[df['is_fraud'] == 1].sample(500, random_state=42)
    legit = df[df['is_fraud'] == 0].sample(500, random_state=42)
    return pd.concat([fraud, legit]).reset_index(drop=True)

sample_df = load_sample_data()

# -------------------------
# PREDICT FUNCTION
# -------------------------
def predict(amt, category, merchant, gender, state, city_pop, job,
            hour, day_of_week, age, distance, tx_velocity, amt_deviation, tx_per_day):

    data = {
        'merchant': merchant, 'category': category, 'amt': amt,
        'gender': 1 if gender == 'F' else 0, 'state': state,
        'city_pop': city_pop, 'job': job, 'hour': hour,
        'day_of_week': day_of_week, 'age': age, 'distance': distance,
        'tx_velocity': tx_velocity, 'amt_deviation': amt_deviation,
        'tx_per_day': tx_per_day
    }

    df = pd.DataFrame([data])

    cat_cols = ['job', 'category', 'merchant', 'state']
    for col in cat_cols:
        df[col] = df[col].map(target_encoders[col])
        df[col] = df[col].fillna(target_encoders[col].mean())

    df = df[FEATURE_COLS]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=FEATURE_COLS)

    pred_ae = autoencoder.predict(df_scaled, verbose=0)
    recon_error = float(np.mean(np.power(df_scaled.values - pred_ae, 2)))

    df_scaled['recon_error'] = recon_error
    fraud_prob = float(xgb_model.predict_proba(df_scaled)[:, 1][0])
    is_fraud = int(fraud_prob > best_threshold)

    return "Fraud" if is_fraud else "Legitimate", round(fraud_prob * 100, 2), round(recon_error, 4)

# -------------------------
# SESSION STATE
# -------------------------
if 'loaded_tx' not in st.session_state:
    st.session_state.loaded_tx = None

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
        st.session_state.loaded_tx = sample_df.sample(1).iloc[0]
with c2:
    if st.button("Load Fraud Case"):
        st.session_state.loaded_tx = sample_df[sample_df['is_fraud'] == 1].sample(1).iloc[0]
with c3:
    if st.button("Load Legit Case"):
        st.session_state.loaded_tx = sample_df[sample_df['is_fraud'] == 0].sample(1).iloc[0]

tx = st.session_state.loaded_tx

if tx is not None:
    st.markdown(f"""
    <div class="tx-info">
        Merchant: <span>{str(tx['merchant'])[:30]}</span> &nbsp;|&nbsp;
        Amount: <span>${tx['amt']:.2f}</span> &nbsp;|&nbsp;
        Category: <span>{tx['category']}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)

cat_options = list(target_encoders['category'].index)
state_options = list(target_encoders['state'].index)
job_options = list(target_encoders['job'].index)[:100]
merchant_options = list(target_encoders['merchant'].index)[:100]

with col1:
    st.markdown('<div class="section-header">Financial Info</div>', unsafe_allow_html=True)
    amt = st.number_input("Amount ($)", value=float(tx['amt']) if tx is not None else 0.0, min_value=0.0, format="%.2f")
    default_cat = tx['category'] if tx is not None and tx['category'] in cat_options else cat_options[0]
    category = st.selectbox("Category", cat_options, index=cat_options.index(default_cat))
    default_merch = tx['merchant'] if tx is not None and tx['merchant'] in merchant_options else merchant_options[0]
    merchant = st.selectbox("Merchant", merchant_options, index=merchant_options.index(default_merch) if default_merch in merchant_options else 0)

with col2:
    st.markdown('<div class="section-header">Customer Info</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ['M', 'F'], index=0 if tx is None or tx['gender'] == 'M' else 1)
    default_state = tx['state'] if tx is not None and tx['state'] in state_options else state_options[0]
    state = st.selectbox("State", state_options, index=state_options.index(default_state))
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
    dist = float(((tx['lat'] - tx['merch_lat'])**2 + (tx['long'] - tx['merch_long'])**2)**0.5) if tx is not None else 0.0
    distance = st.number_input("Distance to Merchant", value=dist, min_value=0.0, format="%.4f")
    tx_velocity = st.number_input("Total Transactions (this card)", value=10, min_value=1)
    amt_deviation = st.number_input("Amount Deviation", value=0.0, min_value=0.0, format="%.4f")
    tx_per_day = st.number_input("Transactions Today", value=1, min_value=1)

st.markdown("---")

if st.button("Analyze Transaction"):
    with st.spinner("Running models..."):
        final, prob, recon = predict(
            amt, category, merchant, gender, state, city_pop,
            job, hour, day_of_week, age, distance,
            tx_velocity, amt_deviation, tx_per_day
        )

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
