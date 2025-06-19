import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import base64
import smtplib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="CKD Smart Predictor", page_icon="🧬", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        text-align: center;
        color: #fffde7;
    }
    .stButton>button {
        background: linear-gradient(to right, #00b894, #00cec9);
        color: white;
        border-radius: 12px;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        background-color: #ffffff22;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- DATA --------------------
df = pd.read_csv("kidney_cleaned_full.csv")
X = df.drop(columns=['id', 'classification'])
y = df['classification']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_scaled, y)

# -------------------- APP --------------------
st.markdown("""
    <h1>🌿 Chronic Kidney Disease Predictor</h1>
    <h4>Enter patient details to check risk and forward to doctor</h4>
""", unsafe_allow_html=True)

form = st.form("ckd_form")

columns = X.columns
user_data = {}

for col in columns:
    if df[col].dtype in [np.float64, np.int64]:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        val = int(df[col].mean())
        user_data[col] = form.slider(col.capitalize(), min_val, max_val, val)
    else:
        options = df[col].unique().tolist()
        user_data[col] = form.selectbox(col.capitalize(), options)

symptoms = form.text_area("📝 Doctor Notes (symptoms, conditions, lifestyle)")
email_to = form.text_input("📧 Doctor Email")

submit = form.form_submit_button("🔍 Predict & Email")

if submit:
    input_df = pd.DataFrame([user_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader(":bell: Prediction Result")
    st.metric("Kidney Disease Risk", f"{prediction:.2f}%")

    status = ""
    if prediction < 20:
        st.success("🟢 Low Risk — Regular monitoring advised.")
        status = "Low Risk"
    elif prediction < 60:
        st.warning("🟡 Moderate Risk — Recommend follow-up tests.")
        status = "Moderate Risk"
    else:
        st.error("🔴 High Risk — Urgent medical attention recommended.")
        status = "High Risk"

    if email_to:
        email_content = f"""
        Subject: CKD Prediction Report

        Dear Doctor,

        A patient has completed a CKD assessment via the smart predictor tool.

        Risk Level: {status} ({prediction:.2f}%)
        Symptoms/Notes:
        {symptoms}

        Patient Input Data:
        {user_data}

        Kindly advise the patient on next steps.

        Regards,
        CKD SmartApp
        """
        st.info(f"Email composed for {email_to}.")
        st.code(email_content)
        st.download_button("Download Email Text", email_content, file_name="ckd_email.txt")

# -------------------- FOOTER --------------------
st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <small>Made with ❤️ by Pranav Kamboji</small>
    </div>
""", unsafe_allow_html=True)
