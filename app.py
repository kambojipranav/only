import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# -------------------- LANGUAGE SUPPORT --------------------
lang_options = {'English': 'en', 'Hindi': 'hi', 'Telugu': 'te'}
language = st.sidebar.selectbox("🌐 Choose Language", list(lang_options.keys()))
lang_code = lang_options[language]

translations = {
    'en': {
        'title': "Chronic Kidney Disease Predictor",
        'subtitle': "Enter patient details below to assess the risk",
        'age': "Age", 'bp': "Blood Pressure", 'sg': "Specific Gravity",
        'al': "Albumin", 'su': "Sugar", 'bgr': "Blood Glucose Random", 
        'bu': "Blood Urea", 'sc': "Serum Creatinine", 'sod': "Sodium", 
        'pot': "Potassium", 'hemo': "Hemoglobin", 'pcv': "Packed Cell Volume", 
        'wc': "WBC Count", 'rc': "RBC Count", 'rbc': "Red Blood Cells", 
        'pc': "Pus Cell", 'pcc': "Pus Cell Clumps", 'ba': "Bacteria", 
        'htn': "Hypertension", 'dm': "Diabetes Mellitus", 'cad': "Coronary Artery Disease", 
        'appet': "Appetite", 'pe': "Pedal Edema", 'ane': "Anemia",
        'predict': "🔍 Predict", 'risk': "Kidney Disease Risk",
        'low': "🟢 Low Risk — Regular monitoring advised.",
        'med': "🟡 Moderate Risk — Recommend follow-up tests.",
        'high': "🔴 High Risk — Urgent medical attention recommended.",
        'precaution': "📝 Precaution: Manage diabetes and blood pressure, follow a kidney-friendly diet, stay active, and avoid smoking."
    },
    'hi': {
        'title': "क्रॉनिक किडनी रोग पूर्वानुमान",
        'subtitle': "जोखिम का आकलन करने के लिए नीचे विवरण भरें",
        'age': "आयु", 'bp': "रक्तचाप", 'sg': "विशिष्ट गुरुत्वाकर्षण",
        'al': "एलबुमिन", 'su': "शुगर", 'bgr': "ब्लड ग्लूकोज", 
        'bu': "ब्लड यूरिया", 'sc': "सीरम क्रिएटिनिन", 'sod': "सोडियम", 
        'pot': "पोटेशियम", 'hemo': "हीमोग्लोबिन", 'pcv': "पैक्ड सेल वॉल्यूम", 
        'wc': "डब्ल्यूबीसी गिनती", 'rc': "आरबीसी गिनती", 'rbc': "लाल रक्त कोशिकाएं", 
        'pc': "पुस सेल", 'pcc': "पुस क्लंप", 'ba': "बैक्टीरिया", 
        'htn': "हाई ब्लड प्रेशर", 'dm': "मधुमेह", 'cad': "कोरोनरी आर्टरी डिजीज", 
        'appet': "भूख", 'pe': "पैरों में सूजन", 'ane': "अनीमिया",
        'predict': "🔍 पूर्वानुमान करें", 'risk': "किडनी रोग जोखिम",
        'low': "🟢 कम जोखिम — नियमित निगरानी की सिफारिश की जाती है।",
        'med': "🟡 मध्यम जोखिम — अतिरिक्त परीक्षण की सिफारिश की जाती है।",
        'high': "🔴 उच्च जोखिम — तत्काल चिकित्सा सलाह आवश्यक है।",
        'precaution': "📝 सावधानी: मधुमेह और रक्तचाप को नियंत्रित करें, किडनी के अनुकूल आहार लें, सक्रिय रहें और धूम्रपान से बचें।"
    },
    'te': {
        'title': "దీర్ఘకాల కిడ్నీ వ్యాధి అంచనా",
        'subtitle': "రిస్క్ అంచనా వేసేందుకు వివరాలు నమోదు చేయండి",
        'age': "వయస్సు", 'bp': "బ్లడ్ ప్రెజర్", 'sg': "స్పెసిఫిక్ గ్రావిటీ",
        'al': "అల్బ్యూమిన్", 'su': "షుగర్", 'bgr': "బ్లడ్ గ్లూకోజ్", 
        'bu': "బ్లడ్ యూరియా", 'sc': "సీరమ్ క్రియాటినిన్", 'sod': "సోడియం", 
        'pot': "పొటాషియం", 'hemo': "హిమోగ్లోబిన్", 'pcv': "ప్యాక్డ్ సెల్ వాల్యూమ్", 
        'wc': "డబ్ల్యూబీసీ కౌంట్", 'rc': "ఆర్బీసీ కౌంట్", 'rbc': "ఎర్ర రక్తకణాలు", 
        'pc': "పస్ సెల్", 'pcc': "పస్ క్లంప్స్", 'ba': "బాక్టీరియా", 
        'htn': "హై బీపీ", 'dm': "డయాబెటిస్", 'cad': "కోరోనరీ ఆర్టరీ వ్యాధి", 
        'appet': "ఆహారం ఇష్టం", 'pe': "కాళ్ళ వాపు", 'ane': "అనీమియా",
        'predict': "🔍 అంచనా వేయి", 'risk': "కిడ్నీ వ్యాధి ప్రమాదం",
        'low': "🟢 తక్కువ ప్రమాదం — సాధారణంగా పర్యవేక్షణ అవసరం.",
        'med': "🟡 మధ్యస్థ ప్రమాదం — పరీక్షలు చేయించుకోవాలి.",
        'high': "🔴 ఎక్కువ ప్రమాదం — వెంటనే వైద్య సేవలు అవసరం.",
        'precaution': "📝 జాగ్రత్తలు: షుగర్ మరియు బీపీ నియంత్రించండి, కిడ్నీకి అనుకూలమైన ఆహారం తీసుకోండి, చురుకుగా ఉండండి మరియు పొగతాగడం నివారించండి."
    }
}

def t(key):
    return translations[lang_code].get(key, key)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="CKD Predictor", page_icon="🧬", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #00c6ff, #0072ff);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3, h4 {
            color: #fff7e6;
            text-align: center;
        }
        .stButton>button {
            background: linear-gradient(to right, #00b894, #00cec9);
            color: white;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.5rem 1rem;
        }
        .stSlider > div > div > div {
            background: #ffffff33;
        }
        .stSelectbox > div > div {
            background-color: #ffffff22;
        }
        .stMetricLabel, .stMetricValue {
            color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- DATA PREP --------------------
df = pd.read_csv("kidney_disease.csv")
df.dropna(inplace=True)

encode_map = {
    'yes': 1, 'no': 0,
    'normal': 1, 'abnormal': 0,
    'present': 1, 'notpresent': 0,
    'good': 1, 'poor': 0
}
df.replace(encode_map, inplace=True)

X = df[[
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
    'htn', 'dm', 'cad', 'appet', 'pe', 'ane']]
y = LabelEncoder().fit_transform(df['classification'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use GridSearchCV to tune RandomForestClassifier
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
model = grid.best_estimator_

# -------------------- FORM --------------------
st.title(t("title"))
st.caption(t("subtitle"))

with st.form("prediction_form"):
    age = st.slider(t("age"), 1, 100, 45)
    bp = st.slider(t("bp"), 50, 180, 80)
    sg = st.selectbox(t("sg"), [1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.slider(t("al"), 0, 5, 1)
    su = st.slider(t("su"), 0, 5, 0)
    bgr = st.number_input(t("bgr"), 70, 490, 120)
    bu = st.number_input(t("bu"), 1.0, 200.0, 50.0)
    sc = st.number_input(t("sc"), 0.1, 15.0, 1.2)
    sod = st.number_input(t("sod"), 120.0, 150.0, 135.0)
    pot = st.number_input(t("pot"), 2.0, 7.0, 4.5)
    hemo = st.number_input(t("hemo"), 5.0, 20.0, 13.5)
    pcv = st.number_input(t("pcv"), 20.0, 60.0, 41.0)
    wc = st.number_input(t("wc"), 2000.0, 20000.0, 8500.0)
    rc = st.number_input(t("rc"), 2.0, 6.5, 5.2)
    rbc = st.selectbox(t("rbc"), ["normal", "abnormal"])
    pc = st.selectbox(t("pc"), ["normal", "abnormal"])
    pcc = st.selectbox(t("pcc"), ["present", "notpresent"])
    ba = st.selectbox(t("ba"), ["present", "notpresent"])
    htn = st.selectbox(t("htn"), ["yes", "no"])
    dm = st.selectbox(t("dm"), ["yes", "no"])
    cad = st.selectbox(t("cad"), ["yes", "no"])
    appet = st.selectbox(t("appet"), ["good", "poor"])
    pe = st.selectbox(t("pe"), ["yes", "no"])
    ane = st.selectbox(t("ane"), ["yes", "no"])

    submitted = st.form_submit_button(t("predict"))

if submitted:
    input_data = pd.DataFrame([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr,
                                 bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm,
                                 cad, appet, pe, ane]],
                               columns=X.columns)
    input_data.replace(encode_map, inplace=True)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader("🔔 " + t("risk"))
    st.metric(t("risk"), f"{prediction:.2f}%")

    if prediction < 20:
        st.success(t("low"))
    elif prediction < 60:
        st.warning(t("med"))
    else:
        st.error(t("high"))
        st.info(t("precaution"))

