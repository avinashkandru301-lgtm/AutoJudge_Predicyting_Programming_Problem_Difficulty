import streamlit as st
import numpy as np
import re
import joblib
from scipy.sparse import hstack

#  Load models
clf_class = joblib.load("reg_class.pkl")
reg_score = joblib.load("reg.pkl")
tfidf_char = joblib.load("tfidf_char.pkl")
tfidf_keywords = joblib.load("tfidf_keywords.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Feature extraction
def extract_numeric_features(text):
    return [
        len(text),
        len(re.findall(r'\d+', text)),
        len(re.findall(r'10\^\d+', text)),
        len(re.findall(r'1e\d+', text)),
        text.count("≤") + text.count("<="),
        text.count("log"),
        text.count("mod"),
        len(re.findall(r'O\(.+?\)', text))
    ]

def build_features(problem_text, input_text, output_text):
    full_text = (problem_text + " " + input_text + " " + output_text).lower()
    io_text = (input_text + " " + output_text).lower()

    num_main = np.array([extract_numeric_features(full_text)])
    num_io = np.array([extract_numeric_features(io_text)])

    numeric_all = scaler.transform(np.hstack([num_main, num_io]))

    X_char = tfidf_char.transform([full_text])
    X_keywords = tfidf_keywords.transform([full_text])

    return hstack([X_char, X_keywords, numeric_all])

#  Streamlit UI 
st.set_page_config(page_title="Problem Difficulty Predictor", layout="centered")

st.title("📊 Problem Difficulty Predictor")

st.markdown("Paste the problem details below and click **Predict**")

problem_text = st.text_area("📝 Problem Description", height=150)
input_text = st.text_area("📥 Input Description", height=120)
output_text = st.text_area("📤 Output Description", height=120)

if st.button("🚀 Predict"):
    if not problem_text.strip():
        st.error("Problem description is required.")
    else:
        X_new = build_features(problem_text, input_text, output_text)

       
        class_id = clf_class.predict(X_new)[0]
        class_name = le.inverse_transform([class_id])[0]

     
        score = float(reg_score.predict(X_new)[0])

        st.success("Prediction complete!")
        st.markdown(f"### 🧠 Predicted Class: **{class_name}**")
        st.markdown(f"### 🔢 Predicted Score: **{score:.2f}**")
