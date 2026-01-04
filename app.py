import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/final_model.pkl")

st.title("Earnings Manipulation Detector")

st.write("Enter Beneish Ratios:")

DSRI = st.number_input("DSRI")
GMI = st.number_input("GMI")
AQI = st.number_input("AQI")
SGI = st.number_input("SGI")
DEPI = st.number_input("DEPI")
SGAI = st.number_input("SGAI")
ACCR = st.number_input("ACCR")
LEVI = st.number_input("LEVI")

if st.button("Check Manipulation"):
    input_df = pd.DataFrame([[DSRI, GMI, AQI, SGI, DEPI, SGAI, ACCR, LEVI]],
        columns=["DSRI","GMI","AQI","SGI","DEPI","SGAI","ACCR","LEVI"])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠ Likely Manipulator (Risk: {prob:.2f})")
    else:
        st.success(f"✅ Likely Non-Manipulator (Risk: {prob:.2f})")
