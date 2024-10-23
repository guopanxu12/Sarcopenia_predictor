import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost_best_model.pkl')

# Define feature names used for the model
feature_names = [
    "Age", "BMI", "WC", "DBP", "Pulse", "UA", "HGB"
]

# Streamlit user interface
st.title("Sarcopenia Risk Prediction for Middle-aged and Elderly Cardiovascular Patients")
st.title("中老年心血管病患者肌少症风险预测")

# Age: categorical selection
age = st.selectbox("Age (年龄) <years>:", options=[1, 2, 3], format_func=lambda x: "45-64" if x == 1 else "65-74" if x == 2 else "≥75")

# BMI: categorical selection
bmi = st.selectbox("BMI (体重指数) <kg/m2>:", options=[1, 2, 3], format_func=lambda x: "<18.5" if x == 1 else "18.5-23.9" if x == 2 else "≥24")

# WC: numerical input
wc = st.number_input("WC (Waist Circumference) (腰围) <cm>:", min_value=20, max_value=180, value=80)

# DBP: numerical input
dbp = st.number_input("DBP (Diastolic Blood Pressure) (舒张压) <mmHg>:", min_value=40, max_value=160, value=80)

# Pulse: numerical input
pulse = st.number_input("Pulse (脉搏) <beats/min>:", min_value=30, max_value=180, value=75)

# UA: numerical input
ua = st.number_input("UA (Uric Acid) (尿酸) <μmol/L>:", min_value=100, max_value=800, value=350)
# Convert UA from μmol/L to mg/dL
ua = ua / 59.48

# HGB: numerical input
hgb = st.number_input("HGB (Hemoglobin) (血红蛋白) <g/L>:", min_value=50, max_value=280, value=130)
# Convert HGB from g/L to g/dL
hgb = hgb / 10

# Process inputs and make predictions
feature_values = [age, bmi, wc, dbp, pulse, ua, hgb]
features = np.array([feature_values], dtype=float)

if st.button("Predict (预测)"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class (预测类别):** {predicted_class}")
    st.write(f"**Prediction Probabilities (预测概率):** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            "According to the model, you may have a high risk of developing sarcopenia.\n"
            f"The predicted probability of developing sarcopenia is {probability:.1f}%.\n"
            "It is recommended that you see a doctor as soon as possible for a more detailed diagnosis and appropriate treatment.\n\n"
            "根据模型预测，您可能存在较高的肌少症发病风险。\n"
            f"模型预测的发病概率为 {probability:.1f}%。\n"
            "建议您尽快就医，以进行更详细的诊断和采取适当的治疗措施。"
        )
    else:
        advice = (
            "According to the model, your risk of sarcopenia is low.\n"
            f"The predicted probability of not having sarcopenia is {probability:.1f}%.\n"
            "It is recommended that you maintain a healthy lifestyle and monitor your health regularly. If you experience any symptoms, please see a doctor promptly.\n\n"
            "根据模型预测，您的肌少症风险较低。\n"
            f"模型预测的无肌少症概率为 {probability:.1f}%。\n"
            "建议您继续保持健康的生活方式，并定期观察健康状况。如有任何异常症状，请及时就医。"
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
