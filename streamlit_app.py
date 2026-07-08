import os

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("HEART_DISEASE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon=":stethoscope:",
    layout="wide",
)

st.title("Heart Disease Risk Predictor")
st.caption("Interactive frontend for the deployed FastAPI prediction service")

with st.sidebar:
    st.header("API Connection")
    api_base_url = st.text_input("Prediction API base URL", value=DEFAULT_API_URL).rstrip("/")
    st.markdown("Expected endpoint: `/predict`")

st.markdown(
    "Enter patient details below and the app will send them to your deployed model API."
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=55)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest pain type (cp)", options=[0, 1, 2, 3], index=2)
    trestbps = st.number_input("Resting blood pressure", min_value=80, max_value=250, value=130)
    chol = st.number_input("Cholesterol", min_value=100, max_value=700, value=245)

with col2:
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2], index=1)
    thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise induced angina (exang)", options=[0, 1])
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)

with col3:
    slope = st.selectbox("Slope", options=[0, 1, 2, 3], index=1)
    ca = st.selectbox("Number of major vessels (ca)", options=[0, 1, 2, 3, 4], index=0)
    thal = st.selectbox("Thal", options=[0, 3, 6, 7], index=2)

payload = {
    "instances": [
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
    ]
}

with st.expander("Preview request payload"):
    st.json(payload)

if st.button("Predict risk", type="primary", use_container_width=True):
    try:
        response = requests.post(
            f"{api_base_url}/predict",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach prediction API: {exc}")
    else:
        prediction = result["predictions"][0]
        probability = None
        if result.get("probabilities"):
            probability = float(result["probabilities"][0])

        if prediction == 1:
            st.error("Model prediction: Higher heart disease risk")
        else:
            st.success("Model prediction: Lower heart disease risk")

        if probability is not None:
            st.metric("Predicted probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))

        st.subheader("Raw API response")
        st.json(result)

st.markdown(
    """
    Notes:
    - Use the public Render API URL in the sidebar when deploying this frontend separately.
    - For local testing, run the FastAPI service on `http://127.0.0.1:8000`.
    """
)
