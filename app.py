
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM WHITE THEME / CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #1f2937;
    }

    [data-testid="stHeader"] {
        background-color: #ffffff;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #172554;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-heading {
        color: #1e3a8a;
        font-size: 19px;
        font-weight: 650;
        padding-bottom: 8px;
        border-bottom: 1px solid #e5e7eb;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    div[data-testid="stForm"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 22px;
        background-color: #ffffff;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #1d4ed8;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1e40af;
        color: white;
    }

    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 16px;
        border-radius: 10px;
    }

    .footer {
        color: #64748b;
        text-align: center;
        font-size: 13px;
        padding-top: 30px;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD SAVED MODEL FILES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "heart_disease_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("❤️ Heart Disease Prediction")

st.markdown(
    '<p class="subtitle">'
    'Machine learning-based heart disease risk prediction using '
    'a Support Vector Machine (SVM) model.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">Patient Information</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):

    # Personal and clinical measurements
    st.markdown("### Basic & Clinical Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=40
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=120
        )

    with col2:
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0,
            max_value=700,
            value=200
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )

    with col3:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar",
            options=[0, 1],
            format_func=lambda x: (
                "Greater than 120 mg/dL" if x == 1
                else "120 mg/dL or below"
            )
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=-2.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )

    st.divider()

    # Categorical measurements
    st.markdown("### Patient Health Details")

    col4, col5, col6 = st.columns(3)

    with col4:
        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ASY", "ATA", "NAP", "TA"],
            format_func=lambda x: {
                "ASY": "Asymptomatic",
                "ATA": "Atypical Angina",
                "NAP": "Non-Anginal Pain",
                "TA": "Typical Angina"
            }[x]
        )

    with col5:
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["No", "Yes"]
        )

    with col6:
        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "Predict Heart Disease"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    # Create all 15 features using the training encodings
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_M": 1 if sex == "Male" else 0,

        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain == "TA" else 0,

        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,

        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,

        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0
    }

    # Maintain the exact training feature order
    input_df = pd.DataFrame(
        [[input_data[col] for col in feature_columns]],
        columns=feature_columns
    )

    numeric_features = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak"
    ]

    # Apply the same scaling approach used during training
    scaler_features = getattr(scaler, "n_features_in_", None)

    try:
        if scaler_features == len(feature_columns):
            # Scaler was fitted on all 15 features
            input_scaled = scaler.transform(input_df)

        elif scaler_features == len(numeric_features):
            # Scaler was fitted on only 6 numerical features
            input_scaled = input_df.copy()

            input_scaled[numeric_features] = scaler.transform(
                input_df[numeric_features]
            )

        else:
            st.error(
                "Scaler feature count does not match the expected "
                "6 numerical or 15 total features. Check your "
                "training preprocessing."
            )
            st.stop()

        prediction = model.predict(input_scaled)[0]

        st.divider()
        st.markdown("## Prediction Result")

        if prediction == 1:
            st.error(
                "Prediction: Heart Disease Detected"
            )

            st.markdown(
                "The model predicts the presence of heart disease "
                "based on the supplied input values."
            )

        else:
            st.success(
                "Prediction: No Heart Disease Detected"
            )

            st.markdown(
                "The model predicts the absence of heart disease "
                "based on the supplied input values."
            )

        st.info(
            "This is an educational machine learning prediction, "
            "not a medical diagnosis. Consult a qualified healthcare "
            "professional for medical assessment."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.markdown("## About This Project")

col7, col8, col9 = st.columns(3)

with col7:
    st.metric("Algorithm", "SVM")

with col8:
    st.metric("Test Accuracy", "84.78%")

with col9:
    st.metric("Input Features", "15")

st.markdown("""
This application uses a Support Vector Machine classifier trained
on a heart disease dataset. It accepts clinical measurements and
categorical patient information to generate a binary prediction.
""")

st.markdown(
    '<div class="footer">'
    'Heart Disease Prediction | Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
