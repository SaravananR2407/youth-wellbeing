import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OrdinalEncoder

# ------------------------------------------------------------
# 1. Load artifacts (cached for performance)
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    lr = joblib.load("model/linear_regression_model.pkl")
    ann = keras.models.load_model("model/ann_improved_best.keras")
    scaler = joblib.load("model/scaler.pkl")
    feature_cols = joblib.load("model/feature_columns.pkl")
    ordinal_encoders = joblib.load("model/ordinal_encoders.pkl")
    nominal_features = joblib.load("model/nominal_features.pkl")
    metrics = joblib.load("model/metrics.pkl")
    return lr, ann, scaler, feature_cols, ordinal_encoders, nominal_features, metrics

lr_model, ann_model, scaler, feature_columns, ord_encoders, nom_features, metrics = load_artifacts()

# ------------------------------------------------------------
# 2. Input form – you MUST adapt this to your 37 original features
# ------------------------------------------------------------
def create_input_form():
    st.subheader("📋 Youth Digital Behavior Profile")

    # ----- Numeric features (replace with your actual numeric columns) -----
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 12, 30, 18)
        screen_time = st.slider("Daily screen time (hours)", 0, 16, 6)
        social_media_usage = st.slider("Social media usage (hours/day)", 0, 12, 3)
        gaming_hours = st.slider("Gaming hours per day", 0, 10, 2)
    with col2:
        study_hours = st.slider("Study hours per day", 0, 12, 4)
        sleep_hours = st.slider("Sleep hours per night", 4, 12, 7)
        physical_activity = st.slider("Physical activity (hours/week)", 0, 20, 3)
        family_time = st.slider("Family time (hours/day)", 0, 8, 2)
    # Add all other numeric columns here ...

    # ----- Categorical features (categories must match training) -----
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    urban_rural = st.selectbox("Area", ["Urban", "Rural"])
    device_access = st.selectbox("Device access", ["Smartphone", "Laptop/Desktop", "Tablet", "Multiple"])
    field_of_study = st.selectbox("Field of study", ["STEM", "Humanities", "Business", "Arts", "Other"])
    cyberbullying_exposure = st.selectbox("Cyberbullying exposure", ["Never", "Rarely", "Sometimes", "Often"])
    adult_content_exposure = st.selectbox("Adult content exposure", ["Never", "Rarely", "Sometimes", "Often"])

    # Ordinal features – order must be exactly as in training
    family_income_level = st.select_slider("Family income level", options=["Low", "Middle", "High"])
    education_level = st.select_slider("Education level", options=["Dropout", "School", "Diploma", "Graduate", "Postgraduate", "PhD"])
    late_night_usage = st.select_slider("Late night usage", options=["Never", "Sometimes", "Often"])

    # Collect into a dictionary with the ORIGINAL column names
    raw_data = {
        "age": age,
        "screen_time": screen_time,
        "social_media_usage": social_media_usage,
        "gaming_hours": gaming_hours,
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "physical_activity": physical_activity,
        "family_time": family_time,
        # add all other numeric columns here...
        "gender": gender,
        "urban_rural": urban_rural,
        "device_access": device_access,
        "field_of_study": field_of_study,
        "cyberbullying_exposure": cyberbullying_exposure,
        "adult_content_exposure": adult_content_exposure,
        "family_income_level": family_income_level,
        "education_level": education_level,
        "late_night_usage": late_night_usage,
    }
    return raw_data

# ------------------------------------------------------------
# 3. Preprocessing (exactly as in notebook)
# ------------------------------------------------------------
def preprocess_input(raw_dict):
    df = pd.DataFrame([raw_dict])
    # Ordinal encoding
    for col, encoder in ord_encoders.items():
        df[col] = encoder.transform(df[[col]])
    # One-hot encoding
    df = pd.get_dummies(df, columns=nom_features, drop_first=False)
    # Reindex to match training columns
    df = df.reindex(columns=feature_columns, fill_value=0)
    # Scale
    X_scaled = scaler.transform(df)
    return X_scaled

# ------------------------------------------------------------
# 4. Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Youth Wellbeing Predictor", layout="wide")
st.title("🌱 Digital Wellbeing Prediction for Malaysian Youth")
st.markdown("Enter the digital behavior profile to predict the **Wellbeing Index** (0–100).")

user_data = create_input_form()

col1, col2 = st.columns(2)
with col1:
    predict_lr = st.button("🔮 Predict with Linear Regression")
with col2:
    predict_ann = st.button("🧠 Predict with Improved ANN")

if predict_lr or predict_ann:
    with st.spinner("Preprocessing inputs and predicting..."):
        X_input = preprocess_input(user_data)
        if predict_lr:
            pred = lr_model.predict(X_input)[0]
            model_name = "Linear Regression"
            mae = metrics["lr"]["MAE"]
        else:
            pred = ann_model.predict(X_input, verbose=0)[0][0]
            model_name = "Improved ANN"
            mae = metrics["imp"]["MAE"]
    st.success(f"✅ **Predicted Wellbeing Index** = {pred:.2f} / 100")
    st.caption(f"Model: {model_name} | Typical prediction error (MAE) = ±{mae:.2f}")
    with st.expander("📊 Model performance on test set"):
        st.dataframe(pd.DataFrame(metrics).T.round(4))