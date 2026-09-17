import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("StudyHrs.pkl")

# Title
st.title("🎓 Student Result Predictor")

st.write("Enter the number of hours you study per day.")

# Input
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=12.0,
    value=5.0,
    step=0.5
)

# Button
if st.button("Predict Result"):

    # Create input DataFrame
    input_data = pd.DataFrame(
        [[study_hours]],
        columns=["StudyHours"]
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    if prediction == 1:
        st.success("✅ PASS")
    else:
        st.error("❌ FAIL")

    st.write(
        "Probability of Pass:",
        round(probability * 100, 2),
        "%"
    )
