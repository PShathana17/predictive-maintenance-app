import streamlit as st
import pandas as pd
import joblib


# Load trained model and threshold
model = joblib.load("predictive_maintenance_model.pkl")
threshold = joblib.load("failure_threshold.pkl")


# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)


# Title
st.title("⚙️ Predictive Maintenance")
st.write(
    "Enter the machine sensor values to predict the risk of machine failure."
)


# Machine inputs
machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=310.0
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=320.0
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=3000,
    value=1200
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=65.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=200
)


# Prediction button
if st.button("Predict Machine Failure"):

    # Create input DataFrame
    new_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    # Create the same engineered features used during training
    new_data["Temperature_Difference"] = (
        new_data["Process temperature [K]"]
        - new_data["Air temperature [K]"]
    )

    new_data["Power_Proxy"] = (
        new_data["Rotational speed [rpm]"]
        * new_data["Torque [Nm]"]
    )

    new_data["ToolWear_Torque"] = (
        new_data["Tool wear [min]"]
        * new_data["Torque [Nm]"]
    )

    # Predict failure probability
    failure_probability = (
        model.predict_proba(new_data)[:, 1][0]
    )

    # Apply saved threshold
    prediction = int(
        failure_probability >= threshold
    )

    # Display probability
    st.subheader("Prediction Result")

    st.write(
        f"Failure Probability: "
        f"**{failure_probability * 100:.2f}%**"
    )

    # Display final prediction
    if prediction == 1:
        st.error("⚠️ Machine Failure Risk")
    else:
        st.success("✅ Normal Operation")