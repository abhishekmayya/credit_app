import streamlit as st
import pickle
import numpy as np

# Load your model (assuming it's a pickle file)
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def load_transform():
    with open("transform.pkl", "rb") as file:
        scaler = pickle.load(file)
    return scaler

# Define the app
st.title("Credit Card Default Prediction")

st.markdown("""
This application predicts whether a user will default on their credit card payment next month.
""")

prediction = None  # Initialize prediction outside the sidebar

with st.sidebar:
    # st.image("image.jpeg", use_column_width=True)  # Add image to sidebar

    # Collect user input
    st.header("Please enter the required information below.")

    # Create a dictionary for payment status options
    payment_status_dict = {
        -1: "No Due",
        1: "1 Month Delay",
        2: "2 Months Delay",
        3: "3 Months Delay",
        4: "4 Months Delay",
        5: "5 Months Delay",
        6: "6 Months Delay",
        7: "7 Months Delay",
        8: "8 Months Delay",
        9: "9 Months Delay or more",
    }

    LIMIT_BAL = st.number_input("Limit Balance", min_value=0, step=1000)

    st.markdown("Payment delay for the past months:")
    PAY_0 = st.selectbox("Last Month", options=list(payment_status_dict.values()))
    PAY_2 = st.selectbox("2nd Last Month", options=list(payment_status_dict.values()))
    PAY_3 = st.selectbox("3rd Last Month", options=list(payment_status_dict.values()))
    PAY_4 = st.selectbox("4th Last Month", options=list(payment_status_dict.values()))
    PAY_5 = st.selectbox("5th Last Month", options=list(payment_status_dict.values()))
    PAY_6 = st.selectbox("6th Last Month", options=list(payment_status_dict.values()))

    # Predict button below inputs
    if st.button("Predict"):

        # Convert user-selected options back to numerical values
        PAY_0 = list(payment_status_dict.keys())[list(payment_status_dict.values()).index(PAY_0)]
        PAY_2 = list(payment_status_dict.keys())[list(payment_status_dict.values()).index(PAY_2)]
        PAY_3 = list(payment_status_dict.keys())[list(payment_status_dict.values()).index(PAY_3)]
        PAY_4 = list(payment_status_dict.keys())[list(payment_status_dict.values()).index(PAY_4)]
        PAY_5 = list(payment_status_dict.keys())[list(payment_status_dict.values()).index(PAY_5)]
        PAY_6 = list(payment_status_dict.keys())[list(payment_status_dict.values()).index(PAY_6)]

        # Prepare input for the model
        model = load_model()
        scaler = load_transform()
        input_data = np.array([[LIMIT_BAL, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6]])
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)

# Display prediction below title and description
if prediction is not None:
    if prediction[0] == 1:
        st.error("The user is likely to default on their credit card payment next month.")
    else:
        st.success("The user is unlikely to default on their credit card payment next month.")