import streamlit as st
import pickle
import pandas as pd

# Load the trained model
# Load the trained model using pickle
with open('E:/Project  1/One/accuracy.pkl', 'rb') as file:
    model = pickle.load(file)


print(model)

# Streamlit app
def prediction_app():
    st.title("Loan Approval Prediction")

    # Input form
    st.sidebar.header("User Input")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Marital Status", ["Married", "Single"])
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    credit_history = st.sidebar.selectbox("Credit History", [0, 1])

    # Make prediction
    if st.sidebar.button("Predict"):
        # Create a dictionary with user input
        user_input = {
            'Gender': gender,
            'Married': married,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'Education': education,
            'Credit_History': credit_history
        }

        # Convert the user input to a DataFrame
        input_df = pd.DataFrame([user_input])

        # Preprocess the input data (you may need to adjust this based on your preprocessing steps during training)
        # For example, convert categorical variables to numerical, handle missing values, etc.

        # Make prediction
        prediction = model.predict(input_df)

        # Display the prediction
        st.success(f"The loan is {'Approved' if prediction[0] == 1 else 'Not Approved'}.")

if __name__ == '__main__':
    prediction_app()
