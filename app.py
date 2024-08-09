import streamlit as st
import pickle
import numpy as np

# Load the trained SVC model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function for prediction
def predict(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction

# Streamlit application interface
st.title("Loan Approval Prediction")

# Sidebar input for user data
st.sidebar.header("Enter the Applicant's Details:")

# Input fields for all features
gender = st.sidebar.selectbox("Gender", (0, 1))  # Assuming 0: Male, 1: Female
married = st.sidebar.selectbox("Married", (0, 1))  # Assuming 0: No, 1: Yes
dependents = st.sidebar.selectbox("Dependents", (0, 1, 2, 3))  # 0:0, 1:1, 2:2, 3:3+
education = st.sidebar.selectbox("Education", (0, 1))  # Assuming 0: Graduate, 1: Not Graduate
self_employed = st.sidebar.selectbox("Self Employed", (0, 1))  # Assuming 0: No, 1: Yes
applicant_income = st.sidebar.number_input("Applicant Income", value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", value=0)
loan_amount = st.sidebar.number_input("Loan Amount", value=0)
loan_amount_term = st.sidebar.number_input("Loan Amount Term (in days)", value=360)
credit_history = st.sidebar.selectbox("Credit History", (0, 1))  # Assuming 0: No, 1: Yes
property_area = st.sidebar.selectbox("Property Area", (0, 1, 2))  # 0: Urban, 1:Semiurban, 2: Rural

# Collect input data into a list
input_data = [gender, married, dependents, education, self_employed, 
              applicant_income, coapplicant_income, loan_amount, 
              loan_amount_term, credit_history, property_area]

# Predict button
if st.sidebar.button("Predict"):
    result = predict(input_data)
    if result[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Not Approved.")

