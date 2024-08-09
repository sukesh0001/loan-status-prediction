# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:39:31 2024

@author: A S U S
"""

import numpy as np
import pickle
import streamlit as st
loaded_model=pickle.load(open("C:/Users/A S U S/Documents/trained_model.sav",'rb'))
def loan_prediction(input_data):

    # Changing the input data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
        return 'You are eligible for loan'
    else:
        return 'Sorry!,you are not eligible'

def main():
    st.title('Bank loan prediction')
    # Get the input
    Gender = st.text_input('Gender')
    Married = st.text_input('Married')
    Dependents = st.text_input('Dependents')
    Education = st.text_input('Education')
    Self_Employed = st.text_input('Self_Employed')
    ApplicantIncome = st.text_input('ApplicantIncome')
    CoapplicantIncome = st.text_input('CoapplicantIncome')
    LoanAmount = st.text_input('LoanAmount')
    Loan_Amount_Term = st.text_input('Loan_Amount_Term')
    Credit_History = st.text_input('Credit_History')
    Property_Area = st.text_input('Property_Area')  # Corrected indentation

    approval = ''
    if st.button('Loan Result'):
        approval = loan_prediction([
            Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, Property_Area
        ])
    st.success(approval)

if __name__ == '__main__':
    main()






    

