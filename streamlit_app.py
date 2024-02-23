import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# Load the trained model
model = joblib.load('loan_default_prediction.joblib')


# Function to preprocess input data
def data_preprocess(age, income, loan_amount, credit_score, months_employed, credit_lines, interest_rate,
                    loan_term, dti_ratio, education, employment_type, marital_status, has_mortgage, loan_purpose,
                    has_cosigner, has_dependents):
    """
    Preprocesses the input data for prediction.

    Parameters:
        age (int): Age of the customer.
        income (int): Income of the customer.
        loan_amount (int): Loan amount requested by the customer.
        # Include descriptions for other parameters
        
    Returns:
        pd.DataFrame: Preprocessed data as a DataFrame.
    """
    age = [age]
    income = [income]
    loan_amount = [loan_amount]
    credit_score = [credit_score]
    months_employed = [months_employed]
    credit_lines = [credit_lines]
    interest_rate = [interest_rate]
    loan_term = [loan_term]
    dti_ratio = [dti_ratio]
    education = [education]
    employment_type = [employment_type]
    marital_status = [marital_status]
    has_mortgage = [has_mortgage]
    loan_purpose = [loan_purpose]
    has_cosigner = [has_cosigner]
    has_dependents = [has_dependents]

    data = pd.DataFrame({
        'Age': age, 'Income': income, 'LoanAmount': loan_amount,
        'CreditScore': credit_score, 'MonthsEmployed': months_employed,
        'NumCreditLines': credit_lines, 'InterestRate': interest_rate,
        'LoanTerm': loan_term, 'DTIRatio': dti_ratio, 'Education': education,
        'EmploymentType': employment_type, 'MaritalStatus': marital_status,
        'HasMortgage': has_mortgage, 'LoanPurpose': loan_purpose,
        'Has_Cosigner': has_cosigner, 'Has_Dependents': has_dependents
    })

    # Mapping to encode categorical columns
    data['Has_Cosigner'] = data['Has_Cosigner'].map({'No': 0, 'Yes': 1})
    data['HasMortgage'] = data['HasMortgage'].map({'No': 0, 'Yes': 1})
    data['Has_Dependents'] = data['Has_Dependents'].map({'No': 0, 'Yes': 1})

    data['Education'] = data['Education'].map({"Bachelor's": 0, "High School": 1, "Master's": 2, "PhD": 3})
    data['EmploymentType'] = data['EmploymentType'].map({"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3})
    data['MaritalStatus'] = data['MaritalStatus'].map({"Divorced": 0, "Married": 1, "Single": 2})
    data['LoanPurpose'] = data['LoanPurpose'].map({"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4})


    # Scale the input variables
    scaler = MinMaxScaler()
    col = ['InterestRate', 'DTIRatio', 'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines']
    data[col] = scaler.fit_transform(data[col])
    return data

def main():
    """
    Main function to run the Streamlit app.
    """
    # Streamlit interface
    st.title('This is a Loan Default Prediction App')

    # Streamlit app layout
    st.markdown("---")
    st.markdown("**This App will predict whether the customer will default on loan repayment or not**.")
    st.image("stream_image.jpg")
    st.divider()

    st.markdown("##### *Enter customer information below* :")

    # Get user input
    age = st.slider('Age', min_value=18, max_value=100, value=25, step=1)
    income = st.number_input('Income', value=50000, step=1000)
    loan_amount = st.number_input('Loan Amount', value=10000, step=100)
    credit_score = st.number_input('Credit Score', value=600, step=10)
    months_employed = st.number_input('Months Employed', value=12, step=1)
    credit_lines = st.number_input('Num Credit Lines', value=2, step=1)
    interest_rate = st.number_input('Interest Rate (%)', value=10, step=1)
    loan_term = st.slider('Loan Term (months)', min_value=12, max_value=360, value=120, step=12)
    dti_ratio = st.slider('DTI Ratio', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    education = st.selectbox("Education", ["Bachelor's", "Master's", 'High School', 'PhD'])
    employment_type = st.selectbox('Employment Type', ['Full-time', 'Unemployed', 'Self-employed', 'Part-time'])
    marital_status = st.selectbox('Marital Status', ['Divorced', 'Married', 'Single'])
    has_mortgage = st.radio("Has Mortgage", ['No', 'Yes'])
    loan_purpose = st.selectbox('Loan Purpose', ['Other', 'Auto', 'Business', 'Home', 'Education'])
    has_cosigner = st.radio("Has Cosigner", ['No', 'Yes'])
    has_dependents = st.radio("Has Dependents", ['No', 'Yes'])

    # Preprocess the user input
    user_data = data_preprocess(age, income, loan_amount, credit_score, months_employed, credit_lines, interest_rate,
                                loan_term, dti_ratio, education, employment_type, marital_status, has_mortgage,
                                loan_purpose, has_cosigner, has_dependents)
    
    st.markdown("---")
    st.markdown("##### *View your answer*")

    # Make prediction with loaded model
    prediction = model.predict(user_data)
    st.write("1 means Loan will default --//-- 0 means Loan will not default")
    result = 'Will Default? - Yes.' if prediction[0] == 1 else 'Will Default? - No.'
    st.write(prediction[0], result)


if __name__ == "__main__":
    main()