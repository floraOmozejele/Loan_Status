import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('Loan_Data.csv')
data['Dependents'] = pd.to_numeric(data['Dependents'], errors = 'coerce')

st.markdown("<h1 style = 'color: #EFBC9B; text-align: center; font-size: 60px; font-family: Georgia'>LOAN PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: italic'>BUILT BY FLORA </h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)

# #add image
st.image('pngwing.com (17).png')

st.markdown("<h2 style = 'color: #A34343; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)


st.markdown("<p>Banks and financial institutions receive numerous loan applications from customers seeking financial assistance for various purposes such as purchasing a home, starting a business, or funding education. However, approving loans without proper assessment of creditworthiness can lead to financial losses due to defaults. The objective is to develop a predictive model that evaluates the credit risk associated with each loan applicant and predicts whether they are qualified to receive a loan or not. This model will analyze various features or attributes of the customer and their financial history to make an informed decision</p>", unsafe_allow_html = True)

st.sidebar.image('pngwing.com (18).png',caption = 'Welcome User')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width = True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)



# primaryColor="#FF4B4B"  
# backgroundColor="#99E6FF"
# secondaryBackgroundColor="#CCFCFF"
# textColor="#331133"
# font="sans serif"


st.sidebar.subheader('User Input Variables')
sel_columns = ['ApplicantIncome', 'LoanAmount', 'CoapplicantIncome', 'Dependents',
                'Property_Area', 'Credit_History', 'Loan_Amount_Term' , 'Loan_Status']

app_income = st.sidebar.number_input('Applicant Income', data['ApplicantIncome'].min(), data['ApplicantIncome'].max())
loan_amt = st.sidebar.number_input('Loan Amount', data['LoanAmount'].min(), data['LoanAmount'].max())
coapp_income = st.sidebar.number_input('CoApplicant Income', data['CoapplicantIncome'].min(), data['CoapplicantIncome'].max())
dep = st.sidebar.number_input('Dependents', data['Dependents'].min(), data['Dependents'].max())
prop_area = st.sidebar.selectbox('Property Area', data['Property_Area'].unique())
cred_hist = st.sidebar.number_input('Credit History', data['Credit_History'].min(), data['Credit_History'].max())
loan_amt_term = st.sidebar.number_input('Loan Amount Term', data['Loan_Amount_Term'].min(), data['Loan_Amount_Term'].max())

#users input
input_var = pd.DataFrame()
input_var['ApplicantIncome'] = [app_income]
input_var['LoanAmount'] = [loan_amt]
input_var['CoapplicantIncome'] = [coapp_income]
input_var['Dependents'] = [dep]
input_var['Property_Area'] = [prop_area]
input_var['Credit_History'] = [cred_hist]
input_var['Loan_Amount_Term'] = [loan_amt_term]

# in a situation where the loanamount was scaled you should save it to a new variable 
LoanAmount = int(input_var['LoanAmount'].values[0])

st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width = True)

# import the transformers
app_trans = joblib.load('ApplicantIncome_scaler.pkl')
co_app_trans = joblib.load('CoapplicantIncome_scaler.pkl')
prop_trans = joblib.load('Property_Area_encoder.pkl')
# dep_trans =joblib.load('Dependents_encoder.pkl')

# transform the users input with the imported scalers
input_var['ApplicantIncome'] = app_trans.transform(input_var[['ApplicantIncome']])
input_var['CoapplicantIncome'] = co_app_trans.transform(input_var[['CoapplicantIncome']])
input_var['Property_Area'] = prop_trans.transform(input_var[['Property_Area']])
#input_var['Dependents'] = dep_trans.transform(input_var[['Dependents']])

# st.header('Transformed Input Variable')
# st.dataframe(input_var, use_container_width = True)


model = joblib.load('LoanModel.pkl')
predict = model.predict(input_var)

if st.button('Check Your Loan Approval Status'):
    if predict[0] == 0:
        st.error(f"Unfortunately...Your Loan of {LoanAmount} dollar has been declined")
        st.image('pngwing.com (20).png', width = 300)
    else:
        st.success(f"Congratulations... Your loan of {LoanAmount} dollar has been approved. Pls come to the office to process your loan")
        st.image('pngwing.com (21).png', width = 300)
        st.balloons()
# import pandas as pd
# import streamlit as st
# import joblib
# import warnings
# warnings.filterwarnings('ignore')

# ds = pd.read_csv('Loan_Data.csv')

# st.markdown("<h1 style = 'color: #EFBC9B; text-align: center; font-size: 60px; font-family:Helvetica'>LOAN PREDICTOR APP</h1>", unsafe_allow_html = True)
# st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Build By Salmon FJM</h4>", unsafe_allow_html = True)
# st.markdown("<br>", unsafe_allow_html= True)

# #Add an Image 
# st.image('pngwing.com (10).png', caption = 'Built by Salmon')

# # Add Project Problem Statement 
# st.markdown("<h2 style = 'color: #A34343; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)

# st.markdown("<p>Leveraging Predictive Modeling for Fair Loan Qualification Based on Customer Features, this study aims to develop a robust predictive model that accurately determines loan eligibility by analyzing customer features, promoting efficiency and fairness in decision-making, and deploying it into a user-friendly system for real-time loan eligibility predictions.</p>", unsafe_allow_html= True)

# #SideBar Design
# st.sidebar.image('pngwing.com (11).png')

# st.sidebar.markdown('<br>', unsafe_allow_html = True)
# st.sidebar.markdown('<br>', unsafe_allow_html = True)
# st.divider()
# st.header('Loan ds')
# st.dataframe(ds, use_container_width = True)

# #User Inputs
# app_income = st.sidebar.number_input('Applicant Income', ds['ApplicantIncome'].min(), ds['ApplicantIncome'].max())
# co_app_income = st.sidebar.number_input('Co Applicant Income', ds['CoapplicantIncome'].min(), ds['CoapplicantIncome'].max())
# education = st.sidebar.selectbox('Education', ds['Education'].unique())
# gender = st.sidebar.selectbox('Gender', ds['Gender'].unique())
# loan_status = st.sidebar.selectbox('Loan_Status', ds['Loan_Status'].unique())
# married = st.sidebar.selectbox('Married', ds['Married'].unique())
# ppty_area = st.sidebar.selectbox('Property_Area', ds['Property_Area'].unique())
# self_employed = st.sidebar.selectbox('Self_Employed', ds['Self_Employed'].unique())



# #import transformers.... used to bring our scaler 
# app_income = joblib.load('ApplicantIncome_scaler.pkl')
# co_app_income =joblib.load('CoapplicantIncome_scaler.pkl')
# education = joblib.load('Education_encoder.pkl')
# gender = joblib.load('Gender_encoder.pkl')
# loan_status = joblib.load('Loan_Status_encoder.pkl')
# married = joblib.load('Married_encoder.pkl')
# ppty_area = joblib.load('Property_Area_encoder.pkl')
# self_employed = joblib.load('Self_Employed_encoder.pkl')
# model = joblib.load('LoanModel.pkl')


# #user input DataFrame
# user_input = pd.DataFrame()
# user_input['ApplicantIncome']= [app_income]
# user_input['CoapplicantIncome']= [co_app_income]
# user_input['Education']= [education]
# user_input['Gender']= [gender]
# user_input['Married']= [married]
# user_input['Property_Area']= [ppty_area]
# user_input['Self_Employed']= [self_employed]


# st.markdown("<br>", unsafe_allow_html= True)
# st.header('Input Variable')
# st.dataframe(user_input, use_container_width = True)

# st.header('Transformed Input Variable')
# st.dataframe(user_input, use_container_width = True)

# #user_input['ApplicantIncome'] =  app_income_scaler.transform(user_input[['ApplicantIncome']])
# user_input['CoapplicantIncome'] =  app_income.transform(user_input[['CoapplicantIncome']])
# user_input['Education'] =  education_encoder.transform(user_input[['Education']])
# user_input['Gender'] =  gender_encoder.transform(user_input[['Gender']])
# user_input['Married'] =  married_encoder.transform(user_input[['Married']])
# user_input['Property_Area'] =  propertyarea_encoder.transform(user_input[['Property_Area']])
# user_input['Self_Employed'] =  selfemployed_encoder.transform(user_input[['Self_Employed']])