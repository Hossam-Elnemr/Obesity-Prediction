import streamlit as st
import requests
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import joblib
import numpy as np
import PIL as Image
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title='Salary Classifier',
    page_icon=':gem:',
    initial_sidebar_state='collapsed'  # Collapsed sidebar
)

def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
model = joblib.load(open("diabetesClassifier", 'rb'))

def predict(age,Height,weight,FCVC,NCP, CH2O,FAF,THE,BMI,Gender_Female,Gender_Male,family_history_with_overweight_no,
            family_history_with_overweight_yes,FAVC_no,FAVC_yes ,CAEC_Always,CAEC_Frequently, CAEC_Sometimes, CAEC_no,
              SMOKE_no, SMOKE_yes, SCC_no, SCC_yes, CALC_Always, CALC_Frequently,CALC_Sometimes,CALC_no, MTRANS_Automobile, MTRANS_Bike,
                MTRANS_Motorbike ,MTRANS_Public_Transportation,MTRANS_Walking):

    features = np.array([age,Height,weight,FCVC,NCP, CH2O,FAF,THE,BMI,Gender_Female,Gender_Male,family_history_with_overweight_no,
            family_history_with_overweight_yes,FAVC_no,FAVC_yes ,CAEC_Always,CAEC_Frequently, CAEC_Sometimes, CAEC_no,
              SMOKE_no, SMOKE_yes, SCC_no, SCC_yes, CALC_Always, CALC_Frequently,CALC_Sometimes,CALC_no, MTRANS_Automobile, MTRANS_Bike,
                MTRANS_Motorbike ,MTRANS_Public_Transportation,MTRANS_Walking]).reshape(1, -1)
    
    prediction = model.predict(features)
    return prediction

with st.sidebar:
    choose = option_menu(None, ["Home", "Graphs", "About", "Contact"],
                         icons=['house', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose=='Home':
       st.write('# Salary Classifier')
       st.write('---')
       st.subheader('Enter your details to classify your salary')
       # User input
#---------------------------------------------------------------------------------------------------------------------------------------------

st.title("Obesity Risk Prediction")
# Numerical Inputs
age = st.number_input("Enter your age:", min_value=0)
Height = st.number_input("Enter your height (in meters):", min_value=0.0, format="%.2f")
weight = st.number_input("Enter your weight (in kg):", min_value=0.0, format="%.2f")
FCVC = st.slider("Frequency of vegetable consumption (1-3):", min_value=1, max_value=3)
NCP = st.slider("Number of main meals (1-4):", min_value=1, max_value=4)
CH2O = st.slider("Daily water consumption (liters, 1-3):", min_value=1.0, max_value=3.0, step=0.1)
FAF = st.slider("Physical activity frequency (hours per week, 0-3):", min_value=0.0, max_value=3.0, step=0.1)
THE = st.slider("Time using technology devices (hours per day, 0-24):", min_value=0.0, max_value=24.0, step=0.5)
BMI = st.number_input("Enter your BMI:", min_value=0.0, format="%.2f")

# Gender
gender = st.radio("Select your gender:", ("Male", "Female"))
Gender_Female = 1 if gender == "Female" else 0
Gender_Male = 1 if gender == "Male" else 0

# Family history with overweight
family_history = st.radio("Family history with overweight?", ("Yes", "No"))
family_history_with_overweight_yes = 1 if family_history == "Yes" else 0
family_history_with_overweight_no = 1 if family_history == "No" else 0

# Frequent consumption of high caloric food
FAVC = st.radio("Do you frequently consume high caloric food?", ("Yes", "No"))
FAVC_yes = 1 if FAVC == "Yes" else 0
FAVC_no = 1 if FAVC == "No" else 0

# Consumption of food between meals (CAEC)
CAEC = st.selectbox("Consumption of food between meals:", ("no", "Sometimes", "Frequently", "Always"))
CAEC_no = int(CAEC == "no")
CAEC_Sometimes = int(CAEC == "Sometimes")
CAEC_Frequently = int(CAEC == "Frequently")
CAEC_Always = int(CAEC == "Always")

# Smoking
SMOKE = st.radio("Do you smoke?", ("Yes", "No"))
SMOKE_yes = int(SMOKE == "Yes")
SMOKE_no = int(SMOKE == "No")

# Monitoring caloric consumption (SCC)
SCC = st.radio("Do you monitor your caloric intake?", ("Yes", "No"))
SCC_yes = int(SCC == "Yes")
SCC_no = int(SCC == "No")

# Alcohol consumption (CALC)
CALC = st.selectbox("Alcohol consumption frequency:", ("no", "Frequently", "Always"))
CALC_no = int(CALC == "no")
CALC_Frequently = int(CALC == "Frequently")
CALC_Sometimes = int(CALC == "Sometimes")
CALC_Always = int(CALC == "Always")

# Transportation method (MTRANS)
MTRANS = st.selectbox("Transportation method:", ("Walking", "Bike", "Motorbike", "Automobile", "Public_Transportation"))
MTRANS_Walking = int(MTRANS == "Walking")
MTRANS_Bike = int(MTRANS == "Bike")
MTRANS_Motorbike = int(MTRANS == "Motorbike")
MTRANS_Automobile = int(MTRANS == "Automobile")
MTRANS_Public_Transportation = int(MTRANS == "Public_Transportation")


# age,Height,weight,FCVC,NCP, CH2O,FAF,THE,BMI,Gender_Female,Gender_Male,family_history_with_overweight_no,
            # family_history_with_overweight_yes,FAVC_no,FAVC_yes ,CAEC_Always,CAEC_Frequently, CAEC_Sometimes, CAEC_no,
            #   SMOKE_no, SMOKE_yes, SCC_no, SCC_yes, CALC_Always, CALC_Frequently,CALC_no, MTRANS_Automobile, MTRANS_Bike,
                # MTRANS_Motorbike ,MTRANS_Public_Transportation,MTRANS_Walking
# Final input vector
input_data = np.array([[age, Height, weight, FCVC, NCP, CH2O, FAF, THE, BMI,
                        Gender_Female, Gender_Male, 
                        family_history_with_overweight_no, family_history_with_overweight_yes,
                        FAVC_no, FAVC_yes,
                        CAEC_Always, CAEC_Frequently, CAEC_Sometimes, CAEC_no,
                        SMOKE_no, SMOKE_yes,
                        SCC_no, SCC_yes,
                        CALC_Always, CALC_Frequently,CALC_Sometimes, CALC_no,
                        MTRANS_Automobile, MTRANS_Bike, MTRANS_Motorbike,
                        MTRANS_Public_Transportation, MTRANS_Walking]])

# st.write("User Input Vector:")
# st.write(input_data)


    #-------------------------------------------------------------------------------------------------------------------------------------------------
    # Predict the cluster
# sample_prediction = predict(*input_data[0])


# ['Insufficient_Weight' 'Normal_Weight' 'Obesity_Type_I' 'Obesity_Type_II'
#  'Obesity_Type_III' 'Overweight_Level_I' 'Overweight_Level_II']

if st.button("Predict", key="predict_button"):
    sample_prediction = predict(*input_data[0])
    
    if sample_prediction == 0:
        st.warning("NObeyesdad: Insufficient_Weight")
    elif sample_prediction == 1:
        st.success("NObeyesdad: Normal_Weight")
    elif sample_prediction == 2:
        st.warning("NObeyesdad: Obesity_Type_I")
    elif sample_prediction == 3:
        st.warning("NObeyesdad: Obesity_Type_II")
    elif sample_prediction == 4:
        st.warning("NObeyesdad: Obesity_Type_III")
    elif sample_prediction == 5:
        st.warning("NObeyesdad: Overweight_Level_I")
    elif sample_prediction == 6:
        st.warning("NObeyesdad: Overweight_Level_II")
    else:
        st.error("Unknown prediction value.")

        




elif choose=='About':
    st.write('# About Page')
    st.write('---')
    st.write("🎯💡 Welcome to Salary Classification Deployment! We specialize in providing advanced salary classification solutions that help individuals understand their income better. Our data-driven approach combines analytics, machine learning, and financial expertise to create customized salary classification models tailored to your needs. By implementing salary classification, you can gain insights into your income level, plan your finances effectively, and make informed decisions about your career and lifestyle. ✨🚀 Partner with us to unlock the power of salary classification and take control of your financial future. Contact us today to learn more. 📞📧")
    st.image("5355919-removebg-preview.png")


elif choose == "Contact":
    st.write('# Contact Us')
    st.write('---')
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        st.write('## Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') 
        Email=st.text_input(label='Please Enter Email')
        Message=st.text_input(label='Please Enter Your Message') 
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')



elif choose == 'Graphs':
    st.write('# Salary Classifier Graphs')
    st.write('---')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("## Female Vs Male workers Graph:")
    st.image("output.png")
    st.write("## Race vs Income Graph:")
    st.image("output2.png")
    st.write("## Income Vs Gender Graph:")
    st.image("output3.png")
    st.write("## Age Period Graph")
    st.image("output4.png")
    st.write("## Age period Vs Gender Graph")
    st.image("output5.png")
    st.write("## Age Period Vs Income Graph")
    st.image("output6.png")
    st.write("## Workclass Vs Income Graph")
    st.image("output7.png")
    st.write("## Education Vs Income Graph")
    st.image("output8.png")
    st.write("## Occupation Vs Income Graph")
    st.image("output9.png")
    st.write("## Working Hours Period Graph")
    st.image("output10.png")
    st.write("## Age Period Vs Working Hours Period Graph")
    st.image("output11.png")
    
    data = pd.read_csv('Obesity-Prediction/train_dataset.csv')
    # Create a DataFrame
    df = pd.DataFrame(data)
    
