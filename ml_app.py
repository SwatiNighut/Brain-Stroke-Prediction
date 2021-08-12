import pandas as pd
import numpy as np
import pickle
from pickle import load
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingClassifier

#model= pickle.load(open ('model.pkl','rb'))

import streamlit as st
from PIL import Image

mlmodel= pickle.load(open ('mlmodel.pkl','rb'))


# defining the function which will make the prediction using 
# the data which the user inputs



          
      
  
# this is the main function in which we define our webpage 
def main1():
      # giving the webpage a title
    st.title("Brain Stroke Prediction")
      
    
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Brain Stroke Predicator ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    
    # the data required to make the prediction
    
    gender_show = ('Female','Male')
    gender_choice = list(range(len(gender_show)))
    gender = st.selectbox("gender",gender_choice, format_func=lambda x: gender_show[x])


    age = st.number_input("Enter your age in years")

    
    hypertension_show = ('No','Yes')
    hypertension_choice = list(range(len(hypertension_show)))
    hypertension = st.selectbox("hypertension",hypertension_choice, format_func=lambda x: hypertension_show[x])

    
    
    heart_disease_show = ('No','Yes')
    heart_disease_choice = list(range(len(heart_disease_show)))
    heart_disease = st.selectbox("heart_disease",heart_disease_choice, format_func=lambda x: heart_disease_show[x])


    
    
    ever_married_show = ('No','Yes')
    ever_married_choice = list(range(len(ever_married_show)))
    ever_married = st.selectbox("ever_married",ever_married_choice, format_func=lambda x: ever_married_show[x])


    work_type_show = ('Private','self-employed','children','Govt_job','Never_worked')
    work_type_choice = list(range(len(work_type_show)))
    work_type = st.selectbox("work_type",work_type_choice, format_func=lambda x: work_type_show[x])


    
    
    Residence_type_show = ('Rural','Urban')
    Residence_type_choice = list(range(len(Residence_type_show)))
    Residence_type = st.selectbox("Residence_type",Residence_type_choice, format_func=lambda x: Residence_type_show[x])

    
    avg_glucose_level = st.number_input("Enter your glucose level")

    
    bmi = st.number_input("Enter your bmi in numbers")

    
    #smoking_status = st.selectbox("Smoking :",["never smoked","Unknown","formerly smoked","smokes"])

    smoking_status_show = ('formerly smoked','never smoked','smokes','Unknown')
    smoking_status_choice = list(range(len(smoking_status_show)))
    smoking_status = st.selectbox("smoking_status",smoking_status_choice, format_func=lambda x: smoking_status_show[x])

          
    
                 
          
            
    if st.button("Predict"):
        inputs= [[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]]

        prediction =mlmodel.predict(inputs)
              
        result = prediction
        if (result == 0):
            st.success('No chances of Brain Stroke')
        else:
            st.error('Chances of Brain Stroke')
    
     
main1()



