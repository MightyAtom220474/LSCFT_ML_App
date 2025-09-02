import streamlit as st
import pandas as pd

st.subheader("Machine Learning Inputs")

with st.sidebar:

    st.subheader("Your Data")

    uploaded_file = st.file_uploader('Please select a csv file containing the data you want to ' \
                'analyse',type='csv')
    
    uploaded_df = pd.read(uploaded_file)

    column_headers = [column[0] for column in uploaded_df.description] 

    field_of_interest = st.multiselect('Please select the data item we are trying to predict',
                   options=column_headers,help='Please select just one value'
                   ,max_selections=1)
    
    train_percent_input = st.number_input("Please select the % of data to be used " \
                                    " to train the models",
                                    min_value=0, max_value=25, step=1,
                                    value=200,help='Too large = less reliable'\
                                    ' with new data, Too small = less data to '\
                                    'learn from so less reliable. Typical '\
                                    'valuea are between 20% and 30%')

#file_data = uploaded_file.read()

st.write(f'File {uploaded_file.name} has been successfully uploaded')

st.write(f'The thing we are trying to predict is {field_of_interest}')

st.write(f'We are using {train_percent_input}% of the data to train the '\
         'models')

# Parameters used to run the model
train_pc = train_percent_input / 100

