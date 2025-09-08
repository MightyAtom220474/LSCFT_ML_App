import streamlit as st
#from app_style import global_page_style


st.logo("https://lancsvp.org.uk/wp-content/uploads/2021/08/nhs-logo-300x189.png")

# with open("style.css") as css:
#     st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

#global_page_style('static/css/style.css')

st.subheader("Machine Learning Processor")
st.subheader("Turn your data into insights with just one upload")

st.markdown(
    """
    ### How It Works!!!

    1. Upload your data - Just choose a CSV file to get started.

    2. Pick out the field within your data that you want to be able to predict

    3. The app does the hard work - It prepares your data and runs several 
    machine learning models.

    4. See the results - The app shows you how well each model performs, 
    so you can quickly see which one works best for your data. It also shows you
    the Top 10 Features used to make the decision
    """
)

st.write("Head to the 'Machine Learning Processor' page to get started.")