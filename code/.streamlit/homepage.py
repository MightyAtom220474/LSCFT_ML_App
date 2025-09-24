import streamlit as st
#from app_style import global_page_style


st.logo("https://lancsvp.org.uk/wp-content/uploads/2021/08/nhs-logo-300x189.png")

# with open("style.css") as css:
#     st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

#global_page_style('static/css/style.css')

st.subheader("Machine Learning Processor")
st.subheader("Turn your data into insights with just one upload")

st.title("Logistic Regression: A Simple Guide")

st.markdown("""
Logistic regression is a type of machine learning that predicts **yes/no outcomes** 
by estimating the **probability of an event** (e.g. admission, readmission).
""")

# --- Expandable detailed explanation ---
with st.expander("Learn more about how it works"):
    st.markdown("""
    ### 🔎 How does it work?
    1. Each patient has a set of **features** (e.g. age, previous attendances, risk factors).  
    2. Logistic regression combines these features into a single **score** (*z*).  
    3. This score is converted into a **probability between 0 and 1** using an “S-shaped” curve.  
       - Closer to 0 → unlikely.  
       - Closer to 1 → very likely.  
    4. By default, if the probability is **≥ 0.5**, the model predicts **YES** (e.g. admit).  

    ### 📊 How to interpret the results
    - **Probabilities** → likelihood of the outcome for a given patient.  
    - **Odds ratios** → effect of each feature:  
        - >1 = increases the odds.  
        - <1 = decreases the odds.  
    - **Confusion matrix** → where the model got predictions right vs wrong.  
    - **ROC curve** → how well the model separates outcomes across thresholds.  
    - **Calibration curve** → whether predicted probabilities match reality.  

    ### ⚠️ Important to remember
    - Logistic regression shows **associations, not causes**.  
    - It is a **support tool**, not a replacement for clinical judgment.  
    """)

st.markdown(
    """
    ### How To use the App!!!

    1. Upload your data - Just choose a CSV file to get started.

    2. Pick out the field within your data that you want to be able to predict

    3. The app does the hard work - It prepares your data and runs several 
    machine learning models.

    4. See the results - The app shows you how well the model performs, and
       the Top 10 Features used to make the decision

        """
)

st.write("Head to the 'Machine Learning Processor' page to get started.")