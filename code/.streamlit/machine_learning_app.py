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

import streamlit as st

st.markdown("""
## 🩺 How to Interpret the Model Results  

This tool uses a **logistic regression machine learning model**. Logistic regression predicts the **probability** that something is true (e.g. whether a patient has a certain outcome).  

---

### 1. 🔢 Probabilities and Predictions
- Each patient is given a **probability score** between 0 and 1.  
- If the probability is greater than **0.5**, the model predicts **Yes (Class 1)**.  
- If it is less than 0.5, the model predicts **No (Class 0)**.  

---

### 2. 📊 Feature Effects (Odds Ratios)
- The model looks at all the features in the dataset (e.g. age, BMI, smoking status).  
- Each feature has a **coefficient**, which we turn into an **odds ratio**.  
- **Odds ratio > 1** → the feature **increases** the likelihood of the outcome.  
- **Odds ratio < 1** → the feature **decreases** the likelihood of the outcome.  
- Example:  
  - Odds ratio = 1.5 → a one-unit increase makes the outcome **50% more likely**.  
  - Odds ratio = 0.7 → a one-unit increase makes the outcome **30% less likely**.  

---

### 3. ✅ Confusion Matrix
- Shows how often the model was correct or incorrect on the test data.  
- **True positives** → correctly predicted cases with the outcome.  
- **True negatives** → correctly predicted cases without the outcome.  
- **False positives / negatives** → cases where the model got it wrong.  
- This helps staff see the trade-off between **missed cases** and **over-calling**.  

---

### 4. 📈 Probability Distributions
- Shows how well the model separates the two groups.  
- Ideally, patients with the outcome cluster on the **right** (higher probabilities),  
  and patients without the outcome cluster on the **left** (lower probabilities).  

---

### 5. 🎯 Calibration Curve
- Checks if the probabilities are **trustworthy**.  
- Example: If the model says “70% probability”, then about 70% of those patients should really have the outcome.  

---

### 6. 🚦 ROC Curve (Discrimination Ability)
- Shows how well the model distinguishes between patients with and without the outcome.  
- The closer the curve is to the top left, the better.  
- **AUC value (Area Under Curve)**:  
  - 0.5 = no better than chance  
  - 0.7–0.8 = acceptable  
  - 0.8–0.9 = good  
  - >0.9 = excellent  

---

✅ **In summary**:  
- **Probabilities** → how confident the model is.  
- **Odds ratios** → which features push predictions up or down.  
- **Plots** → how reliable and accurate the model is.  
""")

##############################
## Specific Patient Example ##
##############################

# Select a patient from uploaded dataset
st.subheader("🔍 Individual Patient Explanation")

# Let user pick a row
patient_index = st.number_input("Select patient index", min_value=0, max_value=len(X_test)-1, value=0)

# Extract sample
if hasattr(X_test, "iloc"):
    sample = X_test.iloc[patient_index]
else:
    sample = pd.Series(X_test[patient_index], index=[f"X{i}" for i in range(X_test.shape[1])])

# Compute linear combination (z) and probability
coeffs = model.coef_[0]
intercept = model.intercept_[0]
z = intercept + np.dot(coeffs, sample)
prob = 1 / (1 + np.exp(-z))
pred_class = int(prob >= 0.5)

# Contribution of each feature
contributions = coeffs * sample
contrib_df = pd.DataFrame({
    "Feature": sample.index,
    "Value": sample.values,
    "Coefficient (β)": coeffs,
    "Contribution (β*x)": contributions
}).sort_values(by="Contribution (β*x)", ascending=False)

# Display results
st.write(f"**Predicted probability of outcome (Class 1): {prob:.2f}**")
st.write(f"**Predicted class:** {pred_class}")

st.markdown("### 🧮 Feature Contributions")
st.dataframe(contrib_df)

# Bar chart of contributions
st.bar_chart(contrib_df.set_index("Feature")["Contribution (β*x)"])
🔹 Step 2. Streamlit-friendly explanation text
python
Copy code
st.markdown("""
### 🧑‍⚕️ How to interpret the patient explanation  

- Each feature contributes to the final prediction through a **coefficient × value** calculation.  
- Positive contributions push the prediction **towards Class 1** (higher probability).  
- Negative contributions push the prediction **towards Class 0** (lower probability).  
- The size of the contribution shows how strongly that feature influenced the result.  

✅ Example:  
- If **Smoking = 1** and the coefficient for Smoking is **-0.9**, the contribution is negative, reducing the probability of the outcome.  
- If **Age = 70** and the coefficient for Age is **0.05**, the contribution is positive, pushing the probability higher.  

This allows clinicians to see **why the model made its prediction for this specific patient**.
""")


