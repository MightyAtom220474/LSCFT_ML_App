import streamlit as st
#from app_style import global_page_style

st.set_page_config(
    page_title="Machine Learning Processor",
    layout="wide"
    )


st.logo("https://lancsvp.org.uk/wp-content/uploads/2021/08/nhs-logo-300x189.png")

# with open("style.css") as css:
#     st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

#global_page_style('static/css/style.css')

st.subheader("Machine Learning Processor")
st.subheader("Turn your data into insights with just one upload")

st.title("Machine Learning Processor")

st.markdown("""
This application allows you to build, evaluate and explain machine learning models
without writing any code.

Simply upload a CSV file, select the outcome you want to predict, and the application
will prepare the data, train a machine learning model and explain the results.

The application supports both automatic model selection and manual model selection,
making it suitable for beginners and advanced users alike.
""")

# --------------------------------------------------
# Model Overview
# --------------------------------------------------

with st.expander("Machine Learning Models Available"):

    st.markdown("""

### 🤖 Best Model (Automatic)

This option automatically tests multiple machine learning algorithms and selects
the best-performing model based on predictive performance.

The current AutoML process compares:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Histogram Gradient Boosting

This is the recommended option if you are unsure which model to choose.

---

### 📈 Logistic Regression

A simple and highly interpretable model.

Best suited to:

- Understanding relationships between variables
- Binary outcomes (Yes/No)
- Producing explainable results

Advantages:

- Fast
- Easy to understand
- Works well with structured data

---

### 🌳 Decision Tree

Creates a series of decision rules to make predictions.

Example:

- Age > 65?
- Previous DNA?
- Distance from clinic?

Advantages:

- Easy to visualise
- Captures non-linear relationships
- Easy to explain

---

### 🌲 Random Forest

Builds hundreds of decision trees and combines their predictions.

Advantages:

- Usually more accurate than a single decision tree
- Less prone to overfitting
- Handles large datasets well

---

### ⚡ XGBoost

One of the most powerful machine learning algorithms available.

Advantages:

- Often achieves excellent predictive performance
- Handles complex relationships
- Commonly used in machine learning competitions

---

### 🚀 LightGBM

A highly optimised gradient boosting algorithm.

Advantages:

- Extremely fast
- Handles large datasets efficiently
- Often performs similarly to XGBoost

---

### 🐱 CatBoost

A boosting algorithm designed specifically to handle categorical data well.

Advantages:

- Strong performance
- Minimal preprocessing required
- Excellent with healthcare datasets

---

### 📊 Histogram Gradient Boosting

A modern boosting technique designed for speed and scalability.

Advantages:

- Fast training
- Good predictive performance
- Efficient on larger datasets

---

### 🧠 Support Vector Machine

Attempts to find the optimal boundary between different outcomes.

Advantages:

- Powerful for certain datasets
- Effective with complex relationships

---

### 📉 Naive Bayes

A simple probabilistic model.

Advantages:

- Extremely fast
- Useful as a baseline model
- Works well on some classification problems

""")

# --------------------------------------------------
# Explainability
# --------------------------------------------------

with st.expander("How Predictions Are Explained"):

    st.markdown("""

The application includes Explainable AI (XAI) features to help you understand
why predictions are being made.

### Global Explanations

Global explanations identify:

- Which variables are most important overall
- Which factors generally increase risk
- Which factors generally decrease risk

### SHAP Feature Importance

SHAP (SHapley Additive exPlanations) identifies the features that have the
greatest impact on model predictions.

This allows you to understand:

- What factors drive outcomes
- Which predictors matter most
- How different variables interact

### Individual Prediction Explanations

For individual records, the application can show:

- Predicted probability
- Main factors increasing risk
- Main factors reducing risk
- SHAP Waterfall plots explaining how the prediction was generated

""")

# --------------------------------------------------
# Metrics
# --------------------------------------------------

with st.expander("Understanding Model Performance Metrics"):

    st.markdown("""

### Accuracy

How often the model was correct overall.

### Precision

Of all records predicted as positive:

How many were actually positive?

### Recall (Sensitivity)

Of all actual positive records:

How many did the model successfully identify?

### F1 Score

Balances Precision and Recall.

Often one of the most useful measures for healthcare datasets where outcomes
may be rare.

### ROC Curve

Shows how well the model separates the two outcome groups across different
thresholds.

### Confusion Matrix

Shows:

- True Positives
- False Positives
- True Negatives
- False Negatives

This helps identify where prediction errors occur.

""")

# --------------------------------------------------
# Imbalanced Data
# --------------------------------------------------

with st.expander("Handling Rare Outcomes"):

    st.markdown("""

Many healthcare outcomes are relatively uncommon.

Examples:

- Did Not Attend (DNA)
- Admission
- Readmission
- Deterioration
- Crisis events

The application includes several methods for handling imbalanced data:

### None

Standard machine learning approach.

### Class Weighting

Gives additional importance to rare outcomes.

### Oversampling

Creates additional examples of the minority class by duplicating records.

### SMOTE

Creates synthetic examples of the minority outcome.

This can significantly improve model performance when the event being predicted
is rare.

""")

# --------------------------------------------------
# User Guide
# --------------------------------------------------

st.markdown("""

## How to Use the App

### Step 1
Upload a CSV file.

**Important:**
Remove all patient identifiable information before uploading.

Examples:

✅ Age

✅ Gender

✅ Diagnosis

✅ Appointment Type

❌ NHS Number

❌ Name

❌ Address

❌ Postcode

### Step 2
Select the field you want to predict.

Examples:

- DNA (Yes/No)
- Admission (Yes/No)
- Readmission (Yes/No)

The target field should contain binary values such as:

- 0 / 1
- Yes / No
- Y / N
- True / False

### Step 3
Optional configuration

You can:

- Remove unwanted columns
- Select a machine learning model
- Adjust prediction thresholds
- Configure tree depth
- Handle imbalanced data

### Step 4
Run the model

The application will:

- Prepare the data
- Train the model
- Evaluate performance
- Generate explanations
- Display feature importance

### Step 5
Explore the results

Review:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Confusion Matrix
- SHAP Explanations
- Individual Prediction Explanations

""")

st.success(
    "🚀 Head to the 'Machine Learning Processor' page in the navigation menu to get started."
)

