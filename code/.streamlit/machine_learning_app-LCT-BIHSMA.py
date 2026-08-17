import streamlit as st
import pandas as pd
import numpy as np
import machine_learning_new as ml
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve

##############################
##     Input Selectors      ##
##############################
from pathlib import Path
import pandas as pd
import streamlit as st

DEFAULT_FILE = Path(r"\\xlancashirecare.nhs.uk\shares\Applications$\BITeam\Machine Learning\inpatient_readmissions.csv")

with st.sidebar:
    
    st.subheader("Machine Learning Inputs")
    st.divider()

    st.subheader("Your Data")

    uploaded_file = st.file_uploader(
        "Please select a csv file containing the data you want to analyse",
        type="csv"
    )

    # Load uploaded file, otherwise use default file
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.success(f"File {uploaded_file.name} has been successfully uploaded")

    elif DEFAULT_FILE.exists():
        uploaded_df = pd.read_csv(DEFAULT_FILE)
        st.info(f"Using default file: {DEFAULT_FILE.name}")

    else:
        uploaded_df = None

    if uploaded_df is not None:

        column_headers = ['-- select an option --'] + uploaded_df.columns.tolist()

        st.divider()
        st.subheader("Field of Interest")

        field_of_interest = st.selectbox(
            "Please select the data item we are trying to predict",
            options=column_headers
        )

        st.divider()
        st.subheader("Columns to Remove")

        fields_to_remove = st.multiselect(
            "Select columns you want to exclude (optional):",
            options=column_headers,
            help="""
            Select any columns you are not interested in
            or that are interfering with your results.
            """
        )

        st.divider()
        st.subheader("Train Your Model")

        train_percent_input = st.number_input(
            "Please select the % of data to be used to train the model",
            min_value=0,
            max_value=50,
            step=1,
            value=20
        )

        st.divider()

        run_model = (
            field_of_interest != "-- select an option --"
            and st.button("Run Machine Learning")
        )

    else:
        field_of_interest = None
        train_percent_input = None
        run_model = False
        uploaded_df = None
        st.error(
            f"No uploaded file and default file not found: {DEFAULT_FILE}"
        )


# -----------------------------
# Main page for results
# -----------------------------
if run_model and uploaded_df is not None:
    with st.spinner('Running Machine Learning...'):
        train_pc = train_percent_input / 100

        if fields_to_remove:
          modified_df = uploaded_df.drop(fields_to_remove, axis=1)
        else:
          modified_df = uploaded_df

        # prepare data
        X_train, X_test, y_train, y_test = ml.prepare_data(
           modified_df, field_of_interest, train_pc
        )

        # run model
        model, accuracy_train, accuracy_test, co_eff_df, top_10_df, intercept = ml.run_log_reg(
            X_train, X_test, y_train, y_test
        )

        # shap values
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        y_pred = model.predict(X_test)

        # --- Now results show in the main body ---
        st.header("Model Results")

        st.write("Here is a Preview of the Data you are Analysing:")
        st.dataframe(modified_df.head())

        st.write(f'The thing we are trying to predict is: **{field_of_interest}**')
        st.write(f'We are using **{train_percent_input}%** of the data to train the model')

        st.header("Model Performance Metrics")
        
        st.write(f"The Accuracy of Training Dataset is: {accuracy_train}%")
        ml.display_metric_status("Training Accuracy", accuracy_train)
        st.write(f"The Accuracy of Test Dataset is: {accuracy_test}%")
        ml.display_metric_status("Test Accuracy", accuracy_test)
        st.write("Intercept = The model's starting prediction when all input variables are set to 0."
                "The baseline level before any factors are taken into account")
        st.write("A higher intercept means the model starts with a higher probability of a positive outcome."
                "A lower (more negative) intercept means the model starts with a lower probability of a positive outcome.")
        st.write(f"Intercept (β0): {intercept}")

        precision = precision_score(y_test, y_pred, zero_division=0)
        st.write("Precision = Of all the cases the model predicted as positive, how many were actually positive?")
        ml.display_metric_status("Precision", precision)
        st.write("Recall = Of all the cases that were actually positive, how many did the model successfully find?")
        recall = recall_score(y_test, y_pred, zero_division=0)
        ml.display_metric_status("Recall (Sensitivity)", recall)
        st.write("F1 Score = A single metric that balances Precision and Recall."
                "How good is the model overall at finding positives without making too many mistakes?")
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ml.display_metric_status("F1 Score", f1)
               
        class_balance = y_train.value_counts(normalize=True)

        st.header("Outcome Balance")

        st.write(
            f"{class_balance.iloc[0]:.1%} of records are {field_of_interest} = 0 and "
            f"{class_balance.iloc[1]:.1%} are {field_of_interest} = 1"
            )
        
        st.markdown("""
                    ### How to interpret the chart below

                    Machine learning models learn from historical examples. 
                    Before looking at model performance, it is important to 
                    understand how common the outcome is within the dataset.

                    - A balanced dataset contains similar numbers of records in each group.
                    - An imbalanced dataset contains many more records in one group than the other.
                    - Highly imbalanced datasets can make accuracy appear better than it really is.

                    For example, if 95% of patients do not miss their appointment,
                     a model that predicts "No DNA" for every patient would achieve 95% accuracy without actually learning anything useful.

                    This information helps put the model's performance results into context.
                    """)

        
        fig, ax = plt.subplots()
        y_train.value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)
        
        
        fig1, ax = plt.subplots()
        co_eff_df.head(10).plot(
            kind='barh', 
            x='feature', 
            y='coefficient (β)', 
            legend=False, 
            ax=ax
        )
        ax.set_title("Top 10 Influential Features")
        ax.invert_yaxis()

        st.pyplot(fig1)

        # 3. Feature effects: coefficients + odds ratios
        co_eff = model.coef_[0]
        intercept = model.intercept_[0]

        co_eff_df = pd.DataFrame({
            "feature": list(X_train.columns) if hasattr(X_train, "columns") else [f"X{i}" for i in range(X_train.shape[1])],
            "coefficient (β)": co_eff,
            "odds_ratio (exp(β))": np.exp(co_eff),
            "abs_co_eff": np.abs(co_eff)
        })

        co_eff_df.sort_values(by="abs_co_eff", ascending=False, inplace=True)

        print("\nIntercept (β0):", intercept)
        print("\nFeature Effects (sorted by influence):")
        print(co_eff_df[["feature", "coefficient (β)", "odds_ratio (exp(β))"]])

        # Probability Predictions Distribution
        st.header("Probability Distribution")

        # Get predicted probabilities for the positive class
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        fig1, ax = plt.subplots()
        sns.histplot(y_pred_proba, bins=20, kde=True, ax=ax)
        ax.set_xlabel("Predicted probability of Class 1")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Predicted Probabilities")
        st.pyplot(fig1)

        st.markdown("""
        **How to interpret:**  
        - This chart shows how confident the model is when predicting.  
        - Ideally, predictions for patients **with the outcome** cluster on the **right** (high probabilities),  
          and predictions for patients **without the outcome** cluster on the **left** (low probabilities).  
        - A clear separation means the model is good at distinguishing the two classes.
        """)

        # Feature Effects 
        st.header("Feature Effects (Odds Ratios)")
        
        st.markdown("""
                    ### Understanding the Model Results

                    #### Feature
                    **What patient factor are we looking at?**

                    Examples include:
                    - Age
                    - Previous missed appointments (DNAs)
                    - Referral source
                    - Diagnosis

                    ---

                    #### Coefficient (β)
                    **Does this factor make the outcome more or less likely?**

                    - **Positive value** → makes the outcome more likely
                    - **Negative value** → makes the outcome less likely
                    - **Larger values** → have a stronger influence on the prediction

                    **Example:**  
                    If *Previous DNA* has a positive coefficient, the model has
                    learned that patients who previously missed appointments are
                    more likely to miss future appointments.

                    ---

                    #### Odds Ratio (exp(β))
                    **How much does this factor change the likelihood?**

                    - **1** = no effect
                    - **2** = about twice as likely
                    - **0.5** = about half as likely

                    **Example:**  
                    If *Previous DNA* has an Odds Ratio of **2.0**, patients 
                    with a previous DNA are about **twice as likely** to miss 
                    another appointment compared with patients who have not 
                    previously missed one.

                    ---

                    💡 **Important:**  
                    Machine learning identifies factors that are associated with
                    an outcome. It cannot tell us why something happens, but it
                    can help us understand which factors have the strongest 
                    influence on the prediction which can then be used to target
                    more detailed qualitative research to help understand the 
                    possible cause(s)
                    """)

        coeffs = model.coef_[0]
        feature_names = X_train.columns if hasattr(X_train, "columns") else [f"X{i}" for i in range(len(coeffs))]
        results = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient (β)": coeffs,
            "Odds Ratio (exp(β))": np.exp(coeffs)
        }).sort_values(by="Odds Ratio (exp(β))", ascending=False)

        st.dataframe(results)

        top_10_df["odds_ratio (exp(β))"] = pd.to_numeric(top_10_df["odds_ratio (exp(β))"], errors='coerce')
        top_10_df = top_10_df.dropna(subset=["odds_ratio (exp(β))"])

        # Add color based on OR
        top_10_df["color"] = top_10_df["odds_ratio (exp(β))"].apply(lambda x: "green" if x > 1 else "orange")

        # Ensure 'effect' column exists before plotting
        if "effect" not in top_10_df.columns:
            top_10_df["effect"] = np.where(
                top_10_df["odds_ratio (exp(β))"] > 1,
                "Increases Prob",
                "Decreases Prob"
            )
        
        fig2, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            data=top_10_df,
            x="odds_ratio (exp(β))",
            y="feature",
            hue="effect",              # use categorical hue
            palette={"Increases Prob": "green", "Decreases Prob": "orange"},
            dodge=False,               # bars should stay in single column
            ax=ax
        )

        ax.axvline(1, color="red", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title("Top 10 Feature Effects (Odds Ratios, Log Scale)")
        ax.set_xlabel("Odds Ratio (log scale)")
        ax.set_ylabel("Feature")
        ax.invert_yaxis()

        st.pyplot(fig2)
        st.markdown("""
        **Feature Effects (Top 10 Odds Ratios):**  
        - The **red dashed line** represents an odds ratio of **1** (no effect).  
        - **Green bars (OR > 1)** increase the likelihood of the outcome.  
        - **Orange bars (OR < 1)** decrease the likelihood of the outcome.  
        - The further a bar is from the red line, the stronger its influence 
        - on the prediction.
        """)

        baseline_p = y_train.mean()
        prob_table = ml.prob_change_table_with_interpretation(co_eff_df, X_train, baseline_prob=baseline_p)
        prob_table_top_10 = prob_table.head(10)

        # st.markdown('How to interpret the results')
        
        # st.write(prob_table_top_10)

        
        # Confusion Matrix
        st.header("Confusion Matrix")

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig3, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig3)

        st.markdown("""
        **How to interpret:**  
        - **True Positives (top-left)** = correct predictions for patients with outcome.  
        - **True Negatives (bottom-right)** = correct predictions for patients without outcome.  
        - **False Positives / False Negatives** = model errors.  
        - Helps clinicians understand the trade-off between missed cases and false alarms.
        """)

        # Calibration Curve
        st.header("Calibration Curve")

        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        fig4, ax = plt.subplots()
        ax.plot(prob_pred, prob_true, marker='o', label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        st.pyplot(fig4)

        st.markdown("""
        **How to interpret:**  
        - Checks if predicted probabilities are **trustworthy**.  
        - Points close to the dashed line mean good calibration (e.g., if the model says 70%,  
          about 70% of those patients really have the outcome).  
        - If the curve is **above** the line, the model is under-confident;  
          if it's **below**, it's over-confident.
        - HINT If the model is over-confident try reducing the amount of training
          data used, under-confident - try using more data to train your model. This
          can be done by adjusting the input in the menu on the left hand side.
        """)
        
        # ROC Curve
        st.header("ROC Curve")

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fig5, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title("Receiver Operating Characteristic (ROC)")
        ax.legend()
        st.pyplot(fig5)

        st.markdown("""
        **How to interpret:**  
        - The ROC curve shows how well the model distinguishes between classes.  
        - **Closer to the top-left corner = better performance.**  
        - **AUC values:**  
          - 0.5 = no better than chance  
          - 0.7–0.8 = acceptable  
          - 0.8–0.9 = good  
          - >0.9 = excellent  
        """)
        
        ml.display_metric_status("AUC", roc_auc)
        
        avg_metric = np.mean([precision, recall, f1, roc_auc])

        if avg_metric >= 0.90:
            st.success(
                "Overall Assessment: This model appears to be highly effective at distinguishing between outcomes."
            )

        elif avg_metric >= 0.80:
            st.success(
                "Overall Assessment: This model performs well and may provide useful decision support."
            )

        elif avg_metric >= 0.70:
            st.warning(
                "Overall Assessment: The model shows reasonable predictive ability but should be used with caution."
            )

        else:
            st.error(
                "Overall Assessment: The model currently demonstrates limited predictive value and may require additional data or feature engineering."
            )

        st.subheader("Global Feature Importance (SHAP Summary)")

        # Global importance (summary plot)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig)

        st.markdown("""
        - Features at the **top** are the most influential across all patients.  
        - **Positive values** push predictions towards Class 1.  
        - **Negative values** push predictions towards Class 0.  
        """)
        

else: 
  st.info('''Please upload a file and select your column of interest to 
          continue''')
    
##############################
## Specific Patient Example ##
##############################

# # Select a patient from uploaded dataset
# st.subheader("Individual Patient Explanation")

# # Let user pick a row
# patient_index = st.number_input("Select patient index", min_value=0, max_value=len(X_test)-1, value=0)

# # Extract sample
# if hasattr(X_test, "iloc"):
#     sample = X_test.iloc[patient_index]
# else:
#     sample = pd.Series(X_test[patient_index], index=[f"X{i}" for i in range(X_test.shape[1])])

# # Compute linear combination (z) and probability
# coeffs = model.coef_[0]
# intercept = model.intercept_[0]
# z = intercept + np.dot(coeffs, sample)
# prob = 1 / (1 + np.exp(-z))
# pred_class = int(prob >= 0.5)

# # Contribution of each feature
# contributions = coeffs * sample
# contrib_df = pd.DataFrame({
#     "Feature": sample.index,
#     "Value": sample.values,
#     "Coefficient (β)": coeffs,
#     "Contribution (β*x)": contributions
# }).sort_values(by="Contribution (β*x)", ascending=False)

# # Display results
# st.write(f"**Predicted probability of outcome (Class 1): {prob:.2f}**")
# st.write(f"**Predicted class:** {pred_class}")

# st.markdown("### 🧮 Feature Contributions")
# st.dataframe(contrib_df)

# # Bar chart of contributions
# st.bar_chart(contrib_df.set_index("Feature")["Contribution (β*x)"])
# # 🔹 Step 2. Streamlit-friendly explanation text
# # python
# # Copy code
# st.markdown("""
# ### 🧑‍⚕️ How to interpret the patient explanation  

# - Each feature contributes to the final prediction through a **coefficient × value** calculation.  
# - Positive contributions push the prediction **towards Class 1** (higher probability).  
# - Negative contributions push the prediction **towards Class 0** (lower probability).  
# - The size of the contribution shows how strongly that feature influenced the result.  

# ✅ Example:  
# - If **Smoking = 1** and the coefficient for Smoking is **-0.9**, the contribution is negative, reducing the probability of the outcome.  
# - If **Age = 70** and the coefficient for Age is **0.05**, the contribution is positive, pushing the probability higher.  

# This allows clinicians to see **why the model made its prediction for this specific patient**.
# """)


