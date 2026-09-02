import numpy as np
import pandas as pd
import re
import streamlit as st
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report # accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

def model_runner(X_train, y_train, X_test, y_test, model_type, model_name):
    model = model_type
    model = model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    accuracy_train = np.mean(y_pred_train == y_train)
    accuracy_test = np.mean(y_pred_test == y_test)
    precision_score_test = precision_score(
                                            y_test,
                                            y_pred_test,
                                            zero_division=0
                                        )

    recall_sensitivity_score_test = recall_score(
                                                y_test,
                                                y_pred_test,
                                                zero_division=0
                                                )

    f1_score_result = f1_score(
                                y_test,
                                y_pred_test,
                                zero_division=0
                                )
    specificity_score_test = precision_score(y_test, y_pred_test, pos_label=0)
    mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
    mae_test = metrics.mean_absolute_error(y_test, y_pred_test)
    mse_train = metrics.mean_squared_error(y_train, y_pred_train)
    mse_test = metrics.mean_squared_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    r2_train = metrics.r2_score(y_train, y_pred_train)
    r2_test = metrics.r2_score(y_test, y_pred_test)

    return {
        'Model': model_name,
        'Trained_Model':model,
        'Training_accuracy': accuracy_train,
        'Test_accuracy': accuracy_test,
        'Precision': precision_score_test,
        'Recall': recall_sensitivity_score_test,
        'Specificity': specificity_score_test,
        'F1 Score': f1_score_result,
        'Training MAE': mae_train,
        'Testing MAE': mae_test,
        'Training MSE': mse_train,
        'Test MSE': mse_test,
        'Training RMSE': rmse_train,
        'Test RMSE': rmse_test,
        'Training R2': r2_train,
        'Test R2': r2_test
    }

def run_all_models(
    X_train,
    y_train,
    X_test,
    y_test):

    results = []

    progress_bar = st.progress(0)

    status_text = st.empty()
    
    total_models = 8

    current_model = 0

    # Logistic Regression
    status_text.text("Running Logistic Regression...")

    results.append(
        model_runner(
            X_train,
            y_train,
            X_test,
            y_test,
            LogisticRegression(max_iter=1000),
            "Logistic Regression"
        )
    )

    current_model += 1

    progress_bar.progress(
        current_model / total_models
    )
    
    temp_df = pd.DataFrame(results)

    leader = (
        temp_df
        .sort_values(
            by="F1 Score",
            ascending=False
        )
        .iloc[0]
    )

    status_text.info(
        f"""
    Running model {current_model}/{total_models}

    Current leader:
    {leader['Model']}

    F1 Score:
    {leader['F1 Score']:.3f}
    """
    )

    # Decision Tree
    status_text.text("Running Decision Tree...")

    results.append(
        model_runner(
            X_train,
            y_train,
            X_test,
            y_test,
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ),
            "Decision Tree"
        )
    )

    current_model += 1

    progress_bar.progress(
        current_model / total_models
    )
    
    temp_df = pd.DataFrame(results)
    
    leader = (
        temp_df
        .sort_values(
            by="F1 Score",
            ascending=False
        )
        .iloc[0]
    )

    status_text.info(
        f"""
    Running model {current_model}/{total_models}

    Current leader:
    {leader['Model']}

    F1 Score:
    {leader['F1 Score']:.3f}
    """
    )

    # Random Forest
    
    status_text.text("Running Random Forest...")
    
    results.append(
        model_runner(
            X_train,
            y_train,
            X_test,
            y_test,
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            "Random Forest"
        )
    )
    
    current_model += 1
    
    progress_bar.progress(
        current_model / total_models
    )
    
    temp_df = pd.DataFrame(results)
    
    leader = (
        temp_df
        .sort_values(
            by="F1 Score",
            ascending=False
        )
        .iloc[0]
    )

    status_text.info(
        f"""
    Running model {current_model}/{total_models}

    Current leader:
    {leader['Model']}

    F1 Score:
    {leader['F1 Score']:.3f}
    """
    )

    # XGBoost
    
    status_text.text("Running XGBoost...")
    
    results.append(
        model_runner(
            X_train,
            y_train,
            X_test,
            y_test,
            XGBClassifier(
                max_depth=5,
                eval_metric="logloss",
                random_state=42
            ),
            "XGBoost"
        )
    )
    
    current_model += 1
    
    progress_bar.progress(
        current_model / total_models
    )
    
    temp_df = pd.DataFrame(results)
    
    leader = (
        temp_df
        .sort_values(
            by="F1 Score",
            ascending=False
        )
        .iloc[0]
    )

    status_text.info(
        f"""
    Running model {current_model}/{total_models}

    Current leader:
    {leader['Model']}

    F1 Score:
    {leader['F1 Score']:.3f}
    """
    )

    # LightGBM
    
    status_text.text("Running LightGBM Classifier...")
    
    results.append(
        model_runner(
            X_train,
            y_train,
            X_test,
            y_test,
            LGBMClassifier(
                random_state=42
            ),
            "LightGBM"
        )
    )
    
    current_model += 1
    
    progress_bar.progress(
        current_model / total_models
    )
    
    temp_df = pd.DataFrame(results)
    
    leader = (
        temp_df
        .sort_values(
            by="F1 Score",
            ascending=False
        )
        .iloc[0]
    )

    status_text.info(
        f"""
    Running model {current_model}/{total_models}

    Current leader:
    {leader['Model']}

    F1 Score:
    {leader['F1 Score']:.3f}
    """
    )

    # CatBoost
    
    status_text.text("Running CatBoost...")
    
    results.append(
        model_runner(
            X_train,
            y_train,
            X_test,
            y_test,
            CatBoostClassifier(
                silent=True,
                random_state=42
            ),
            "CatBoost"
        )
    )
    
    current_model += 1
    
    progress_bar.progress(
        current_model / total_models
    )
    
    temp_df = pd.DataFrame(results)
    
    leader = (
        temp_df
        .sort_values(
            by="F1 Score",
            ascending=False
        )
        .iloc[0]
    )

    status_text.info(
        f"""
    Running model {current_model}/{total_models}

    Current leader:
    {leader['Model']}

    F1 Score:
    {leader['F1 Score']:.3f}
    """
    )

    # Histogram Gradient Boost
    
    status_text.text("Running Histogram Gradient Boost...")
    
    results.append(
        model_runner(
            X_train,
            y_train,
            X_test,
            y_test,
            HistGradientBoostingClassifier(),
            "Histogram Gradient Boosting"
        )
    )
    
    current_model += 1

    progress_bar.progress(
        current_model / total_models
    )
    
    temp_df = pd.DataFrame(results)
    
    leader = (
        temp_df
        .sort_values(
            by="F1 Score",
            ascending=False
        )
        .iloc[0]
    )

    status_text.info(
        f"""
    Running model {current_model}/{total_models}

    Current leader:
    {leader['Model']}

    F1 Score:
    {leader['F1 Score']:.3f}
    """
    )

    # Naive Bayes
    
    status_text.text("Running Naive Bayes...")
    
    results.append(
        model_runner(
            X_train,
            y_train,
            X_test,
            y_test,
            GaussianNB(),
            "Naive Bayes"
        )
    )
    
    current_model += 1

    progress_bar.progress(
        current_model / total_models
    )
    
    temp_df = pd.DataFrame(results)
    
    leader = (
        temp_df
        .sort_values(
            by="F1 Score",
            ascending=False
        )
        .iloc[0]
    )

    status_text.info(
        f"""
    Running model {current_model}/{total_models}

    Current leader:
    {leader['Model']}

    F1 Score:
    {leader['F1 Score']:.3f}
    """
    )
    
    progress_bar.progress(1.0)

    status_text.success(
        "✅ Model comparison complete"
    )

    return pd.DataFrame(results)

def run_selected_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
    depth=5,
    class_weight=None
    ):

    if model_name == "Logistic Regression":

        model = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight
        )

    elif model_name == "Decision Tree*":

        model = DecisionTreeClassifier(
            max_depth=depth,
            class_weight=class_weight,
            random_state=42
        )

    elif model_name == "Random Forest":

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=depth,
            class_weight=class_weight,
            random_state=42
        )

    elif model_name == "XGBoost":

        model = XGBClassifier(
            max_depth=depth,
            eval_metric="logloss",
            random_state=42
        )

    elif model_name == "AdaBoost*":

        model = AdaBoostClassifier(
            random_state=42
        )

    elif model_name == "CatBoost":

        model = CatBoostClassifier(
            silent=True,
            random_state=42
        )

    elif model_name == "LightGBM":

        model = LGBMClassifier(
            random_state=42
        )

    elif model_name == "Histogram Gradient Boosting":

        model = HistGradientBoostingClassifier()

    elif model_name == "Support Vector Machine":

        model = svm.SVC(
            probability=True
        )

    elif model_name == "Naive Bayes":

        model = GaussianNB()

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model.fit(
        X_train,
        y_train
    )

    accuracy_train = round(
        model.score(
            X_train,
            y_train
        ) * 100,
        2
    )

    accuracy_test = round(
        model.score(
            X_test,
            y_test
        ) * 100,
        2
    )

    return (
        model,
        accuracy_train,
        accuracy_test
    )


# def run_all_models(X_train, y_train, X_test, y_test):
#     results = []

#     # Models with varying depth
#     for i in range(1, 10):
#         results.append(model_runner(X_train, y_train, X_test, y_test,
#                         DecisionTreeClassifier(max_depth=i),
#                         f'Decision Tree - Depth:{i}'))

#         results.append(model_runner(X_train, y_train, X_test, y_test,
#                         RandomForestClassifier(max_depth=i),
#                         f'Random Forest - Depth:{i}'))

#         results.append(model_runner(X_train, y_train, X_test, y_test,
#                         XGBClassifier(max_depth=i, use_label_encoder=False, eval_metric='mlogloss'),
#                         f'XG Boost - Depth:{i}'))

#     # Models without depth tuning
#     results.append(model_runner(X_train, y_train, X_test, y_test,
#                     LogisticRegression(max_iter=1000), 'Logistic Regression'))

#     results.append(model_runner(X_train, y_train, X_test, y_test,
#                     AdaBoostClassifier(), 'ADA Boost'))

#     results.append(model_runner(X_train, y_train, X_test, y_test,
#                     CatBoostClassifier(silent=True), 'Cat Boost'))

#     results.append(model_runner(X_train, y_train, X_test, y_test,
#                     LGBMClassifier(), 'Light Gradient Boost'))

#     results.append(model_runner(X_train, y_train, X_test, y_test,
#                     HistGradientBoostingClassifier(), 'Histogram Gradient Boost'))

#     results.append(model_runner(X_train, y_train, X_test, y_test,
#                     svm.SVC(), 'Support Vector Machine'))

#     results.append(model_runner(X_train, y_train, X_test, y_test,
#                     GaussianNB(), 'Naive Bayes'))

    # Return as DataFrame
    # return pd.DataFrame(results)

def fix_dtypes(df):
    df = df.copy()  # avoid changing original

    for col in df.columns:
        # If it's already numeric or datetime, skip
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        # Try to convert to datetime
        try:
            converted = pd.to_datetime(df[col], errors='raise')
            df[col] = converted
            print(f"[INFO] Converted '{col}' to datetime")
            continue
        except (ValueError, TypeError):
            pass
        
        # Try to convert to numeric
        try:
            converted = pd.to_numeric(df[col], errors='raise')
            df[col] = converted
            print(f"[INFO] Converted '{col}' to numeric")
            continue
        except (ValueError, TypeError):
            pass
        
        # If it's object or string with few unique values, convert to category
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            num_unique = df[col].nunique(dropna=True)
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # tweak threshold if needed
                df[col] = df[col].astype('category')
                print(f"[INFO] Converted '{col}' to category")

    return df

# used to identify columns already one hot encoded in the source data
def is_one_hot_column(series):
    return set(series.dropna().unique()).issubset({0, 1})

def clean_column(name):

    name = str(name)

    name = re.sub(
        r'[^A-Za-z0-9_]',
        '_',
        name
    )

    return name

def make_unique_columns(columns):

    seen = {}

    new_cols = []

    for col in columns:

        if col not in seen:

            seen[col] = 0

            new_cols.append(col)

        else:

            seen[col] += 1

            new_cols.append(
                f"{col}_{seen[col]}"
            )

    return new_cols

def prepare_data(source_df, targ_col, train_pc):
    
    # Separate features and target
    X = source_df.drop(targ_col, axis=1)
    


    # detect deprivation index columns and keep as categorical
    imd_columns = []

    for col in X.columns:

        col_lower = str(col).lower()

        # Common IMD naming patterns
        if (
            "imd" in col_lower
            or "index of multiple deprivation" in col_lower
            or ("index" in col_lower and "dep" in col_lower)
            or ("multiple" in col_lower and "deprivation" in col_lower)
            or ("deprivation" in col_lower and "decile" in col_lower)
            or ("deprivation" in col_lower and "index" in col_lower)
        ):
            imd_columns.append(col)

    # Convert any detected IMD columns to categorical
    for col in imd_columns:

        X[col] = (
            X[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    if imd_columns:
        print(f"IMD columns detected: {imd_columns}")
    else:
        print("No IMD columns detected")
        
    y = source_df[targ_col].astype(str).str.strip().str.lower()

    # Map text labels to binary
    y = y.replace({
        "y": 1, "yes": 1, "true": 1, "1": 1,
        "n": 0, "no": 0, "false": 0, "0": 0
    })

    # Convert to numeric, invalid entries -> NaN
    y = pd.to_numeric(y, errors="coerce")

    # Keep only valid rows (drop NaNs in y)
    mask = ~y.isna()
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int)

    # Clean column names
    X.columns = [clean_column(col) for col in X.columns]

    # Convert obvious numerics, leave objects for encoding
    for col in X.columns:
        if col not in imd_columns:
            X[col] = pd.to_numeric(X[col],errors="ignore")

    # Detect column types
    categorical_cols = X.select_dtypes(include=['string', 'object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'uint8']).columns

    # Identify one-hot encoded columns
    one_hot_cols = [col for col in numeric_cols if is_one_hot_column(X[col])]

    # Exclude one-hot columns from scaling
    numerical_cols = [col for col in numeric_cols if col not in one_hot_cols]

    # Fill missing categorical values
    for col in X.columns:
        if col not in imd_columns:
            try:
                X[col] = pd.to_numeric(X[col])
            except (ValueError, TypeError):
                pass

    for col in imd_columns:
        X[col] = X[col].astype("category")

    # Fill missing numeric values
    X[numerical_cols] = X[numerical_cols].fillna(0)

    # Add prefixes to categorical dummies
    #prefixes = {col: col[:5] for col in categorical_cols}
    prefixes = {col: clean_column(col) for col in categorical_cols
}

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=categorical_cols, prefix=prefixes, dtype=int)
    
    # Final XGBoost-safe column names
    X.columns = [
        re.sub(
            r'[^A-Za-z0-9_]',
            '_',
            str(col)
        )
        for col in X.columns
    ]
    
    # Remove duplicate columns created during cleaning

    X.columns = make_unique_columns(X.columns)

    # Scale numerical columns if they exist
    scaler = StandardScaler()
    if len(numerical_cols) > 0:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_pc, random_state=42
    )

    return X_train, X_test, y_train, y_test


def run_log_reg(X_train,X_test,y_train,y_test,class_weight=None):

    # 1. Fit logistic regression
    model = LogisticRegression(max_iter=1000,class_weight=class_weight)
    model.fit(X_train, y_train)

    # 2. Predictions & accuracy
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    accuracy_train = np.mean(y_pred_train == y_train)
    accuracy_test = np.mean(y_pred_test == y_test)

    print(f'Accuracy of predicting training data = {accuracy_train:.3f}')
    print(f'Accuracy of predicting test data = {accuracy_test:.3f}')
    print("\nClassification Report (test set):\n", classification_report(y_test, y_pred_test))

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

    top_10_df = co_eff_df.head(10)

    return model, accuracy_train, accuracy_test, co_eff_df, top_10_df, intercept


# this function turns the probability coefficients and odds ration into a 
# probability % increase with explanation
def prob_change_table_with_interpretation(coeff_df, X_train, baseline_prob=0.2, delta_x=1):
    """
    Create a probability change table with human-readable interpretations
    for one-hot encoded features.

    Args:
        coeff_df (pd.DataFrame): Must have 'feature' and 'coefficient (β)' columns.
        X_train (pd.DataFrame): Training data (used to detect one-hot columns).
        baseline_prob (float): Baseline probability (0–1).
        delta_x (float): Change in feature value (default 1).

    Returns:
        pd.DataFrame: Table with interpretation text, probabilities and changes.
    """
    baseline_odds = baseline_prob / (1 - baseline_prob)

    results = coeff_df.copy()
    results["baseline_prob"] = baseline_prob * 100
    results["new_prob"] = (baseline_odds * np.exp(results["coefficient (β)"] * delta_x)) / \
                          (1 + (baseline_odds * np.exp(results["coefficient (β)"] * delta_x))) * 100
    results["absolute_change"] = (results["new_prob"] - results["baseline_prob"])
    # Detect one-hot columns: only 0/1 values
    one_hot_cols = [col for col in X_train.columns if set(X_train[col].unique()) <= {0, 1}]

    interpretations = []
    for feature in results["feature"]:
        if feature in one_hot_cols:
            # Make interpretation more readable: replace underscores and prefix "Effect of being"
            human_readable = feature.replace("_", " ")
            interpretations.append(f"Effect of being {human_readable}")
        else:
            interpretations.append(f"Effect of increasing {feature} by {delta_x}")
    
    results["interpretation"] = interpretations

    # Sort by absolute effect
    results = results.sort_values(by="absolute_change", ascending=False)

    return results[[
        "feature", "interpretation", "coefficient (β)", "odds_ratio (exp(β))",
        "baseline_prob", "new_prob", "absolute_change"
    ]]

# Example usage:
# prob_table = prob_change_table_with_interpretation(co_eff_df, X_train, baseline_prob=0.2)
# print(prob_table.head(10))

def display_metric_status(metric_name, value):
    """
    Display model metric with traffic-light interpretation.

    >= 0.90 = Excellent
    >= 0.80 = Good
    >= 0.70 = Acceptable
    < 0.70 = Needs Improvement
    """

    if value >= 0.90:
        st.success(
            f"🟢 {metric_name}: {value:.1%} "
            f"(Excellent - the model is performing very well)"
        )

    elif value >= 0.80:
        st.success(
            f"🟢 {metric_name}: {value:.1%} "
            f"(Good - the model is performing well)"
        )

    elif value >= 0.70:
        st.warning(
            f"🟠 {metric_name}: {value:.1%} "
            f"(Acceptable - the model may still be useful but should be interpreted with caution)"
        )

    else:
        st.error(
            f"🔴 {metric_name}: {value:.1%} "
            f"(Poor - the model is struggling to make accurate predictions)"
        )
        
def build_model_from_name(model_name):

    if "Logistic Regression" in model_name:
        return LogisticRegression(max_iter=1000)

    elif "Decision Tree" in model_name:

        depth = int(
            model_name.split(":")[-1]
        )

        return DecisionTreeClassifier(
            max_depth=depth
        )

    elif "Random Forest" in model_name:

        depth = int(
            model_name.split(":")[-1]
        )

        return RandomForestClassifier(
            max_depth=depth
        )

    elif "XG Boost" in model_name:

        depth = int(
            model_name.split(":")[-1]
        )

        return XGBClassifier(
            max_depth=depth,
            eval_metric="logloss"
        )

    elif "ADA Boost" in model_name:

        return AdaBoostClassifier()

    elif "Cat Boost" in model_name:

        return CatBoostClassifier(
            silent=True
        )

    elif "Light Gradient Boost" in model_name:

        return LGBMClassifier()

    elif "Histogram Gradient Boost" in model_name:

        return HistGradientBoostingClassifier()

    elif "Support Vector Machine" in model_name:

        return svm.SVC(
            probability=True
        )

    elif "Naive Bayes" in model_name:

        return GaussianNB()