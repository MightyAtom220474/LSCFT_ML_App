import numpy as np
import pandas as pd
import re
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
    precision_score_test = precision_score(y_test, y_pred_test, average='micro')
    recall_sensitivity_score_test = recall_score(y_test, y_pred_test, average='micro')
    specificity_score_test = precision_score(y_test, y_pred_test, pos_label=0)
    f1_score_result = f1_score(y_test, y_pred_test, average='micro')       
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

def run_all_models(X_train, y_train, X_test, y_test):
    results = []

    # Models with varying depth
    for i in range(1, 10):
        results.append(model_runner(X_train, y_train, X_test, y_test,
                        DecisionTreeClassifier(max_depth=i),
                        f'Decision Tree - Depth:{i}'))

        results.append(model_runner(X_train, y_train, X_test, y_test,
                        RandomForestClassifier(max_depth=i),
                        f'Random Forest - Depth:{i}'))

        results.append(model_runner(X_train, y_train, X_test, y_test,
                        XGBClassifier(max_depth=i, use_label_encoder=False, eval_metric='mlogloss'),
                        f'XG Boost - Depth:{i}'))

    # Models without depth tuning
    results.append(model_runner(X_train, y_train, X_test, y_test,
                    LogisticRegression(max_iter=1000), 'Logistic Regression'))

    results.append(model_runner(X_train, y_train, X_test, y_test,
                    AdaBoostClassifier(), 'ADA Boost'))

    results.append(model_runner(X_train, y_train, X_test, y_test,
                    CatBoostClassifier(silent=True), 'Cat Boost'))

    results.append(model_runner(X_train, y_train, X_test, y_test,
                    LGBMClassifier(), 'Light Gradient Boost'))

    results.append(model_runner(X_train, y_train, X_test, y_test,
                    HistGradientBoostingClassifier(), 'Histogram Gradient Boost'))

    results.append(model_runner(X_train, y_train, X_test, y_test,
                    svm.SVC(), 'Support Vector Machine'))

    results.append(model_runner(X_train, y_train, X_test, y_test,
                    GaussianNB(), 'Naive Bayes'))

    # Return as DataFrame
    return pd.DataFrame(results)

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
    # Make sure it's a string first
    name = str(name)
    # Replace forbidden characters with underscore
    return re.sub(r'[\[\]<>]', '_', name)

def prepare_data(source_df,targ_col,train_pc):
    
    # Separate features and target
    X = source_df.drop(targ_col, axis=1)
    y = source_df[targ_col]
    # clean column names
    X.columns = [clean_column(col) for col in X.columns]

    # Detect column types
    categorical_cols = X.select_dtypes(include=['string','object','category']).columns.tolist()

    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'uint8']).columns

    # Identify one-hot encoded columns
    one_hot_cols = [col for col in numeric_cols if is_one_hot_column(X[col])]

    # Now exclude them from numerical preprocessing
    numerical_cols = [col for col in numeric_cols if col not in one_hot_cols]

    # Fill missing values 
    for col in categorical_cols:
        if pd.api.types.is_categorical_dtype(X[col]):
            if 'Missing' not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories('Missing')
        X[col] = X[col].fillna('Missing')

    X[numerical_cols] = X[numerical_cols].fillna(0)

    # add prefixes to identify category groups
    prefixes = {col: col[:3] for col in categorical_cols}

    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=categorical_cols, prefix=(prefixes), dtype=int)

    # Scale numeric columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_pc, random_state=42)
    
    return X_train, X_test, y_train, y_test

def run_log_reg(X_train, X_test, y_train, y_test):

    # 1. Fit logistic regression
    model = LogisticRegression(max_iter=1000)
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