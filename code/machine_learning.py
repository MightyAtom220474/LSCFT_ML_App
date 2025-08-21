import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

# Import functions for evaluating ML model
from sklearn.metrics import recall_score, precision_score, f1_score,\
                            classification_report, \
                            confusion_matrix, ConfusionMatrixDisplay,\
                            auc, roc_curve
from sklearn.inspection import permutation_importance

from sklearn.neighbors import NearestNeighbors

def standardise_data(X_train, X_test):
    
    # Initialise a new scaling object for normalising input data
    sc = StandardScaler() 

    # Apply the scaler to the training and test sets
    train_std=sc.fit_transform(X_train)
    test_std=sc.fit_transform(X_test)
    
    return train_std, test_std

def model_runner(X_train, y_train, X_test, y_test, model_type, model_name):
    model = model_type
    model = model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    accuracy_train = np.mean(y_pred_train == y_train)
    accuracy_test = np.mean(y_pred_test == y_test)
    precision_score_test = precision_score(y_test, y_pred_test, average='micro')
    recall_sensitivity_score_test = recall_score(y_test, y_pred_test, 
                                                average='micro')
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