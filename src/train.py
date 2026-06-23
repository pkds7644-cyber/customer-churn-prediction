# import os
# import joblib
# import os
# import pandas as pd
# import numpy as np
# from preprocess import prepare_pipeline

# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score,
#     f1_score, roc_auc_score, confusion_matrix
# )

# def load_and_prepare_data(filepath):
#     df = pd.read_csv(filepath)
#     X_train, X_test, y_train, y_test, scaler, encoder = prepare_pipeline(df)
#     return X_train, X_test, y_train, y_test, scaler, encoder

# def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
#     print(f"\n--- {model_name} Performance ---")

#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     roc_auc = roc_auc_score(y_true, y_pred_proba)

#     print(f"Accuracy:  {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall:    {recall:.4f}")
#     print(f"F1 Score:  {f1:.4f}")
#     print(f"ROC-AUC:   {roc_auc:.4f}")
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_true, y_pred))

# def train_baseline_models(X_train, y_train, X_test, y_test):
#     log_reg = LogisticRegression(random_state=42, max_iter=1000)
#     log_reg.fit(X_train, y_train)

#     y_pred_log = log_reg.predict(X_test)
#     y_prob_log = log_reg.predict_proba(X_test)[:, 1]

#     evaluate_model(
#         y_test,
#         y_pred_log,
#         y_prob_log,
#         "Logistic Regression"
#     )

#     tree = DecisionTreeClassifier(
#         random_state=42,
#         max_depth=5
#     )

#     tree.fit(X_train, y_train)

#     y_pred_tree = tree.predict(X_test)
#     y_prob_tree = tree.predict_proba(X_test)[:, 1]

#     evaluate_model(
#         y_test,
#         y_pred_tree,
#         y_prob_tree,
#         "Decision Tree"
#     )

# def tune_random_forest(X_train, y_train, X_test, y_test):
#     print("\n--- Tuning Random Forest (This might take a minute) ---")

#     rf = RandomForestClassifier(random_state=42)

#     param_grid = {
#         'n_estimators': [100, 200],
#         'max_depth': [5, 10, None],
#         'min_samples_split': [2, 5, 10]
#     }

#     grid_search = GridSearchCV(
#         estimator=rf,
#         param_grid=param_grid,
#         cv=3,
#         scoring='roc_auc',
#         n_jobs=-1
#     )

#     grid_search.fit(X_train, y_train)

#     best_rf = grid_search.best_estimator_

#     print(f"Best Parameters Found: {grid_search.best_params_}")

#     y_pred_rf = best_rf.predict(X_test)
#     y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

#     evaluate_model(
#         y_test,
#         y_pred_rf,
#         y_prob_rf,
#         "Tuned Random Forest"
#     )

#     feature_importances = pd.DataFrame({
#         'Feature': X_train.columns,
#         'Importance': best_rf.feature_importances_
#     }).sort_values(
#         by='Importance',
#         ascending=False
#     )

#     print("\n--- Top 5 Most Important Features ---")
#     print(feature_importances.head(5))

#     return best_rf
# if __name__ == "__main__":
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#     DATA_PATH = os.path.join(
#         BASE_DIR,
#         "data",
#         "churn_data.csv"
#     )

#     X_train, X_test, y_train, y_test, scaler, encoder = load_and_prepare_data(
#         DATA_PATH
#     )

#     train_baseline_models(
#         X_train,
#         y_train,
#         X_test,
#         y_test
#     )

#     best_model = tune_random_forest(
#         X_train,
#         y_train,
#         X_test,
#         y_test
#     )

import os
import joblib
import pandas as pd
import numpy as np
from preprocess import prepare_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    X_train, X_test, y_train, y_test, scaler, encoder = prepare_pipeline(df)
    return X_train, X_test, y_train, y_test, scaler, encoder

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    print(f"\n--- {model_name} Performance ---")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def train_baseline_models(X_train, y_train, X_test, y_test):
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)

    y_pred_log = log_reg.predict(X_test)
    y_prob_log = log_reg.predict_proba(X_test)[:, 1]

    evaluate_model(
        y_test,
        y_pred_log,
        y_prob_log,
        "Logistic Regression"
    )

    tree = DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    )

    tree.fit(X_train, y_train)

    y_pred_tree = tree.predict(X_test)
    y_prob_tree = tree.predict_proba(X_test)[:, 1]

    evaluate_model(
        y_test,
        y_pred_tree,
        y_prob_tree,
        "Decision Tree"
    )

def tune_random_forest(X_train, y_train, X_test, y_test):
    print("\n--- Tuning Random Forest ---")

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    print(f"Best Parameters Found: {grid_search.best_params_}")

    y_pred_rf = best_rf.predict(X_test)
    y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

    evaluate_model(
        y_test,
        y_pred_rf,
        y_prob_rf,
        "Tuned Random Forest"
    )

    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values(
        by='Importance',
        ascending=False
    )

    print("\n--- Top 5 Most Important Features ---")
    print(feature_importances.head(5))

    return best_rf

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "churn_data.csv"
    )

    X_train, X_test, y_train, y_test, scaler, encoder = load_and_prepare_data(
        DATA_PATH
    )

    train_baseline_models(
        X_train,
        y_train,
        X_test,
        y_test
    )

    best_model = tune_random_forest(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # model_dir = os.path.join(BASE_DIR, "models")
    # os.makedirs(model_dir, exist_ok=True)
    # model_path = os.path.join(model_dir, "churn_model.pkl")
    
    # joblib.dump(best_model, model_path)
    # print(f"\nModel successfully saved to: {model_path}")
    
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "churn_model.pkl")
    
    model_artifacts = {
        "model": best_model,
        "scaler": scaler,
        "encoder": encoder
    }
    
    joblib.dump(model_artifacts, model_path)
    print(f"\n Model and preprocessing tools successfully saved to: {model_path}")