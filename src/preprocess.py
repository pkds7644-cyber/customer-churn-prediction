import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    return df

def engineering_features(df):
    df['ChargesRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def prepare_pipeline(df, target_column='Churn'):
    df = clean_data(df)
    df = engineering_features(df)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    encoder = OneHotEncoder(
        sparse_output=False,
        drop='first',
        handle_unknown='ignore'
    )

    X_train_encoded = encoder.fit_transform(X_train[cat_cols])
    X_test_encoded = encoder.transform(X_test[cat_cols])

    encoded_cols = encoder.get_feature_names_out(cat_cols)

    X_train_cat_df = pd.DataFrame(
        X_train_encoded,
        columns=encoded_cols,
        index=X_train.index
    )

    X_test_cat_df = pd.DataFrame(
        X_test_encoded,
        columns=encoded_cols,
        index=X_test.index
    )

    scaler = StandardScaler()

    X_train_num_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_num_scaled = scaler.transform(X_test[num_cols])

    X_train_num_df = pd.DataFrame(
        X_train_num_scaled,
        columns=num_cols,
        index=X_train.index
    )

    X_test_num_df = pd.DataFrame(
        X_test_num_scaled,
        columns=num_cols,
        index=X_test.index
    )

    X_train_final = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
    X_test_final = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

    return X_train_final, X_test_final, y_train, y_test, scaler, encoder