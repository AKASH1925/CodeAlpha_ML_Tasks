"""
Preprocessing — Feature scaling and train-test splitting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def preprocess(df: pd.DataFrame, target_col: str = "target", test_size: float = 0.2, random_state: int = 42):
    """
    Split features and target, scale features, and create train/test sets.
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, feature_names