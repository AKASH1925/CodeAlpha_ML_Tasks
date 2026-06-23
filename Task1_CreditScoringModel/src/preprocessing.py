"""
Preprocessing Module
Feature engineering and data preprocessing for credit scoring.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Debt-to-income ratio
    df['debt_to_income'] = df['debt'] / (df['income'] + 1)
    
    # Payment history ratio (payments per year of account age)
    df['payment_consistency'] = df['payment_history'] / (df['account_age'] / 12 + 1)
    
    # Credit utilization category
    df['high_utilization'] = (df['credit_utilization'] > 0.5).astype(int)
    
    # Income brackets
    df['high_income'] = (df['income'] > 60000).astype(int)
    
    # Account maturity
    df['mature_account'] = (df['account_age'] > 60).astype(int)
    
    # Risk score (simple heuristic)
    df['risk_score'] = (
        df['high_utilization'] * 2 +
        (df['debt_to_income'] > 0.3).astype(int) * 2 +
        (df['recent_inquiries'] > 3).astype(int) +
        (df['payment_history'] < 6).astype(int) * 3
    )
    
    return df
def prepare_data(df: pd.DataFrame, target_col: str = 'target', 
                 test_size: float = 0.2, random_state: int = 42):
    """
    Prepare data for training: feature engineering, split, and scale.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Engineer features
    df = engineer_features(df)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print(f"Features: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
if __name__ == "__main__":
    from data_loader import load_data
    
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    print("\nTraining features sample:")
    print(X_train.head())