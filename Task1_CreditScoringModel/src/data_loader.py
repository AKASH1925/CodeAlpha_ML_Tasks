"""
Data Loader Module
Loads or generates the credit scoring dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
def generate_synthetic_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic credit scoring dataset.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with features and target column
    """
    np.random.seed(random_state)
    
    # Generate features
    income = np.random.normal(55000, 20000, n_samples).clip(15000, 200000)
    debt = np.random.exponential(10000, n_samples).clip(0, 100000)
    payment_history = np.random.randint(0, 25, n_samples)
    credit_utilization = np.random.beta(2, 5, n_samples)
    account_age = np.random.randint(1, 360, n_samples)
    employment_years = np.random.randint(0, 40, n_samples)
    num_credit_accounts = np.random.randint(1, 15, n_samples)
    recent_inquiries = np.random.poisson(2, n_samples).clip(0, 10)
    
    # Create DataFrame
    df = pd.DataFrame({
        'income': income,
        'debt': debt,
        'payment_history': payment_history,
        'credit_utilization': credit_utilization,
        'account_age': account_age,
        'employment_years': employment_years,
        'num_credit_accounts': num_credit_accounts,
        'recent_inquiries': recent_inquiries
    })
    
    # Generate target based on realistic rules
    creditworthiness = (
        (df['income'] > 40000) &
        (df['debt'] < 30000) &
        (df['payment_history'] > 12) &
        (df['credit_utilization'] < 0.6) &
        (df['employment_years'] > 2)
    ).astype(int)
    
    # Add some noise (10% flip)
    noise_mask = np.random.random(n_samples) < 0.1
    creditworthiness[noise_mask] = 1 - creditworthiness[noise_mask]
    
    df['target'] = creditworthiness
    
    return df
def load_data(data_path: str = None) -> pd.DataFrame:
    """
    Load credit scoring data from CSV or generate synthetic data.
    
    Args:
        data_path: Path to CSV file. If None or file doesn't exist, generates synthetic data.
        
    Returns:
        DataFrame with features and target
    """
    if data_path and Path(data_path).exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples")
        return df
    
    print("No data file found. Generating synthetic dataset...")
    df = generate_synthetic_data()
    print(f"Generated {len(df)} synthetic samples")
    return df
if __name__ == "__main__":
    df = load_data()
    print("\nDataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nTarget distribution:")
    print(df['target'].value_counts(normalize=True))