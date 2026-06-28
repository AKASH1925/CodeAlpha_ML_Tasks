"""
Data Loader — Fetches medical datasets from UCI ML Repository and sklearn.
Supports:
  - Heart Disease (UCI)
  - Diabetes (UCI)
  - Breast Cancer (sklearn built-in)
"""
import os
import urllib.request
import pandas as pd
import numpy as np
from io import StringIO
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
def load_heart_disease() -> pd.DataFrame:
    """Load the Cleveland Heart Disease dataset from UCI ML Repository."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    local_path = os.path.join(DATA_DIR, "heart_disease.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    try:
        print("  Downloading Heart Disease dataset from UCI...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
        df = pd.read_csv(StringIO(raw), names=columns, na_values="?")
        df = df.dropna()
        df["target"] = (df["target"] > 0).astype(int)
        df.to_csv(local_path, index=False)
        print(f"  Saved to {local_path}")
        return df
    except Exception as e:
        print(f"  Failed to download: {e}")
        print("  Generating synthetic Heart Disease dataset...")
        return _generate_synthetic_heart()
def load_diabetes() -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset from UCI ML Repository."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    local_path = os.path.join(DATA_DIR, "diabetes.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    columns = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree_function", "age", "target"
    ]
    try:
        print("  Downloading Diabetes dataset...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
        df = pd.read_csv(StringIO(raw), names=columns)
        df.to_csv(local_path, index=False)
        print(f"  Saved to {local_path}")
        return df
    except Exception as e:
        print(f"  Failed to download: {e}")
        print("  Generating synthetic Diabetes dataset...")
        return _generate_synthetic_diabetes()
def load_breast_cancer() -> pd.DataFrame:
    """Load the Breast Cancer dataset from sklearn (built-in, no download needed)."""
    from sklearn.datasets import load_breast_cancer
    local_path = os.path.join(DATA_DIR, "breast_cancer.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    print("  Loading Breast Cancer dataset from sklearn...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df.to_csv(local_path, index=False)
    print(f"  Saved to {local_path}")
    return df
def _generate_synthetic_heart() -> pd.DataFrame:
    """Generate synthetic heart disease data as fallback."""
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "age": np.random.randint(29, 77, n),
        "sex": np.random.randint(0, 2, n),
        "cp": np.random.randint(0, 4, n),
        "trestbps": np.random.randint(94, 200, n),
        "chol": np.random.randint(126, 564, n),
        "fbs": np.random.randint(0, 2, n),
        "restecg": np.random.randint(0, 2, n),
        "thalach": np.random.randint(71, 202, n),
        "exang": np.random.randint(0, 2, n),
        "oldpeak": np.round(np.random.uniform(0, 6.2, n), 1),
        "slope": np.random.randint(0, 3, n),
        "ca": np.random.randint(0, 5, n),
        "thal": np.random.randint(0, 3, n),
        "target": np.random.randint(0, 2, n),
    })
    df.to_csv(os.path.join(DATA_DIR, "heart_disease.csv"), index=False)
    return df
def _generate_synthetic_diabetes() -> pd.DataFrame:
    """Generate synthetic diabetes data as fallback."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "pregnancies": np.random.randint(0, 17, n),
        "glucose": np.random.randint(0, 200, n),
        "blood_pressure": np.random.randint(0, 122, n),
        "skin_thickness": np.random.randint(0, 100, n),
        "insulin": np.random.randint(0, 846, n),
        "bmi": np.round(np.random.uniform(0, 67.1, n), 1),
        "diabetes_pedigree_function": np.round(np.random.uniform(0.078, 2.42, n), 3),
        "age": np.random.randint(21, 81, n),
        "target": np.random.randint(0, 2, n),
    })
    df.to_csv(os.path.join(DATA_DIR, "diabetes.csv"), index=False)
    return df
DATASETS = {
    "heart": ("Heart Disease", load_heart_disease),
    "diabetes": ("Diabetes", load_diabetes),
    "breast_cancer": ("Breast Cancer", load_breast_cancer),
}
def load_dataset(name: str) -> pd.DataFrame:
    """Load a dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(DATASETS.keys())}")
    _, loader = DATASETS[name]
    return loader()