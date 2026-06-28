# Data Directory
This directory stores the medical datasets used in this project.
## Files
| File | Description |
|------|-------------|
| `heart_disease.csv` | Cleveland Heart Disease dataset (303 samples, 13 features) |
| `diabetes.csv` | Pima Indians Diabetes dataset (768 samples, 8 features) |
| `breast_cancer.csv` | Breast Cancer Wisconsin dataset (569 samples, 30 features) |
## How Data is Loaded
Datasets are **automatically downloaded** from the UCI ML Repository
when you run `predict.py` for the first time. If the download fails
(e.g., no internet), synthetic data is generated as a fallback.
Once downloaded, CSV files are cached locally so subsequent runs
are instant.
## Source Links
- Heart Disease: https://archive.ics.uci.edu/ml/datasets/heart+disease
- Diabetes: https://www.kaggle.com/uciml/pima-indians-diabetes-database
- Breast Cancer: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
> This folder is gitignored — CSV files are not committed to the repository.