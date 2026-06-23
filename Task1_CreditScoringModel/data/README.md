# Dataset Information
## Credit Scoring Dataset
This folder contains the dataset used for the Credit Scoring Model.
### Option 1: Auto-Generated (Default)
The main script (`credit_scoring.py`) will automatically generate a synthetic dataset if no CSV file is present. This is useful for testing and demonstration.
### Option 2: Use Real Datasets
For production or real-world applications, you can use these datasets:
1. **UCI German Credit Dataset**
   - Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
   - Download and place as `credit_data.csv` in this folder
2. **Kaggle Credit Risk Dataset**
   - Source: https://www.kaggle.com/datasets/wordsforthewise/lending-club
   - Download and place as `credit_data.csv` in this folder
### Dataset Format
The CSV file should have the following columns:
- `income`: Annual income
- `debt`: Total debt amount
- `payment_history`: Number of on-time payments (0-24)
- `credit_utilization`: Credit utilization ratio (0-1)
- `account_age`: Age of oldest account in months
- `employment_years`: Years at current employment
- `num_credit_accounts`: Number of credit accounts
- `recent_inquiries`: Number of recent credit inquiries
- `target`: Creditworthy (1) or Not Creditworthy (0)