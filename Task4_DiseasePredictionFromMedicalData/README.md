# Task 4: Disease Prediction from Medical Data
A machine learning project that predicts the possibility of diseases based on
patient medical data using multiple classification algorithms.
## Objective
Predict the possibility of diseases based on patient data by applying
classification techniques to structured medical datasets.
## Datasets
| Dataset | Source | Samples | Features | Classes |
|---------|--------|---------|----------|---------|
| Heart Disease | UCI ML Repository | 303 | 13 | 2 (Disease / No Disease) |
| Diabetes | UCI ML Repository | 768 | 8 | 2 (Diabetic / Non-Diabetic) |
| Breast Cancer | UCI ML Repository | 569 | 30 | 2 (Malignant / Benign) |
## Algorithms Used
- **Support Vector Machine (SVM)** — Finds optimal hyperplane to separate classes
- **Logistic Regression** — Linear model for binary classification
- **Random Forest** — Ensemble of decision trees for robust classification
- **XGBoost** — Gradient boosting for high-performance classification
## Features Used
### Heart Disease
Age, Sex, Chest Pain Type, Resting Blood Pressure, Serum Cholesterol,
Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina,
ST Depression, Slope of Peak Exercise ST Segment, Number of Major Vessels, Thalassemia
### Diabetes
Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI,
Diabetes Pedigree Function, Age
### Breast Cancer
30 features from digitized FNA images: radius, texture, perimeter, area,
smoothness, compactness, concavity, symmetry, fractal dimension (mean, SE, worst)
## Results
| Dataset | SVM | Logistic Regression | Random Forest | XGBoost |
|---------|-----|---------------------|---------------|---------|
| Heart Disease | 85.0% | 83.3% | **86.7%** | 85.0% |
| Diabetes | 75.3% | 71.4% | **76.0%** | 75.3% |
| Breast Cancer | **98.2%** | **98.2%** | 95.6% | 94.7% |
## Setup & Usage
```bash
# Clone the repository
git clone https://github.com/<your-username>/CodeAlpha_ML_Tasks.git
cd CodeAlpha_ML_Tasks/Task4_DiseasePredictionfromMedicalData
# Option 1: Run setup script
bash setup.sh
# Option 2: Manual install
pip install -r requirements.txt
# Run all datasets
python predict.py
# Run specific dataset
python predict.py --dataset heart
python predict.py --dataset diabetes
python predict.py --dataset breast_cancer