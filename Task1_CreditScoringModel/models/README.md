# Trained Models Directory
This directory contains the serialized trained machine learning models and preprocessing artifacts used for credit scoring predictions.
##  Files in This Directory
1. **logistic_regression.joblib**
- **Model Type:** Logistic Regression (Linear Classification)
- **Use Case:** Baseline model for binary classification
- **Size:** ~2-5 KB
- **Performance:** 
  - Accuracy: ~78.5%
  - ROC-AUC: ~82.6%
  - Fast predictions
- **Best For:** Interpretable, probabilistic predictions
**Loading:**
```python
import joblib
model = joblib.load('models/logistic_regression.joblib')
prediction = model.predict(X_test)
probability = model.predict_proba(X_test)
2. decision_tree.joblib
Model Type: Decision Tree Classifier
Use Case: Rule-based classification with interpretable splits
Size: ~10-20 KB
Performance:
Accuracy: ~79.0%
ROC-AUC: ~73.4%
Medium prediction speed
Best For: Feature importance analysis, business rules
Loading:

import joblib
model = joblib.load('models/decision_tree.joblib')
prediction = model.predict(X_test)
# Visualize tree structure
from sklearn.tree import plot_tree
plot_tree(model, feature_names=feature_names)
3. random_forest.joblib 
Model Type: Random Forest (Ensemble of Decision Trees)
Use Case: Best overall performance for credit scoring
Size: ~200-500 KB
Performance:
Accuracy: ~85.5% 
ROC-AUC: ~84.7% 
Highest F1-Score: ~77.9% 
Robust predictions
Best For: Production deployments, maximum accuracy
Loading:

import joblib
model = joblib.load('models/random_forest.joblib')
prediction = model.predict(X_test)
probability = model.predict_proba(X_test)[:, 1]

4. scaler.joblib
Type: StandardScaler (Feature Normalization)
Purpose: Scales features to mean=0, std=1
Required: YES - Must be applied to new data before prediction
Size: ~1 KB
Loading & Using:

import joblib
scaler = joblib.load('models/scaler.joblib')
# Scale new data
new_data = [[50000, 15000, 20, 0.5, 60, 5, 8, 2]]
scaled_data = scaler.transform(new_data)
# Then predict
model = joblib.load('models/random_forest.joblib')
prediction = model.predict(scaled_data)