"""
Models — Define all four classifiers.
"""
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
def get_models() -> dict:
    """
    Return a dictionary of model name -> sklearn-compatible classifier.
    All models use consistent hyperparameters suitable for medical classification.
    """
    return {
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    }