"""
Training Module
Train multiple classification models for credit scoring.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
def get_models() -> dict:
    """
    Get dictionary of models to train.
    
    Returns:
        Dictionary of model names and sklearn estimators
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    }
    return models
def train_models(X_train, y_train, cv: int = 5) -> dict:
    """
    Train all models with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary of trained models
    """
    models = get_models()
    trained_models = {}
    
    print("=" * 60)
    print("Training Models with Cross-Validation")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"  Training complete!")
    
    print("\n" + "=" * 60)
    
    return trained_models
def save_model(model, model_name: str, save_dir: str = 'models') -> str:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained sklearn model
        model_name: Name of the model
        save_dir: Directory to save model
        
    Returns:
        Path to saved model
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Clean model name for filename
    clean_name = model_name.lower().replace(' ', '_')
    filepath = Path(save_dir) / f"{clean_name}.joblib"
    
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    
    return str(filepath)
def load_model(model_path: str):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded sklearn model
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import prepare_data
    
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    models = train_models(X_train, y_train)
    
    for name, model in models.items():
        save_model(model, name)