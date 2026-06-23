"""
Credit Scoring Model - Main Script
===================================
This script implements a complete credit scoring system that predicts
individual creditworthiness using classification algorithms.
Task 1: Credit Scoring Model
CodeAlpha Machine Learning Internship
Author: [Your Name]
Date: [Date]
"""
import sys
from pathlib import Path
# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.data_loader import load_data, generate_synthetic_data
from src.preprocessing import prepare_data
from src.train import train_models, save_model
from src.evaluate import (
    evaluate_model, 
    print_metrics, 
    generate_all_plots,
    plot_metrics_comparison
)
def main():
    """Main execution pipeline."""
    
    print("=" * 70)
    print("CREDIT SCORING MODEL")
    print("CodeAlpha Machine Learning Internship - Task 1")
    print("=" * 70)
    
    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    print("\n[1/5] Loading Dataset...")
    print("-" * 50)
    
    data_path = Path('data/credit_data.csv')
    df = load_data(data_path if data_path.exists() else None)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nTarget Distribution:")
    print(df['target'].value_counts().rename({0: 'Not Creditworthy', 1: 'Creditworthy'}))
    
    # ============================================================
    # STEP 2: Preprocess Data
    # ============================================================
    print("\n[2/5] Preprocessing Data...")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # ============================================================
    # STEP 3: Train Models
    # ============================================================
    print("\n[3/5] Training Models...")
    print("-" * 50)
    
    models = train_models(X_train, y_train)
    
    # ============================================================
    # STEP 4: Evaluate Models
    # ============================================================
    print("\n[4/5] Evaluating Models...")
    print("-" * 50)
    
    all_metrics = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        print_metrics(metrics)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    import pandas as pd
    metrics_df = pd.DataFrame(all_metrics).set_index('Model')
    print(metrics_df.to_string())
    
    # Find best model
    best_model_name = metrics_df['F1-Score'].idxmax()
    best_model = models[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {metrics_df.loc[best_model_name, 'F1-Score']:.4f})")
    
    # ============================================================
    # STEP 5: Save Models and Generate Plots
    # ============================================================
    print("\n[5/5] Saving Models and Generating Visualizations...")
    print("-" * 50)
    
    # Save models
    for name, model in models.items():
        save_model(model, name, 'models')
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'models/scaler.joblib')
    print(f"Scaler saved to models/scaler.joblib")
    
    # Generate plots
    feature_names = X_train.columns.tolist()
    generate_all_plots(models, X_test, y_test, feature_names, 'reports/figures')
    
    # Generate comparison plot
    plot_metrics_comparison(all_metrics, 'reports/figures/metrics_comparison.png')
    
    # ============================================================
    # DONE
    # ============================================================
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nOutputs:")
    print(f"  📁 Models:     models/")
    print(f"  📊 Plots:      reports/figures/")
    print(f"  📈 Metrics:    See comparison table above")
    print("\n" + "=" * 70)
    
    return models, metrics_df
if __name__ == "__main__":
    models, metrics = main()