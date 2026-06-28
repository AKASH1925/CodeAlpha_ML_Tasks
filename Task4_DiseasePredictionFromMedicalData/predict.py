#!/usr/bin/env python3
"""
Disease Prediction from Medical Data
=====================================
Main script that runs SVM, Logistic Regression, Random Forest, and XGBoost
on Heart Disease, Diabetes, and Breast Cancer datasets.
Usage:
    python predict.py                      # Run all datasets
    python predict.py --dataset heart      # Heart Disease only
    python predict.py --dataset diabetes   # Diabetes only
    python predict.py --dataset breast_cancer  # Breast Cancer only
"""
import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_dataset, DATASETS
from src.preprocessing import preprocess
from src.models import get_models
from src.evaluate import evaluate_model, print_report
from src.visualize import generate_all_plots
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
def run_pipeline(dataset_name: str) -> tuple:
    """
    Run all models on a single dataset.
    Returns:
        (dataset_results, roc_dataset_data)
        dataset_results: { model_name: metrics_dict }
        roc_dataset_data: { model_name: { fpr, tpr, auc } }
    """
    display_name, _ = DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"  {display_name.upper()} DATASET")
    print(f"{'='*60}")
    # Load data
    df = load_dataset(dataset_name)
    print(f"  Samples: {len(df)} | Features: {len(df.columns) - 1} | Target: 'target'")
    print(f"  Class distribution:\n{df['target'].value_counts().to_string()}\n")
    # Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)
    # Get models
    models = get_models()
    dataset_results = {}
    roc_dataset_data = {}
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        # Evaluate
        results = evaluate_model(y_test, y_pred, y_prob)
        print_report(name, results)
        dataset_results[name] = results
        # ROC data
        if y_prob is not None:
            prob_positive = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            fpr, tpr, _ = roc_curve(y_test, prob_positive)
            roc_dataset_data[name] = {"fpr": fpr, "tpr": tpr, "auc": results.get("roc_auc", 0)}
    return dataset_results, roc_dataset_data
def print_summary_table(all_results: dict):
    """Print a consolidated comparison table."""
    print(f"\n{'='*80}")
    print("  SUMMARY COMPARISON TABLE")
    print(f"{'='*80}")
    rows = []
    for dataset, results in all_results.items():
        for model, metrics in results.items():
            rows.append({
                "Dataset": dataset,
                "Model": model,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1 Score": f"{metrics['f1']:.4f}",
                "ROC AUC": f"{metrics.get('roc_auc', 0):.4f}",
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
def main():
    parser = argparse.ArgumentParser(description="Disease Prediction from Medical Data")
    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        default=None,
        help="Run on a specific dataset (default: all)",
    )
    args = parser.parse_args()
    print("=" * 60)
    print("  DISEASE PREDICTION FROM MEDICAL DATA")
    print("  Algorithms: SVM | Logistic Regression | Random Forest | XGBoost")
    print("=" * 60)
    # Determine datasets to run
    if args.dataset:
        datasets_to_run = [args.dataset]
    else:
        datasets_to_run = list(DATASETS.keys())
    # Run pipelines
    all_results = {}
    all_roc_data = {}
    for dataset_name in datasets_to_run:
        results, roc_data = run_pipeline(dataset_name)
        all_results[dataset_name] = results
        all_roc_data[dataset_name] = roc_data
    # Print summary
    print_summary_table(all_results)
    # Generate visualizations
    generate_all_plots(all_results, all_roc_data)
    print("\n" + "=" * 60)
    print("  DONE — All models trained and evaluated successfully!")
    print("=" * 60)
if __name__ == "__main__":
    main()