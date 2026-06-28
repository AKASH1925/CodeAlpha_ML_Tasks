"""
Handwritten Character Recognition - Main Entry Point
=====================================================
This script provides a unified interface for training, evaluating,
and making predictions with the handwritten character recognition model.
Usage:
    python main.py --mode train --dataset emnist --split letters
    python main.py --mode train --dataset mnist
    python main.py --mode evaluate --model_path models/emnist_letters_best.pth
    python main.py --mode predict --image_path sample.png --model_path models/emnist_letters_best.pth
"""
import os
import sys
import argparse
import torch
import numpy as np
from src.model import get_model
from src.data_loader import load_mnist, load_emnist
from src.train import Trainer
from src.evaluate import evaluate_model
from src.predict import Predictor, load_model_for_prediction
from src.utils import (
    plot_training_history,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    visualize_predictions,
    show_samples,
    count_parameters,
)
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Handwritten Character Recognition using CNN"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "predict", "demo"],
        help="Mode: train, evaluate, predict, or demo",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="emnist",
        choices=["mnist", "emnist"],
        help="Dataset to use (emnist for characters A-Z, mnist for digits 0-9)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="letters",
        choices=["letters", "digits", "byclass", "balanced"],
        help="EMNIST split (only used with --dataset emnist)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate"
    )
    parser.add_argument(
        "--augment", action="store_true", help="Use data augmentation"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to saved model"
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="Path to image for prediction"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    return parser.parse_args()
def setup_device():
    """Setup computation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
def train_mode(args, device):
    """Train the model."""
    print("\n" + "=" * 60)
    print("TRAINING MODE")
    print("=" * 60)
    # Load data
    if args.dataset == "mnist":
        train_loader, val_loader, test_loader, class_names = load_mnist(
            data_dir=args.data_dir, batch_size=args.batch_size, augment=args.augment
        )
        num_classes = 10
    else:
        train_loader, val_loader, test_loader, class_names = load_emnist(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            split=args.split,
            augment=args.augment,
        )
        split_map = {"letters": 26, "digits": 10, "byclass": 62, "balanced": 47}
        num_classes = split_map[args.split]
    # Create model
    model = get_model(
        dataset=args.dataset if args.dataset == "mnist" else f"emnist_{args.split}",
        dropout_rate=args.dropout,
    )
    # Print model summary
    param_info = count_parameters(model)
    print(f"\nModel Parameters: {param_info['total']:,} total, {param_info['trainable']:,} trainable")
    # Create trainer and train
    trainer = Trainer(model, device=device, learning_rate=args.lr)
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        save_dir="./models",
        model_name=f"{args.dataset}_{args.split if args.dataset == 'emnist' else 'digits'}",
        patience=10,
    )
    # Plot training history
    os.makedirs(args.output_dir, exist_ok=True)
    plot_training_history(
        history, save_path=os.path.join(args.output_dir, "training_history.png")
    )
    # Show sample images
    show_samples(
        train_loader,
        class_names=class_names,
        save_path=os.path.join(args.output_dir, "sample_images.png"),
    )
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, device=device, class_names=class_names)
    # Plot confusion matrix
    plot_confusion_matrix(
        results["confusion_matrix"],
        class_names=class_names,
        save_path=os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    # Plot per-class accuracy
    plot_per_class_accuracy(
        results["per_class_accuracy"],
        save_path=os.path.join(args.output_dir, "per_class_accuracy.png"),
    )
    # Visualize predictions
    visualize_predictions(
        model,
        test_loader,
        class_names,
        device=device,
        save_path=os.path.join(args.output_dir, "predictions.png"),
    )
    print(f"\nAll outputs saved to {args.output_dir}/")
    return results
def evaluate_mode(args, device):
    """Evaluate a saved model."""
    print("\n" + "=" * 60)
    print("EVALUATION MODE")
    print("=" * 60)
    if args.model_path is None:
        print("Error: --model_path is required for evaluation mode")
        sys.exit(1)
    # Load test data
    if args.dataset == "mnist":
        _, _, test_loader, class_names = load_mnist(
            data_dir=args.data_dir, batch_size=args.batch_size
        )
        num_classes = 10
    else:
        _, _, test_loader, class_names = load_emnist(
            data_dir=args.data_dir, batch_size=args.batch_size, split=args.split
        )
        split_map = {"letters": 26, "digits": 10, "byclass": 62, "balanced": 47}
        num_classes = split_map[args.split]
    # Load model
    model, checkpoint = load_model_for_prediction(
        args.model_path, num_classes=num_classes, device=device
    )
    # Evaluate
    results = evaluate_model(model, test_loader, device=device, class_names=class_names)
    # Visualize predictions
    os.makedirs(args.output_dir, exist_ok=True)
    visualize_predictions(
        model,
        test_loader,
        class_names,
        device=device,
        save_path=os.path.join(args.output_dir, "eval_predictions.png"),
    )
    return results
def predict_mode(args, device):
    """Make predictions on a single image."""
    print("\n" + "=" * 60)
    print("PREDICTION MODE")
    print("=" * 60)
    if args.model_path is None:
        print("Error: --model_path is required for prediction mode")
        sys.exit(1)
    if args.image_path is None:
        print("Error: --image_path is required for prediction mode")
        sys.exit(1)
    # Determine number of classes from checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 26)
    # Load model
    model, _ = load_model_for_prediction(
        args.model_path, num_classes=num_classes, device=device
    )
    # Create class names
    if num_classes == 26:
        class_names = [chr(ord("A") + i) for i in range(26)]
    elif num_classes == 10:
        class_names = [str(i) for i in range(10)]
    else:
        class_names = [str(i) for i in range(num_classes)]
    # Create predictor and predict
    predictor = Predictor(model, device=device, class_names=class_names)
    result = predictor.predict_single(args.image_path, return_probabilities=True)
    # Display results
    print(f"\nPrediction Results:")
    print(f"  Image: {args.image_path}")
    print(f"  Predicted Character: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    if "probabilities" in result:
        print(f"\n  Top 5 predictions:")
        sorted_probs = sorted(
            result["probabilities"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        for char, prob in sorted_probs:
            print(f"    {char}: {prob:.2%}")
    return result
def demo_mode(args, device):
    """Run a quick demo with random samples."""
    print("\n" + "=" * 60)
    print("DEMO MODE - Quick Test (EMNIST Letters A-Z)")
    print("=" * 60)
    # Load EMNIST letters for quick demo
    _, _, test_loader, class_names = load_emnist(
        data_dir=args.data_dir, batch_size=16, split="letters"
    )
    # Create and evaluate model (26 classes for letters)
    model = get_model(dataset="emnist_letters", dropout_rate=args.dropout)
    # Quick evaluation (no training)
    results = evaluate_model(model, test_loader, device=device, class_names=class_names)
    # Visualize random predictions
    os.makedirs(args.output_dir, exist_ok=True)
    visualize_predictions(
        model,
        test_loader,
        class_names,
        device=device,
        save_path=os.path.join(args.output_dir, "demo_predictions.png"),
    )
    print("\nDemo complete! Check outputs/ directory for visualizations.")
    return results
def main():
    """Main function."""
    args = parse_args()
    device = setup_device()
    if args.mode == "train":
        results = train_mode(args, device)
    elif args.mode == "evaluate":
        results = evaluate_mode(args, device)
    elif args.mode == "predict":
        results = predict_mode(args, device)
    elif args.mode == "demo":
        results = demo_mode(args, device)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
if __name__ == "__main__":
    main()