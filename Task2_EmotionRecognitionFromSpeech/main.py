"""
Emotion Recognition from Speech - Main Entry Point
====================================================
A deep learning project for recognizing human emotions from speech audio.
Usage:
    python main.py --mode train --dataset_path data/
    python main.py --mode evaluate --model_path models/best_model.pth
    python main.py --mode predict --audio_path path/to/audio.wav
"""
import argparse
import os
import sys
import numpy as np
import torch
import librosa
import warnings
warnings.filterwarnings("ignore")
from src.data_loader import load_dataset, create_data_loaders, SpeechEmotionDataset
from src.feature_extraction import AudioFeatureExtractor
from src.model import create_model, model_summary
from src.train import Trainer, train_model
from src.evaluate import Evaluator, predict_emotion
from src.utils import set_seed, get_device, get_emotion_labels, create_directory
def train_pipeline(args):
    """Training pipeline."""
    print("\n" + "=" * 60)
    print("EMOTION RECOGNITION FROM SPEECH - TRAINING")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    audio_paths, emotions = load_dataset(
        dataset_path=args.dataset_path,
        sr=args.sample_rate,
        duration=args.duration
    )
    
    if len(audio_paths) == 0:
        print("\n⚠️  No audio files found!")
        print("Please ensure your dataset is in the correct format:")
        print("  - Place .wav files in the data/ directory")
        print("  - Or use subdirectories named after emotions")
        print("  - See data/README.md for details")
        return
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        audio_paths=audio_paths,
        emotions=emotions,
        batch_size=args.batch_size,
        sr=args.sample_rate,
        duration=args.duration
    )
    
    # Train model
    print("\nStarting training...")
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(get_emotion_labels()),
        model_type=args.model_type,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        results_dir=args.results_dir
    )
    
    results = evaluator.run_full_evaluation(history=history)
    
    print("\n✅ Training and evaluation complete!")
    print(f"Model saved to: {args.save_dir}/")
    print(f"Results saved to: {args.results_dir}/")
def evaluate_pipeline(args):
    """Evaluation pipeline."""
    print("\n" + "=" * 60)
    print("EMOTION RECOGNITION FROM SPEECH - EVALUATION")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = create_model(args.model_type, num_classes=len(get_emotion_labels()))
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from: {args.model_path}")
    else:
        print(f"⚠️  Model not found at: {args.model_path}")
        print("Using randomly initialized model for demo...")
    
    model = model.to(device)
    model.eval()
    
    # Load dataset for evaluation
    if args.dataset_path:
        print("\nLoading test dataset...")
        audio_paths, emotions = load_dataset(
            dataset_path=args.dataset_path,
            sr=args.sample_rate,
            duration=args.duration
        )
        
        _, _, test_loader = create_data_loaders(
            audio_paths=audio_paths,
            emotions=emotions,
            batch_size=args.batch_size,
            sr=args.sample_rate,
            duration=args.duration,
            num_workers=2
        )
        
        # Evaluate
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            results_dir=args.results_dir
        )
        
        results = evaluator.run_full_evaluation()
    else:
        print("\n⚠️  No dataset path provided for evaluation")
def predict_pipeline(args):
    """Prediction pipeline."""
    print("\n" + "=" * 60)
    print("EMOTION RECOGNITION FROM SPEECH - PREDICTION")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    # Load model
    print("\nLoading model...")
    model = create_model(args.model_type, num_classes=len(get_emotion_labels()))
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from: {args.model_path}")
    else:
        print(f"⚠️  Model not found at: {args.model_path}")
        print("Using randomly initialized model for demo...")
    
    model = model.to(device)
    model.eval()
    
    # Load and preprocess audio
    print(f"\nLoading audio: {args.audio_path}")
    audio, _ = librosa.load(args.audio_path, sr=args.sample_rate, duration=args.duration)
    
    # Pad or truncate to fixed length
    max_length = int(args.sample_rate * args.duration)
    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)), mode="constant")
    else:
        audio = audio[:max_length]
    
    # Predict
    print("\nPredicting emotion...")
    emotion, confidence = predict_emotion(model, audio, device)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"\n  Emotion:    {emotion.upper()}")
    print(f"  Confidence: {confidence*100:.2f}%")
    print("=" * 60)
    
    # Display all emotion probabilities
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(audio_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    
    emotion_labels = get_emotion_labels()
    print("\nEmotion Probabilities:")
    print("-" * 30)
    for idx, label in emotion_labels.items():
        prob = probs[idx] * 100
        bar = "█" * int(prob / 2)
        print(f"  {label:12s}: {prob:6.2f}% {bar}")
def demo_pipeline(args):
    """Demo pipeline with synthetic data."""
    print("\n" + "=" * 60)
    print("EMOTION RECOGNITION FROM SPEECH - DEMO")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create synthetic dataset for demonstration
    print("\nCreating synthetic dataset for demonstration...")
    num_samples = 200
    audio_length = args.sample_rate * args.duration
    
    # Generate synthetic audio (sine waves with different frequencies per emotion)
    audio_paths = []
    emotions = []
    emotion_list = list(get_emotion_labels().values())
    
    for i in range(num_samples):
        # Create a simple synthetic audio file
        emotion_idx = i % len(emotion_list)
        emotion = emotion_list[emotion_idx]
        
        # Different base frequency per emotion
        freq = 200 + emotion_idx * 50
        t = np.linspace(0, args.duration, audio_length)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        audio += 0.1 * np.random.randn(audio_length)
        
        # Save temporary file
        temp_path = os.path.join(args.save_dir, f"temp_{i}.wav")
        os.makedirs(args.save_dir, exist_ok=True)
        librosa.output.write_wav(temp_path, audio, args.sample_rate) if hasattr(librosa, 'output') else None
        
        audio_paths.append(temp_path)
        emotions.append(emotion)
    
    print(f"Created {num_samples} synthetic samples")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        audio_paths=audio_paths,
        emotions=emotions,
        batch_size=args.batch_size,
        sr=args.sample_rate,
        duration=args.duration,
        num_workers=2
    )
    
    # Train with fewer epochs for demo
    print("\nTraining demo model (10 epochs)...")
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(get_emotion_labels()),
        model_type="simple_cnn",  # Use simpler model for demo
        num_epochs=10,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    # Evaluate
    print("\nEvaluating demo model...")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        results_dir=args.results_dir
    )
    
    results = evaluator.run_full_evaluation(history=history)
    
    # Clean up temporary files
    for path in audio_paths:
        if os.path.exists(path):
            os.remove(path)
    
    print("\n✅ Demo complete!")
    print("This demonstrates the full training pipeline.")
    print("Use your own dataset for real training!")
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Emotion Recognition from Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with RAVDESS dataset
  python main.py --mode train --dataset_path data/RAVDESS --epochs 50
  
  # Evaluate trained model
  python main.py --mode evaluate --model_path models/best_model.pth --dataset_path data/
  
  # Predict emotion from audio file
  python main.py --mode predict --audio_path sample.wav --model_path models/best_model.pth
  
  # Run demo with synthetic data
  python main.py --mode demo --epochs 10
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "predict", "demo"],
        default="demo",
        help="Pipeline mode (default: demo)"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to audio file for prediction"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["cnn_lstm", "simple_cnn"],
        default="cnn_lstm",
        help="Model architecture (default: cnn_lstm)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.pth",
        help="Path to saved model"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Audio sample rate (default: 22050)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Audio duration in seconds (default: 3.0)"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Directory to save models (default: models)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    
    args = parser.parse_args()
    
    # Run appropriate pipeline
    if args.mode == "train":
        train_pipeline(args)
    elif args.mode == "evaluate":
        evaluate_pipeline(args)
    elif args.mode == "predict":
        if args.audio_path is None:
            parser.error("--audio_path is required for predict mode")
        predict_pipeline(args)
    elif args.mode == "demo":
        demo_pipeline(args)
if __name__ == "__main__":
    main()