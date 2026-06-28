"""
Prediction Script for Handwritten Character Recognition
========================================================
Handles single image inference and batch predictions.
Primary focus: EMNIST Letters (A-Z) character recognition.
"""
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
class Predictor:
    """Handles inference for handwritten character recognition."""
    def __init__(self, model, device="cpu", class_names=None):
        """
        Initialize the predictor.
        Args:
            model: Trained PyTorch model
            device: Device for inference
            class_names: List of class names
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.class_names = class_names or [chr(ord("A") + i) for i in range(26)]  # Default: A-Z letters
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    def predict_single(self, image, return_probabilities=False):
        """
        Predict a single image.
        Args:
            image: PIL Image, numpy array, or file path
            return_probabilities (bool): Return all class probabilities
        Returns:
            dict: Prediction result
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        # Inference
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        result = {
            "predicted_class": self.class_names[predicted.item()],
            "confidence": confidence.item(),
            "predicted_index": predicted.item(),
        }
        if return_probabilities:
            result["probabilities"] = {
                self.class_names[i]: float(probabilities[0][i])
                for i in range(len(self.class_names))
            }
        return result
    def predict_batch(self, images):
        """
        Predict a batch of images.
        Args:
            images: List of PIL Images, numpy arrays, or file paths
        Returns:
            list: List of prediction results
        """
        results = []
        for image in images:
            result = self.predict_single(image)
            results.append(result)
        return results
    def predict_from_tensor(self, tensor):
        """
        Predict from a preprocessed tensor.
        Args:
            tensor: Preprocessed tensor of shape (1, 1, 28, 28) or (N, 1, 28, 28)
        Returns:
            dict or list: Prediction results
        """
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        if tensor.dim() == 4 and tensor.size(0) == 1:
            return {
                "predicted_class": self.class_names[predicted.item()],
                "confidence": confidence.item(),
            }
        results = []
        for i in range(tensor.size(0)):
            results.append(
                {
                    "predicted_class": self.class_names[predicted[i].item()],
                    "confidence": confidence[i].item(),
                }
            )
        return results
def load_model_for_prediction(model_path, num_classes=26, device="cpu"):
    """
    Load a saved model for prediction.
    Args:
        model_path (str): Path to saved model checkpoint
        num_classes (int): Number of classes (default 26 for A-Z letters)
        device: Device to load model on
    Returns:
        tuple: (model, checkpoint)
    """
    from .model import get_model
    # Determine dataset from num_classes
    dataset_map = {26: "emnist_letters", 10: "mnist", 36: "emnist_full", 47: "emnist_balanced"}
    dataset = dataset_map.get(num_classes, "emnist_letters")
    model = get_model(dataset=dataset)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    print(f"Checkpoint accuracy: {checkpoint.get('accuracy', 'N/A')}")
    return model, checkpoint