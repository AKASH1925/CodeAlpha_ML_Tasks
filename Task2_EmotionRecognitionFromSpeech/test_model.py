"""
Test script to verify the model works correctly.
Run this to ensure all components are functioning.
"""
import torch
import numpy as np
from src.model import EmotionRecognitionModel, SimpleCNNModel, create_model, model_summary
from src.feature_extraction import AudioFeatureExtractor
from src.utils import get_emotion_labels, get_device, set_seed
import sys
def test_model_creation():
    """Test model creation and forward pass."""
    print("=" * 60)
    print("TEST 1: Model Creation")
    print("=" * 60)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Test CNN-LSTM model
    print("\nTesting CNN-LSTM Model...")
    model1 = create_model("cnn_lstm", num_classes=7)
    model1 = model1.to(device)
    
    # Create dummy input (batch_size, 1, height, width)
    dummy_input = torch.randn(2, 1, 128, 129).to(device)
    output = model1(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 7), "Output shape mismatch!"
    print("  ✓ CNN-LSTM model works!")
    
    # Test Simple CNN model
    print("\nTesting Simple CNN Model...")
    model2 = create_model("simple_cnn", num_classes=7)
    model2 = model2.to(device)
    
    output2 = model2(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output2.shape}")
    assert output2.shape == (2, 7), "Output shape mismatch!"
    print("  ✓ Simple CNN model works!")
    
    print("\n✅ All models created successfully!\n")
    return True
def test_feature_extraction():
    """Test feature extraction."""
    print("=" * 60)
    print("TEST 2: Feature Extraction")
    print("=" * 60)
    
    extractor = AudioFeatureExtractor()
    
    # Create dummy audio (3 seconds at 22050 Hz)
    dummy_audio = np.random.randn(22050 * 3).astype(np.float32)
    print(f"Input audio length: {len(dummy_audio)} samples ({len(dummy_audio)/22050:.2f}s)")
    
    # Test MFCC extraction
    print("\nTesting MFCC extraction...")
    mfcc = extractor.extract_mfcc(dummy_audio)
    print(f"  MFCC shape: {mfcc.shape}")
    assert mfcc.shape[0] == 120, "MFCC shape mismatch (should be 40*3 for mfcc+deltas)!"
    print("  ✓ MFCC extraction works!")
    
    # Test mel spectrogram
    print("\nTesting Mel Spectrogram extraction...")
    mel = extractor.extract_mel_spectrogram(dummy_audio)
    print(f"  Mel spectrogram shape: {mel.shape}")
    print("  ✓ Mel Spectrogram extraction works!")
    
    # Test combined features
    print("\nTesting combined features...")
    combined = extractor.extract_combined_features(dummy_audio)
    print(f"  Combined features shape: {combined.shape}")
    print("  ✓ Combined feature extraction works!")
    
    print("\n✅ All feature extraction tests passed!\n")
    return True
def test_model_training():
    """Test model training loop."""
    print("=" * 60)
    print("TEST 3: Training Loop (Mini Test)")
    print("=" * 60)
    
    set_seed(42)
    device = get_device()
    
    # Create small dummy dataset
    batch_size = 4
    num_samples = 16
    num_classes = 7
    
    # Random audio data
    X = torch.randn(num_samples, 1, 128, 129)
    y = torch.randint(0, num_classes, (num_samples,))
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = create_model("simple_cnn", num_classes=num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 2 epochs
    print("\nTraining for 2 epochs...")
    model.train()
    
    for epoch in range(2):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        acc = 100. * correct / total
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={acc:.2f}%")
    
    print("  ✓ Training loop works!")
    
    print("\n✅ Training test passed!\n")
    return True
def test_emotion_labels():
    """Test emotion labels."""
    print("=" * 60)
    print("TEST 4: Emotion Labels")
    print("=" * 60)
    
    labels = get_emotion_labels()
    
    print(f"\nNumber of emotions: {len(labels)}")
    print("Emotion mapping:")
    for idx, emotion in labels.items():
        print(f"  {idx}: {emotion}")
    
    assert len(labels) == 7, "Should have 7 emotions!"
    print("\n✓ Emotion labels correct!")
    
    print("\n✅ Emotion label test passed!\n")
    return True
def test_prediction():
    """Test prediction pipeline."""
    print("=" * 60)
    print("TEST 5: Prediction Pipeline")
    print("=" * 60)
    
    device = get_device()
    
    # Create and initialize model
    model = create_model("simple_cnn", num_classes=7)
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_audio = np.random.randn(22050 * 3).astype(np.float32)
    
    # Process audio
    max_length = 22050 * 3
    if len(dummy_audio) < max_length:
        dummy_audio = np.pad(dummy_audio, (0, max_length - len(dummy_audio)))
    else:
        dummy_audio = dummy_audio[:max_length]
    
    # Convert to tensor
    audio_tensor = torch.FloatTensor(dummy_audio).unsqueeze(0).unsqueeze(0)
    audio_tensor = audio_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(audio_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
    
    emotion_labels = get_emotion_labels()
    emotion = emotion_labels[predicted.item()]
    
    print(f"\nInput audio length: {len(dummy_audio)} samples")
    print(f"Predicted emotion: {emotion}")
    print(f"Confidence: {confidence.item()*100:.2f}%")
    print(f"All probabilities: {probs[0].cpu().numpy()}")
    
    print("\n✓ Prediction pipeline works!")
    
    print("\n✅ Prediction test passed!\n")
    return True
def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EMOTION RECOGNITION - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Feature Extraction", test_feature_extraction),
        ("Training Loop", test_model_training),
        ("Emotion Labels", test_emotion_labels),
        ("Prediction Pipeline", test_prediction),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your model is ready to use!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above.")
    print("=" * 60 + "\n")
    
    return all_passed
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)