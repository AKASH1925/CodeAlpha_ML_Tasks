# Models Directory
This directory stores serialized trained models (`.pkl` files)
generated during training runs.
## Usage
Models are saved automatically when training is complete and can be
reloaded for inference without retraining:
```python
import joblib
model = joblib.load("models/heart_random_forest.pkl")
prediction = model.predict(new_data)