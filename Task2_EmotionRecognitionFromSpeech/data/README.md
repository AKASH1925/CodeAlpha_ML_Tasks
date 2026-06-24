# Dataset Directory
## How to Set Up Your Dataset
This project supports several speech emotion recognition datasets. Follow the instructions below to set up your data.
### Supported Datasets
#### 1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Download**: https://zenodo.org/record/1188976
- **Emotions**: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **Structure**: The dataset contains files like `03-01-01-01-01-01-01.wav`
#### 2. TESS (Toronto Emotional Speech Set)
- **Download**: https://tspace.library.utoronto.ca/handle/1807/24487
- **Emotions**: happy, sad, angry, fear, disgust, surprise, neutral
- **Structure**: Files are named like `OAF_happy_angry.wav` or `YAF_happy_angry.wav`
#### 3. EMO-DB (Berlin Database of Emotional Speech)
- **Download**: https://www.kaggle.com/datasets/piyushagnihotri/emodb
- **Emotions**: angry, boredom, disgust, fear, happy, neutral, sad
- **Structure**: Files are named like `03a01Fa.wav`
### Directory Structure Options
#### Option 1: Organized by Emotion (Recommended)
#### Option 2: Flat Structure (with emotion in filename)

### Quick Start
1. **Download** one of the supported datasets
2. **Extract** the audio files
3. **Place** them in the `data/` directory (using either structure above)
4. **Run** the training script:
   ```bash
   python main.py --mode train --dataset_path data/