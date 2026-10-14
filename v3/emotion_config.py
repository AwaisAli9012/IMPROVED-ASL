"""
Emotion Classification Config (4-Class Setup)
==============================================
Emotions: Happy, Neutral, Sad, Angry
Using face landmarks (468 points × 3 coords = 1404 dimensions)
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# Emotion Classes (Restricted to 4 core classes)
EMOTIONS = {
    0: 'happy',
    1: 'neutral',
    2: 'sad',
    3: 'angry'
}

# Paths
EMOTION_KEYPOINTS_DIR = BASE_DIR / "emotion_keypoints_extracted"
EMOTION_MODELS_DIR = BASE_DIR / "Emotion_Models"

# Your actual dataset path
EMOTION_DATASET_PATH = Path("/home/xero1/Documents/IMPROVED ASL/Dataset/emotion_dataset/emotion/train")

# Training Config
EMOTION_TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_splits': 5,
    'n_jobs': -1
}

print(f"Emotion config initialized with {len(EMOTIONS)} classes")
print(f"Emotions: {EMOTIONS}")