"""
Download FER2013 Emotion Dataset
=================================
5 emotions × ~5000 samples = ~25000 images
"""

import os
import zipfile
from pathlib import Path

print("=" * 70)
print("EMOTION DATASET SETUP")
print("=" * 70)

print("\n⚠️  MANUAL DOWNLOAD REQUIRED:")
print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
print("2. Download fer2013.csv")
print("3. Place in: Dataset/fer2013.csv")
print("4. Then run: python extract_emotion_landmarks.py")

print("\nOR use this smaller free dataset:")
print("https://github.com/Akashbarlogs/FER2013-Real-Time-Emotion-Detection-Dataset")

print("\nDataset structure needed:")
print("""
Dataset/
├── emotion_dataset/
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── surprised/
│   └── neutral/
""")

print("\nAfter placing images, run:")
print("python extract_emotion_landmarks.py")
