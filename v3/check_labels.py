import numpy as np
from pathlib import Path

# Load labels
LABEL_FILE = Path(__file__).resolve().parent / "Dataset_Multimodal" / "y_multimodal.npy"

if not LABEL_FILE.exists():
    print(f"Error: Could not find {LABEL_FILE}")
    exit()

y_data = np.load(LABEL_FILE)

print("=" * 60)
print(f"Total Samples: {len(y_data)}")
print("=" * 60)

# Count frequency of each label
unique_labels, counts = np.unique(y_data, return_counts=True)

print("\nDetected Class Labels & Counts:")
for label, count in zip(unique_labels, counts):
    print(f"  - '{label}': {count:,} samples")

print("=" * 60)