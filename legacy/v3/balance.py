import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

print("Loading raw extracted multimodal features...")
X = np.load("Dataset_Multimodal/X_multimodal.npy")
y = np.load("Dataset_Multimodal/y_multimodal.npy")

# 1. Clean out pollution ('asl_alphabet_test') and zero-vector emotion/idle classes
invalid_classes = {
    'asl_alphabet_test', 'angry', 'disgust', 'fear', 
    'happy', 'neutral', 'sad', 'surprise', 'nothing'
}
valid_mask = ~np.isin(y, list(invalid_classes))
X = X[valid_mask]
y = y[valid_mask]

# Target sample bounds per class
MAX_SAMPLES_PER_CLASS = 1000
MIN_SAMPLES_PER_CLASS = 500

print("\n--- STEP 1: Capping Majority Classes ---")
X_capped_list, y_capped_list = [], []
np.random.seed(42)

for class_name in np.unique(y):
    idx = np.where(y == class_name)[0]
    if len(idx) > MAX_SAMPLES_PER_CLASS:
        chosen_idx = np.random.choice(idx, size=MAX_SAMPLES_PER_CLASS, replace=False)
    else:
        chosen_idx = idx
    
    X_capped_list.append(X[chosen_idx])
    y_capped_list.append(y[chosen_idx])

X_capped = np.vstack(X_capped_list)
y_capped = np.concatenate(y_capped_list)

print(f"Capped majority classes at {MAX_SAMPLES_PER_CLASS} samples.")

print("\n--- STEP 2: Exact Oversampling & Landmark Jitter Augmentation ---")
counts = Counter(y_capped)
sampling_strategy = {
    c: max(count, MIN_SAMPLES_PER_CLASS) for c, count in counts.items()
}

# Oversample by repeating true physical hand landmark configurations
ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_capped, y_capped)

# Apply micro Gaussian jitter to oversampled duplicated frames to simulate natural hand tremor
# (Only applies to non-zero landmark coordinates)
duplicated_mask = np.zeros(len(y_resampled), dtype=bool)
duplicated_mask[len(y_capped):] = True  # Identify generated duplicate rows

# Add micro Gaussian noise (sigma = 0.005) to duplicate landmarks
noise = np.random.normal(0, 0.005, size=X_resampled[duplicated_mask].shape).astype(np.float32)
non_zero_mask = (X_resampled[duplicated_mask] != 0)
X_resampled[duplicated_mask] += noise * non_zero_mask

X_balanced, y_balanced = X_resampled, y_resampled

print("\n" + "=" * 60)
print("✓ DATASET BALANCING COMPLETE (JITTER-AUGMENTED RESAMPLING)")
print(f"✓ Total Balanced Samples: {len(y_balanced):,}")
print(f"✓ Total Unique Classes:  {len(np.unique(y_balanced))}")
print(f"✓ Min Samples / Class:   {min(Counter(y_balanced).values())}")
print(f"✓ Max Samples / Class:   {max(Counter(y_balanced).values())}")
print("=" * 60)

# Save balanced dataset
np.save("Dataset_Multimodal/X_balanced.npy", X_balanced)
np.save("Dataset_Multimodal/y_balanced.npy", y_balanced)

# Save summary
df_balanced = pd.DataFrame(Counter(y_balanced).items(), columns=['Class', 'Samples']).sort_values(by='Samples', ascending=False)
df_balanced.to_csv("Dataset_Multimodal/balanced_class_distribution.csv", index=False)
print("✓ Saved balanced files to 'Dataset_Multimodal/X_balanced.npy' and 'y_balanced.npy'")