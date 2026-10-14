import numpy as np

X = np.load("Dataset_Multimodal/X_multimodal.npy")
y = np.load("Dataset_Multimodal/y_multimodal.npy")

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("--- DIAGNOSTIC: Checking Feature Vectors for Emotions ---")
for emo in emotions:
    idx = np.where(y == emo)[0]
    if len(idx) > 0:
        sample = X[idx[0]]
        non_zero = np.count_nonzero(sample)
        print(f"Class: {emo:<10} | Total Features: {len(sample)} | Non-Zero Features: {non_zero}")
    else:
        print(f"Class: {emo:<10} | NOT FOUND in dataset")

# Check 'nothing' class as well
idx_nothing = np.where(y == "nothing")[0]
if len(idx_nothing) > 0:
    sample_n = X[idx_nothing[0]]
    print(f"Class: {'nothing':<10} | Total Features: {len(sample_n)} | Non-Zero Features: {np.count_nonzero(sample_n)}")