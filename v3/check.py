import numpy as np
import pandas as pd

# Load extracted label array
y = np.load("Dataset_Multimodal/y_multimodal.npy")

# Compute sample counts for every class
unique_classes, counts = np.unique(y, return_counts=True)
df_counts = pd.DataFrame({
    'Class': unique_classes, 
    'Samples': counts,
    'Percentage (%)': np.round((counts / len(y)) * 100, 2)
}).sort_values(by='Samples', ascending=False).reset_index(drop=True)

# Display all rows without truncating
pd.set_option('display.max_rows', None)

print("=" * 60)
print(f"TOTAL SAMPLES: {len(y):,}")
print(f"TOTAL CLASSES: {len(unique_classes)}")
print("=" * 60 + "\n")

print(df_counts.to_string(index=True))

# Save full breakdown to CSV for reference
df_counts.to_csv("Dataset_Multimodal/class_distribution.csv", index=False)
print("\n" + "=" * 60)
print("✓ Full class distribution saved to 'Dataset_Multimodal/class_distribution.csv'")
print("=" * 60)