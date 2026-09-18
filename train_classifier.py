"""
MULTI-SUBJECT PUBLIC & LOCAL EMOTION TRAINER
============================================
Trains the MLP classifier on merged public + local 1024D MobileNet embeddings.
Applies embedding noise augmentation and L2 regularization to generalize across all faces.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load unified dataset (Public + Local samples)
data = np.load("data/emotion_embeddings.npz")
X = data["X"]
y = data["y"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
classes = encoder.classes_

Path("models").mkdir(exist_ok=True)
np.save("models/classes.npy", classes)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

# Batch size increased to 64 for large dataset stability
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# 2. MLP Architecture
class RobustEmotionClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=len(classes)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobustEmotionClassifier(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

print(f"[INFO] Training MLP on {len(X_train)} samples across classes: {list(classes)}...")

# 3. Training Loop with Feature Augmentation
model.train()
epochs = 80
for epoch in range(epochs):
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Add slight embedding noise for generalization
        noise = torch.randn_like(batch_X) * 0.015
        augmented_X = batch_X + noise

        outputs = model(augmented_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}")

# 4. Evaluation
model.eval()
with torch.no_grad():
    val_outputs = model(X_val_t.to(device))
    preds = torch.argmax(val_outputs, dim=1)
    acc = (preds == y_val_t.to(device)).float().mean().item()

print(f"\n[SUCCESS] Final Validation Accuracy: {acc * 100:.2f}%")

# Save model weights
torch.save(model.state_dict(), "models/emotion_mlp.pth")
print("[INFO] Model weights saved to 'models/emotion_mlp.pth'")