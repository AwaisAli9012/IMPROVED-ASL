"""
MOBILENETV3 FEATURE EXTRACTOR
=============================
Passes normalized 224x224 face crops through a pretrained MobileNetV3 backbone
to generate 512-dimensional feature embedding vectors.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

class VisionFeatureExtractor:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load lightweight MobileNetV3 Small
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights)
        
        # Remove final classification layer to get raw feature vectors
        self.model.classifier = torch.nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, face_crop):
        """
        Takes a BGR 224x224 face crop, returns a 1D feature vector (numpy array).
        """
        if face_crop is None:
            return None

        # Convert BGR (OpenCV) to RGB
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor_crop = self.transform(rgb_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor_crop)
            # Flatten to 1D vector
            features = features.squeeze().cpu().numpy()

        return features