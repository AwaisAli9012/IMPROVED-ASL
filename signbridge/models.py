"""Neural models and (later) the predictors that wrap trained artifacts."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ARTIFACTS, T_MODEL, TORCH_THREADS, WORD_FEATURES


class ConvBlock(nn.Module):
    """Residual depthwise-separable temporal convolution over (B, T, d)."""

    def __init__(self, d, k=5, p=0.1):
        super().__init__()
        self.dw, self.bn1 = nn.Conv1d(d, d, k, padding=k // 2, groups=d), nn.BatchNorm1d(d)
        self.pw, self.bn2 = nn.Conv1d(d, d, 1), nn.BatchNorm1d(d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        y = self.bn2(self.pw(F.silu(self.bn1(self.dw(x.transpose(1, 2))))))
        return x + self.drop(y).transpose(1, 2)


class WordNet(nn.Module):
    """(B, T_MODEL, WORD_FEATURES) -> (B, n_classes) logits: local conv blocks, then a small transformer."""

    def __init__(self, n_classes, d=192, layers=2, heads=4, conv_blocks=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(WORD_FEATURES, d), nn.LayerNorm(d), nn.SiLU())
        self.convs = nn.Sequential(*[ConvBlock(d) for _ in range(conv_blocks)])
        self.pos = nn.Parameter(torch.zeros(1, T_MODEL, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, n_classes))

    def forward(self, x):
        x = self.encoder(self.convs(self.inp(x)) + self.pos)
        return self.head(torch.cat([x.mean(1), x.amax(1)], dim=1))


class WordPredictor:
    """Loads artifacts/words.pt (one or more seeds, softmax averaged)."""

    def __init__(self, path=None):
        ck = torch.load(path or ARTIFACTS / "words.pt", map_location="cpu", weights_only=True)
        self.labels, cfg = ck["labels"], ck["config"]
        self.models = []
        for state in ck["state_dicts"]:
            m = WordNet(len(self.labels), d=cfg["d_model"], layers=cfg["layers"])
            m.load_state_dict(state)
            self.models.append(m.eval())

    def proba(self, feats):
        """(B, T_MODEL, WORD_FEATURES) float32 -> (B, n_classes)."""
        torch.set_num_threads(TORCH_THREADS)  # the setting is per thread, and FastAPI runs requests in worker threads
        with torch.inference_mode():
            x = torch.from_numpy(feats)
            return torch.stack([F.softmax(m(x), dim=1) for m in self.models]).mean(0).numpy()


class SoftVote:
    """Averages predict_proba of already-fitted scikit-learn style models."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)
