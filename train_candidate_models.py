import json
import os
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# -------------------------------------------------------------------
# 0. High-Performance Feature Extraction Engine
# -------------------------------------------------------------------
def extract_temporal_features(sequence: np.ndarray) -> np.ndarray:
    """
    Computes positional statistics (mean, std, min, max) and kinematics
    (velocity & acceleration) across a keypoint sequence (T, 126).
    """
    seq = np.atleast_2d(sequence)
    
    # Positional Aggregates
    mean_feat = np.mean(seq, axis=0)
    std_feat = np.std(seq, axis=0)
    min_feat = np.min(seq, axis=0)
    max_feat = np.max(seq, axis=0)
    
    # Kinematic Dynamics (Velocity & Acceleration)
    if seq.shape[0] > 1:
        vel = np.diff(seq, axis=0)
        mean_vel = np.mean(vel, axis=0)
        max_vel = np.max(vel, axis=0)
        
        if seq.shape[0] > 2:
            accel = np.diff(vel, axis=0)
            mean_accel = np.mean(accel, axis=0)
        else:
            mean_accel = np.zeros_like(mean_feat)
    else:
        mean_vel = np.zeros_like(mean_feat)
        max_vel = np.zeros_like(mean_feat)
        mean_accel = np.zeros_like(mean_feat)
        
    return np.hstack([mean_feat, std_feat, min_feat, max_feat, mean_vel, max_vel, mean_accel])

# -------------------------------------------------------------------
# 1. Path Setup & Data Ingestion
# -------------------------------------------------------------------
RAW_DATA_DIR = Path("keypoints_extracted")
X_PATH = RAW_DATA_DIR / "keypoints_raw.npy"
Y_PATH = RAW_DATA_DIR / "labels_raw.npy"
MANIFEST_PATH = RAW_DATA_DIR / "keypoints_raw_manifest.csv"

SPLITS_DIR = Path("Models_candidate") / "splits"
OUTPUT_MODELS_DIR = Path("Models_candidate") / "models"
OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset arrays and split indices...")
X_raw = np.load(X_PATH, allow_pickle=True)
y_raw = np.load(Y_PATH, allow_pickle=True)
df_manifest = pd.read_csv(MANIFEST_PATH)

print("Extracting full temporal & kinematic feature matrix...")
X_temporal = np.array([extract_temporal_features(s) for s in X_raw], dtype=np.float32)

train_idx = np.load(SPLITS_DIR / "train_idx.npy")
val_idx = np.load(SPLITS_DIR / "val_idx.npy")
test_idx = np.load(SPLITS_DIR / "test_idx.npy")

EXCLUDED_IDX_PATH = SPLITS_DIR / "excluded_idx.npy"
excluded_idx = np.load(EXCLUDED_IDX_PATH) if EXCLUDED_IDX_PATH.exists() else np.array([], dtype=int)

if len(excluded_idx) > 0:
    ex_set = set(excluded_idx)
    train_idx = np.array([i for i in train_idx if i not in ex_set], dtype=int)
    val_idx = np.array([i for i in val_idx if i not in ex_set], dtype=int)
    test_idx = np.array([i for i in test_idx if i not in ex_set], dtype=int)

# -------------------------------------------------------------------
# 2. Class Mapping & Label Formatting
# -------------------------------------------------------------------
active_classes = sorted(list(set(y_raw[train_idx])))
class_to_idx = {int(c): i for i, c in enumerate(active_classes)}
idx_to_class = {i: int(c) for i, c in enumerate(active_classes)}

with open(OUTPUT_MODELS_DIR / "candidate_label_mapping.json", "w") as f:
    json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

y = np.array([class_to_idx[v] if v in class_to_idx else -1 for v in y_raw])

X_train_raw = X_temporal[train_idx]
X_val_raw = X_temporal[val_idx]
X_test_raw = X_temporal[test_idx]

y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

# Global Scaler Fit (For Production Inference Pipeline)
global_scaler = StandardScaler()
global_scaler.fit(X_train_raw)
joblib.dump(global_scaler, OUTPUT_MODELS_DIR / "candidate_scaler.pkl")

# -------------------------------------------------------------------
# 3. Group Manifest Extraction
# -------------------------------------------------------------------
def get_group_id(row):
    filename = os.path.basename(str(row["source_image"]))
    match = re.match(r"^(\d+)_", filename)
    return f"{row['class_name']}_{match.group(1)}" if match else f"{row['class_name']}_single_{row.name}"

df_manifest["group_id"] = df_manifest.apply(get_group_id, axis=1)
train_groups = df_manifest.iloc[train_idx]["group_id"].values

# -------------------------------------------------------------------
# 4. Out-Of-Fold Ensembling with Zero-Leakage CV
# -------------------------------------------------------------------
num_classes = len(active_classes)
n_tr, n_v, n_te = len(X_train_raw), len(X_val_raw), len(X_test_raw)

base_models = {
    "rf": RandomForestClassifier(n_estimators=200, max_depth=25, class_weight="balanced", random_state=42, n_jobs=-1),
    "et": ExtraTreesClassifier(n_estimators=200, max_depth=25, class_weight="balanced", random_state=42, n_jobs=-1),
    "xgb": XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.08, eval_metric="mlogloss", random_state=42, n_jobs=-1)
}

oof_preds = {m: np.zeros((n_tr, num_classes)) for m in base_models}
val_preds = {m: np.zeros((n_v, num_classes)) for m in base_models}
test_preds = {m: np.zeros((n_te, num_classes)) for m in base_models}

sgkf = StratifiedGroupKFold(n_splits=5)

for name, model in base_models.items():
    print(f"\n[Cross-Validating Base Model: {name.upper()}]")
    
    for fold, (trn_f, val_f) in enumerate(sgkf.split(X_train_raw, y_train, train_groups)):
        # 1. In-Fold Standardization to isolate CV fold statistics strictly
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X_train_raw[trn_f])
        X_va = fold_scaler.transform(X_train_raw[val_f])
        X_v_fold = fold_scaler.transform(X_val_raw)
        X_te_fold = fold_scaler.transform(X_test_raw)
        
        y_tr = y_train[trn_f]
        sw = compute_sample_weight("balanced", y_tr)
        
        # 2. Model Fitting
        model.fit(X_tr, y_tr, sample_weight=sw if name == "xgb" else None)
        
        # 3. Accumulate Fold Predictions
        oof_preds[name][val_f] = model.predict_proba(X_va)
        val_preds[name] += model.predict_proba(X_v_fold) / 5.0
        test_preds[name] += model.predict_proba(X_te_fold) / 5.0
        
    print(f"Refitting full production {name.upper()} model...")
    X_train_scaled = global_scaler.transform(X_train_raw)
    full_sw = compute_sample_weight("balanced", y_train)
    model.fit(X_train_scaled, y_train, sample_weight=full_sw if name == "xgb" else None)
    joblib.dump(model, OUTPUT_MODELS_DIR / f"candidate_{name}.pkl")

# Meta-learner Stacking
X_meta_train = np.hstack([oof_preds[m] for m in base_models])
X_meta_val = np.hstack([val_preds[m] for m in base_models])
X_meta_test = np.hstack([test_preds[m] for m in base_models])

meta_sw = compute_sample_weight("balanced", y_train)
meta_learner = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=42)
meta_learner.fit(X_meta_train, y_train, sample_weight=meta_sw)

joblib.dump(meta_learner, OUTPUT_MODELS_DIR / "candidate_meta.pkl")

# -------------------------------------------------------------------
# 5. Final Evaluation & Classification Metrics
# -------------------------------------------------------------------
def report(title, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[{title}] Accuracy: {acc:.2f}% | Macro F1: {macro_f1:.4f}")

print("\n" + "="*50)
report("VAL (Ensemble Meta-Learner)", y_val, meta_learner.predict(X_meta_val))
report("TEST (Ensemble Meta-Learner)", y_test, meta_learner.predict(X_meta_test))
print("="*50)