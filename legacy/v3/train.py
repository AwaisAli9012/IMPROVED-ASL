import json
import sys
from pathlib import Path
import joblib
import numpy as np
import sklearn
from packaging import version

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
try:
    import Config
except ImportError:
    import config as Config

DATASET_DIR = Path(__file__).resolve().parent / "Dataset_Multimodal"
OUTPUT_DIR = Path(__file__).resolve().parent / "Models_candidate" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ASL MULTIMODAL WIZARD - STACKED ENSEMBLE (GROUPED SPLIT)")
print("=" * 70)

X_raw = np.load(DATASET_DIR / "X_multimodal.npy", allow_pickle=True).astype(np.float32)
y_raw = np.load(DATASET_DIR / "y_multimodal.npy", allow_pickle=True)
g_raw = np.load(DATASET_DIR / "groups.npy", allow_pickle=True)
assert len(X_raw) == len(y_raw) == len(g_raw), "X, y and groups lengths differ - re-run extraction"

invalid_classes = {'asl_alphabet_test', 'angry', 'disgust', 'fear', 'happy',
                   'neutral', 'sad', 'surprise', 'nothing'}
valid_mask = ~np.isin(y_raw, list(invalid_classes))
X_clean, y_clean, g_clean = X_raw[valid_mask], y_raw[valid_mask], g_raw[valid_mask]
assert X_clean.shape[1] == 126, f"Expected 126 features, got {X_clean.shape[1]}"

config_classes = list(Config.ALL_SIGN_NAMES) + list(Config.ALL_ALPHABET_NAMES)
in_data = set(np.unique(y_clean))
print("Missing from data:", set(config_classes) - in_data)
print("Not in config:    ", in_data - set(config_classes))

active_classes = [c for c in config_classes if c in in_data]
class_to_idx = {c: i for i, c in enumerate(active_classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

# keep only config classes, aligned across X, y, groups
keep = np.isin(y_clean, active_classes)
X_clean, y_clean, g_clean = X_clean[keep], y_clean[keep], g_clean[keep]
y_encoded = np.array([class_to_idx[c] for c in y_clean])

with open(OUTPUT_DIR / "candidate_label_mapping.json", "w") as f:
    json.dump({"class_to_idx": class_to_idx,
               "idx_to_class": {str(k): v for k, v in idx_to_class.items()}}, f, indent=2)
print(f"Mapped {len(active_classes)} classes")

# every class needs >= 5 groups for 5-fold grouped CV
for c in active_classes:
    n_groups = len(np.unique(g_clean[y_clean == c]))
    if n_groups < 5:
        print(f"WARNING: class '{c}' has only {n_groups} groups - lower n_splits or add data")

# grouped, stratified test split: first of 5 folds (~20%)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr, te = next(sgkf.split(X_clean, y_encoded, g_clean))
X_train_raw, y_train_raw, g_train_raw = X_clean[tr], y_encoded[tr], g_clean[tr]
X_test_raw, y_test = X_clean[te], y_encoded[te]
print(f"Train {len(tr):,} | Test {len(te):,} | test classes {len(np.unique(y_test))}/{len(active_classes)}")

# cap majority classes (keeping groups aligned) - reduced to 400 to balance words vs letters
MAX_SAMPLES = 400
rng = np.random.default_rng(42)
Xs, ys, gs = [], [], []
for c in np.unique(y_train_raw):
    idx = np.where(y_train_raw == c)[0]
    if len(idx) > MAX_SAMPLES:
        idx = rng.choice(idx, size=MAX_SAMPLES, replace=False)
    Xs.append(X_train_raw[idx]); ys.append(y_train_raw[idx]); gs.append(g_train_raw[idx])
X_train_capped = np.vstack(Xs)
y_train = np.concatenate(ys)
g_train = np.concatenate(gs)

# Guard against missing classes in training split
missing = set(range(len(active_classes))) - set(np.unique(y_train))
assert not missing, f"Classes absent from training split: {[idx_to_class[i] for i in missing]}"

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_capped)
X_test = scaler.transform(X_test_raw)
joblib.dump(scaler, OUTPUT_DIR / "candidate_scaler.pkl")

base_models = {
    "rf": RandomForestClassifier(n_estimators=200, max_depth=25, class_weight="balanced", random_state=42, n_jobs=1),
    "et": ExtraTreesClassifier(n_estimators=200, max_depth=25, class_weight="balanced", random_state=42, n_jobs=1),
    "xgb": XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.08, eval_metric="mlogloss", random_state=42, n_jobs=1),
}

# Inner CV set to 3 folds to prevent fragile splits on rare word classes
cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
sw_train = compute_sample_weight("balanced", y_train)
param_key = "params" if version.parse(sklearn.__version__) >= version.parse("1.4") else "fit_params"

oof_train_preds, test_preds = [], []
for name, model in base_models.items():
    print(f"OOF + training [{name.upper()}]...")
    kw = {param_key: {"sample_weight": sw_train}} if name == "xgb" else {}
    oof = cross_val_predict(model, X_train, y_train, groups=g_train, cv=cv,
                            method="predict_proba", n_jobs=-1, **kw)
    oof_train_preds.append(oof)
    model.fit(X_train, y_train, sample_weight=sw_train if name == "xgb" else None)
    joblib.dump(model, OUTPUT_DIR / f"candidate_{name}.pkl")
    test_preds.append(model.predict_proba(X_test))

meta = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=42)
meta.fit(np.hstack(oof_train_preds), y_train)
joblib.dump(meta, OUTPUT_DIR / "candidate_meta.pkl")

y_pred = meta.predict(np.hstack(test_preds))
print("\n" + "=" * 60)
print(f"Grouped test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Macro F1:              {f1_score(y_test, y_pred, average='macro'):.4f}")
print("=" * 60)

# Separate Word vs. Letter Accuracy Reporting
word_idx = {i for i, c in idx_to_class.items() if c in set(Config.ALL_SIGN_NAMES)}
is_word = np.isin(y_test, list(word_idx))
for name, m in [("WORDS", is_word), ("LETTERS", ~is_word)]:
    if m.any():
        print(f"{name:8s} accuracy: {accuracy_score(y_test[m], y_pred[m])*100:.2f}%  (n={m.sum()})")

labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
print(classification_report(y_test, y_pred, labels=labels,
                            target_names=[idx_to_class[i] for i in labels], zero_division=0))