"""
IMPROVED ASL - Train Models for All Groups
===========================================
Trains RF + XGBoost + Meta-Learner for each of 13 groups
5-fold cross-validation
Saves trained models to Models/ directory
Resume capability: skips already trained groups
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from Config import (
    KEYPOINTS_DIR, MODELS_DIR, GROUPS, TRAINING_CONFIG,
    SIGN_CLASSES, ALPHABET_CLASSES
)

# Create Models directory
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("IMPROVED ASL - TRAIN MODELS FOR ALL GROUPS")
print("=" * 70)
print(f"\nLoading keypoints...")

# Load all data once
keypoints = np.load(KEYPOINTS_DIR / 'keypoints.npy')
labels = np.load(KEYPOINTS_DIR / 'labels.npy')

print(f"✓ Loaded {keypoints.shape[0]} samples, {keypoints.shape[1]} dimensions")

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_class_id_from_name(class_name):
    """Get label ID from class name"""
    # Check signs first
    for idx, name in SIGN_CLASSES.items():
        if name == class_name:
            return idx
    # Check alphabets
    for idx, name in ALPHABET_CLASSES.items():
        if name == class_name:
            return 36 + idx  # Alphabets start from 36
    return None

def extract_group_data(keypoints_all, labels_all, group_classes):
    """Extract keypoints and labels for a specific group"""
    group_label_ids = []
    for class_name in group_classes:
        class_id = get_class_id_from_name(class_name)
        if class_id is not None:
            group_label_ids.append(class_id)
    
    # Filter keypoints and labels for this group
    mask = np.isin(labels_all, group_label_ids)
    group_keypoints = keypoints_all[mask]
    group_labels = labels_all[mask]
    
    # Remap labels to 0, 1, 2, ... (for this group)
    le = LabelEncoder()
    group_labels_remapped = le.fit_transform(group_labels)
    
    return group_keypoints, group_labels_remapped

def train_rf(X_train, y_train):
    """Train Random Forest"""
    rf = RandomForestClassifier(
        n_estimators=TRAINING_CONFIG['random_forest']['n_estimators'],
        max_depth=TRAINING_CONFIG['random_forest']['max_depth'],
        random_state=TRAINING_CONFIG['random_forest']['random_state'],
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf

def train_xgb(X_train, y_train):
    """Train XGBoost"""
    xgb_model = xgb.XGBClassifier(
        n_estimators=TRAINING_CONFIG['xgboost']['n_estimators'],
        max_depth=TRAINING_CONFIG['xgboost']['max_depth'],
        learning_rate=TRAINING_CONFIG['xgboost']['learning_rate'],
        random_state=TRAINING_CONFIG['xgboost']['random_state'],
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_meta_learner(rf_model, xgb_model, X_train, y_train, X_val, y_val):
    """Train Meta-Learner on stacked predictions"""
    # Get predictions from RF and XGBoost on training data
    rf_probs_train = rf_model.predict_proba(X_train)
    xgb_probs_train = xgb_model.predict_proba(X_train)
    
    # Stack predictions
    meta_X_train = np.hstack([rf_probs_train, xgb_probs_train])
    
    # Train meta-learner
    meta = LogisticRegression(
        max_iter=TRAINING_CONFIG['meta_learner']['max_iter'],
        random_state=TRAINING_CONFIG['meta_learner']['random_state']
    )
    meta.fit(meta_X_train, y_train)
    
    return meta

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model accuracy"""
    accuracy = model.score(X_test, y_test)
    return accuracy

def save_models(group_id, rf_model, xgb_model, meta_model):
    """Save trained models"""
    rf_path = MODELS_DIR / f"{group_id}_rf.pkl"
    xgb_path = MODELS_DIR / f"{group_id}_xgb.pkl"
    meta_path = MODELS_DIR / f"{group_id}_meta.pkl"
    
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_model, f)
    
    return rf_path, xgb_path, meta_path

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TRAINING ALL GROUPS")
print("=" * 70)

trained_count = 0
skipped_count = 0

for group_id, group_info in GROUPS.items():
    # Check if already trained
    rf_file = MODELS_DIR / f"{group_id}_rf.pkl"
    xgb_file = MODELS_DIR / f"{group_id}_xgb.pkl"
    meta_file = MODELS_DIR / f"{group_id}_meta.pkl"
    
    if rf_file.exists() and xgb_file.exists() and meta_file.exists():
        print(f"\n⏭️  {group_id}: Already trained! Skipping...")
        skipped_count += 1
        continue
    
    print(f"\n{'='*70}")
    print(f"🔄 Training {group_id}: {group_info['name']}")
    print(f"   Classes: {', '.join(group_info['classes'])}")
    print(f"{'='*70}")
    
    # Extract group data
    group_keypoints, group_labels = extract_group_data(
        keypoints, labels, group_info['classes']
    )
    
    print(f"Group data: {group_keypoints.shape[0]} samples, {len(np.unique(group_labels))} classes")
    
    # 5-fold cross-validation
    skf = StratifiedKFold(
        n_splits=TRAINING_CONFIG['cv_folds'],
        shuffle=True,
        random_state=42
    )
    
    fold_accuracies_rf = []
    fold_accuracies_xgb = []
    fold_accuracies_meta = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(group_keypoints, group_labels)):
        X_train, X_val = group_keypoints[train_idx], group_keypoints[val_idx]
        y_train, y_val = group_labels[train_idx], group_labels[val_idx]
        
        # Train models
        rf_model = train_rf(X_train, y_train)
        xgb_model = train_xgb(X_train, y_train)
        meta_model = train_meta_learner(rf_model, xgb_model, X_train, y_train, X_val, y_val)
        
        # Evaluate
        rf_acc = evaluate_model(rf_model, X_val, y_val, "RF")
        xgb_acc = evaluate_model(xgb_model, X_val, y_val, "XGBoost")
        
        # Meta accuracy
        rf_probs = rf_model.predict_proba(X_val)
        xgb_probs = xgb_model.predict_proba(X_val)
        meta_X_val = np.hstack([rf_probs, xgb_probs])
        meta_acc = evaluate_model(meta_model, meta_X_val, y_val, "Meta")
        
        fold_accuracies_rf.append(rf_acc)
        fold_accuracies_xgb.append(xgb_acc)
        fold_accuracies_meta.append(meta_acc)
        
        print(f"  Fold {fold_idx+1}: RF={rf_acc:.3f}, XGB={xgb_acc:.3f}, Meta={meta_acc:.3f}")
    
    # Final training on all data
    print(f"\nTraining final models on all data...")
    rf_final = train_rf(group_keypoints, group_labels)
    xgb_final = train_xgb(group_keypoints, group_labels)
    meta_final = train_meta_learner(
        rf_final, xgb_final, 
        group_keypoints, group_labels,
        group_keypoints, group_labels
    )
    
    # Save models
    rf_path, xgb_path, meta_path = save_models(group_id, rf_final, xgb_final, meta_final)
    
    # Print summary
    print(f"\n✅ {group_id} COMPLETE!")
    print(f"   RF CV Accuracy: {np.mean(fold_accuracies_rf):.3f} ± {np.std(fold_accuracies_rf):.3f}")
    print(f"   XGB CV Accuracy: {np.mean(fold_accuracies_xgb):.3f} ± {np.std(fold_accuracies_xgb):.3f}")
    print(f"   Meta CV Accuracy: {np.mean(fold_accuracies_meta):.3f} ± {np.std(fold_accuracies_meta):.3f}")
    print(f"   Saved: {rf_path.name}, {xgb_path.name}, {meta_path.name}")
    
    trained_count += 1

# Final summary
print("\n" + "=" * 70)
print("✅ ALL TRAINING COMPLETE!")
print("=" * 70)
print(f"\nSummary:")
print(f"  - Groups trained: {trained_count}")
print(f"  - Groups skipped (already trained): {skipped_count}")
print(f"  - Total groups: {len(GROUPS)}")
print(f"  - Models saved: {trained_count * 3} (3 per group)")
print(f"  - Location: {MODELS_DIR}")
print("\n" + "=" * 70)
