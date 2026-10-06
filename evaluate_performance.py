import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from pathlib import Path

# Config
MODELS_DIR = Path("Models_candidate") / "models"
RAW_DATA_DIR = Path("keypoints_extracted")
X_PATH = RAW_DATA_DIR / "keypoints_raw.npy"
Y_PATH = RAW_DATA_DIR / "labels_raw.npy"
SPLITS_DIR = Path("Models_candidate") / "splits"
TEST_IDX_PATH = SPLITS_DIR / "test_idx.npy"

def load_data():
    print("Loading data...")
    X_raw = np.load(X_PATH)
    y_raw = np.load(Y_PATH)
    test_idx = np.load(TEST_IDX_PATH)
    
    with open(MODELS_DIR / "candidate_label_mapping.json", "r") as f:
        mapping = json.load(f)
    
    class_to_idx = mapping["class_to_idx"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    
    # Remap test labels
    y_test = np.array([class_to_idx[str(val)] if str(val) in class_to_idx else -1 for val in y_raw[test_idx]])
    X_test = X_raw[test_idx]
    
    return X_test, y_test, idx_to_class

def evaluate_ensemble():
    X_test, y_test, idx_to_class = load_data()
    
    print("Loading models...")
    rf = joblib.load(MODELS_DIR / "candidate_rf.pkl")
    et = joblib.load(MODELS_DIR / "candidate_et.pkl")
    xgb = joblib.load(MODELS_DIR / "candidate_xgb.pkl")
    meta = joblib.load(MODELS_DIR / "candidate_meta.pkl")
    
    print("Running inference...")
    rf_probs = rf.predict_proba(X_test)
    et_probs = et.predict_proba(X_test)
    xgb_probs = xgb.predict_proba(X_test)
    
    X_meta = np.hstack([rf_probs, et_probs, xgb_probs])
    y_pred = meta.predict(X_meta)
    
    # Generate labels for display
    unique_indices = sorted(np.unique(y_test))
    target_names = [idx_to_class[i] for i in unique_indices]
    # In Config.py, ALPHABET_CLASSES are letters, SIGN_CLASSES are words
    # We'll tag them for better analysis
    
    from Config import ALPHABET_CLASSES, SIGN_CLASSES
    alphabet_list = list(ALPHABET_CLASSES.values())
    
    def get_category(name):
        return "ALPHABET" if name in alphabet_list else "WORD"

    report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True)).transpose()
    report_df['category'] = report_df.index.map(get_category)
    
    print("\n--- PER-CLASS PERFORMANCE ---")
    print(report_df.sort_values(by="f1-score", ascending=False))
    
    # Summary by Category
    print("\n--- CATEGORY SUMMARY ---")
    cat_summary = report_df.groupby('category')[['precision', 'recall', 'f1-score']].mean()
    print(cat_summary)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(20, 15))
    # sns.heatmap(cm, annot=False, fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Combined Vocabulary')
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved to confusion_matrix.png")

    # Interference Analysis
    print("\n--- TOP INTERFERENCES ---")
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    interferences = []
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            if i != j and cm[i, j] > 5: # Threshold for significant errors
                interferences.append({
                    'true': target_names[i],
                    'pred': target_names[j],
                    'count': cm[i, j],
                    'true_cat': get_category(target_names[i]),
                    'pred_cat': get_category(target_names[j])
                })
    
    int_df = pd.DataFrame(interferences).sort_values(by='count', ascending=False)
    print(int_df.head(20))

if __name__ == "__main__":
    evaluate_ensemble()
