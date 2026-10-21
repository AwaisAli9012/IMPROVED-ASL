"""
MULTI-MODAL FEATURE EXTRACTION (V4 - RESOLUTION-INDEPENDENT + GROUPED)
========================================================================
1. Landmarks are scaled to pixel space (x*w, y*h, z*w) before normalizing, so
   features do not depend on image aspect ratio (square Kaggle photos vs 640x480 webcam).
2. Wrist is the origin; scale = distance wrist -> middle MCP (landmark 9).
3. 126 features: 63 Left + 63 Right. Frames with no detected hand are dropped.
4. Saves groups.npy so train.py can split by clip / photo block (no leakage).

IMPORTANT: extract_frame_features() must be IDENTICAL in app.py.
"""

import re
import cv2
import numpy as np
import mediapipe.python.solutions.hands as mp_hands
from pathlib import Path

# -------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "Dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "Dataset_Multimodal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only scan the two real datasets (skips emotion folders and asl_alphabet_test)
SCAN_DIRS = [
    RAW_DATA_DIR / "frames_20",
    RAW_DATA_DIR / "archive" / "asl_alphabet_train" / "asl_alphabet_train",
]

EXCLUDED_CLASSES = {
    "man", "cool", "walk", "woman", "son", "family",
    "del", "space", "delete", "nothing",
}


# -------------------------------------------------------------------------
# FEATURES (copy this function verbatim into app.py)
# -------------------------------------------------------------------------
def extract_frame_features(results, img_w, img_h):
    if not results or not results.multi_hand_landmarks:
        return np.zeros(126, dtype=np.float32)

    left_hand_coords = np.zeros((21, 3), dtype=np.float32)
    right_hand_coords = np.zeros((21, 3), dtype=np.float32)
    left_present, right_present = False, False
    scale_factors = np.array([img_w, img_h, img_w], dtype=np.float32)

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness_label = "Right"
        if results.multi_handedness and idx < len(results.multi_handedness):
            handedness_label = results.multi_handedness[idx].classification[0].label

        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
        ) * scale_factors

        if handedness_label == "Left":
            left_hand_coords = coords
            left_present = True
        else:
            right_hand_coords = coords
            right_present = True

    def process_geometry(raw_coords, is_present):
        if not is_present:
            return np.zeros(63, dtype=np.float32)
        rel = raw_coords - raw_coords[0]
        span = float(np.linalg.norm(rel[9]))
        if span < 1e-6:
            span = 1e-6
        return (rel / span).flatten()

    left_norm = process_geometry(left_hand_coords, left_present)
    right_norm = process_geometry(right_hand_coords, right_present)
    return np.concatenate([left_norm, right_norm]).astype(np.float32)


# -------------------------------------------------------------------------
# GROUP IDS (for leakage-free splitting)
# -------------------------------------------------------------------------
def make_group_id(img_path: Path, class_name: str) -> str:
    stem = img_path.stem
    if "frames_20" in img_path.parts:
        # word frames: "05727_12" -> clip "05727"
        clip = stem.split("_")[0]
        return f"{class_name}/{clip}"
    # alphabet photos: "A123" -> block of 100 neighbouring files
    m = re.search(r"(\d+)", stem)
    num = int(m.group(1)) if m else 0
    return f"{class_name}_{num // 100}"


# -------------------------------------------------------------------------
# BATCH EXTRACTION
# -------------------------------------------------------------------------
def process_dataset():
    print(f"Extracting from: {[str(d) for d in SCAN_DIRS]}")
    print(f"Saving to: {OUTPUT_DIR}\n")

    missing = [d for d in SCAN_DIRS if not d.exists()]
    if missing:
        print(f"Missing scan directories: {missing}")
        return

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.3,
    )

    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    all_paths = sorted({p for d in SCAN_DIRS for e in exts for p in d.rglob(e)})
    print(f"Discovered {len(all_paths):,} image files.")

    X_data, y_labels, group_labels = [], [], []
    skipped = no_hand = unreadable = 0
    sizes_seen = {}

    for idx, img_path in enumerate(all_paths):
        class_name = img_path.parent.name
        if class_name.lower() in EXCLUDED_CLASSES:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            unreadable += 1
            continue

        img_h, img_w = img.shape[:2]
        sizes_seen[(img_w, img_h)] = sizes_seen.get((img_w, img_h), 0) + 1

        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        vector = extract_frame_features(results, img_w, img_h)
        if np.all(vector == 0):
            no_hand += 1
            continue

        X_data.append(vector)
        y_labels.append(class_name)
        group_labels.append(make_group_id(img_path, class_name))

        if (idx + 1) % 5000 == 0:
            print(f"Processed [{idx + 1:,}/{len(all_paths):,}]...")

    if not X_data:
        print("No valid samples found.")
        return

    X_arr = np.array(X_data, dtype=np.float32)
    y_arr = np.array(y_labels)
    g_arr = np.array(group_labels)

    np.save(OUTPUT_DIR / "X_multimodal.npy", X_arr)
    np.save(OUTPUT_DIR / "y_multimodal.npy", y_arr)
    np.save(OUTPUT_DIR / "groups.npy", g_arr)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"Samples:            {X_arr.shape[0]:,}  ({X_arr.shape[1]} features)")
    print(f"Excluded skipped:   {skipped:,}")
    print(f"Unreadable images:  {unreadable:,}")
    print(f"No-hand dropped:    {no_hand:,}")
    print(f"Unique classes:     {len(np.unique(y_arr))}")
    print(f"Unique groups:      {len(np.unique(g_arr))}")
    print(f"Image sizes seen:   {dict(sorted(sizes_seen.items(), key=lambda kv: -kv[1])[:5])}")
    print("=" * 60)

    # per-class group counts: grouped 5-fold CV needs >= 5 groups per class
    print("\nClasses with fewer than 5 groups:")
    low = False
    for c in np.unique(y_arr):
        n = len(np.unique(g_arr[y_arr == c]))
        if n < 5:
            print(f"  {c}: {n}")
            low = True
    if not low:
        print("  none")


if __name__ == "__main__":
    process_dataset()