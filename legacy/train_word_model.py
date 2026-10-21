import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "keypoints_extracted"
SPLITS_DIR = BASE_DIR / "Models_candidate" / "splits"
MODELS_CANDIDATE_DIR = BASE_DIR / "Models_candidate" / "word_model"
WORD_LABEL_IDS = set(range(36))


def parse_source_image_info(source_image_path: str) -> tuple[str, str, int]:
    """Return filename, source prefix, and numeric frame suffix."""
    path_obj = Path(source_image_path)
    filename = path_obj.name
    match = re.search(r"_(\d+)\.(?:jpg|png|jpeg|npy)$", filename, re.IGNORECASE)
    if match:
        return filename, filename[:match.start()], int(match.group(1))

    numbers = re.findall(r"\d+", filename)
    return filename, path_obj.stem, int(numbers[-1]) if numbers else 0


def run_preflight_checks(
    features: np.ndarray,
    labels: np.ndarray,
    manifest: pd.DataFrame,
    partitions: dict[str, np.ndarray],
) -> None:
    print("Running word-model data and split integrity checks...")
    sample_count = len(features)
    if not (sample_count == len(labels) == len(manifest)):
        raise ValueError(
            f"Row count mismatch: features={sample_count}, labels={len(labels)}, "
            f"manifest={len(manifest)}"
        )
    if features.ndim != 2 or features.shape[1] != 126:
        raise ValueError(f"Expected features shaped (N, 126), got {features.shape}")
    if not np.array_equal(labels, manifest["label_id"].to_numpy()):
        raise ValueError("labels_raw.npy is not aligned with manifest label_id rows")

    for name, indices in partitions.items():
        if indices.ndim != 1:
            raise ValueError(f"Split '{name}' must be a one-dimensional index array")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Split '{name}' contains duplicate sample indices")
        if len(indices) and (indices.min() < 0 or indices.max() >= sample_count):
            raise ValueError(f"Split '{name}' contains out-of-range sample indices")

    split_names = list(partitions)
    for position, first_name in enumerate(split_names):
        first = set(map(int, partitions[first_name]))
        for second_name in split_names[position + 1:]:
            overlap = first.intersection(map(int, partitions[second_name]))
            if overlap:
                raise ValueError(
                    f"Sample-index overlap between '{first_name}' and '{second_name}': "
                    f"{len(overlap)} indices"
                )

    combined = np.concatenate(list(partitions.values()))
    if not np.array_equal(np.sort(combined), np.arange(sample_count)):
        missing = set(range(sample_count)) - set(map(int, combined))
        extra = set(map(int, combined)) - set(range(sample_count))
        raise ValueError(
            f"Splits must cover all samples exactly once; missing={len(missing)}, "
            f"extra={len(extra)}"
        )

    word_manifest = manifest[manifest["label_id"].isin(WORD_LABEL_IDS)]
    inconsistent = word_manifest.groupby(["class_name", "prefix"])["label_id"].nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if not inconsistent.empty:
        raise ValueError("A (class_name, prefix) sequence has multiple label IDs")

    word_group_splits: dict[tuple[str, str], str] = {}
    for split_name, indices in partitions.items():
        part = manifest.iloc[indices]
        part = part[part["label_id"].isin(WORD_LABEL_IDS)]
        group_keys = set(zip(part["class_name"], part["prefix"]))
        for group_key in group_keys:
            prior_split = word_group_splits.setdefault(group_key, split_name)
            if prior_split != split_name:
                raise ValueError(
                    f"Word sequence group {group_key} crosses '{prior_split}' and '{split_name}'"
                )

    print(
        f"Integrity checks passed: {sample_count:,} samples, "
        f"{len(word_group_splits):,} word sequence groups, no split/group overlap."
    )


def build_sequence_features_for_partition(
    features: np.ndarray,
    manifest: pd.DataFrame,
    indices: np.ndarray,
    partition_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    partition = manifest.iloc[indices].copy()
    partition["original_index"] = indices
    partition = partition[partition["label_id"].isin(WORD_LABEL_IDS)].copy()
    if partition.empty:
        raise ValueError(f"No word sequences found in {partition_name} partition")

    sequence_features = []
    sequence_labels = []
    sequence_keys = []
    for (class_name, prefix), sequence in partition.groupby(["class_name", "prefix"]):
        ordered = sequence.sort_values(["frame_num", "original_index"])
        frame_indices = ordered["original_index"].to_numpy(dtype=np.int64)
        frame_numbers = ordered["frame_num"].to_numpy(dtype=np.float32)
        clip = features[frame_indices]
        mean = np.mean(clip, axis=0)
        std = np.std(clip, axis=0)
        minimum = np.min(clip, axis=0)
        maximum = np.max(clip, axis=0)

        if len(clip) > 1:
            frame_gaps = np.diff(frame_numbers)
            frame_gaps = np.where(frame_gaps <= 0, 1.0, frame_gaps)[:, None]
            velocities = np.diff(clip, axis=0) / frame_gaps
            mean_velocity = np.mean(np.abs(velocities), axis=0)
            max_velocity = np.max(np.abs(velocities), axis=0)
            displacement = clip[-1] - clip[0]
            duration = frame_numbers[-1] - frame_numbers[0]
        else:
            mean_velocity = np.zeros_like(mean)
            max_velocity = np.zeros_like(mean)
            displacement = np.zeros_like(mean)
            duration = 1.0

        sequence_features.append(
            np.concatenate(
                [
                    mean,
                    std,
                    minimum,
                    maximum,
                    mean_velocity,
                    max_velocity,
                    displacement,
                    np.asarray([len(clip), duration], dtype=np.float32),
                ]
            )
        )
        sequence_labels.append(int(ordered["label_id"].iloc[0]))
        sequence_keys.append(f"{class_name}::{prefix}")

    return (
        np.asarray(sequence_features, dtype=np.float32),
        np.asarray(sequence_labels, dtype=np.int64),
        sequence_keys,
    )


def check_candidate_output_guard(output_dir: Path) -> None:
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing:
            print(f"Candidate output directory already contains {len(existing)} item(s):")
            for path in existing:
                print(f"  - {path.name}")
            response = input("Overwrite these candidate artifacts? (y/N): ").strip().lower()
            if response != "y":
                print("Aborted without training or overwriting candidate artifacts.")
                sys.exit(0)
    output_dir.mkdir(parents=True, exist_ok=True)


def run_word_model_training() -> None:
    print("--- Word Sequence Candidate Model Pipeline ---")
    data_dir = DATA_DIR
    split_dir = SPLITS_DIR
    required_paths = [
        data_dir / "keypoints_raw.npy",
        data_dir / "labels_raw.npy",
        data_dir / "keypoints_raw_manifest.csv",
        split_dir / "train_idx.npy",
        split_dir / "val_idx.npy",
        split_dir / "test_idx.npy",
        split_dir / "excluded_idx.npy",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input files missing: " + ", ".join(map(str, missing)))

    features = np.load(data_dir / "keypoints_raw.npy", mmap_mode="r")
    labels = np.load(data_dir / "labels_raw.npy", mmap_mode="r")
    manifest = pd.read_csv(data_dir / "keypoints_raw_manifest.csv")
    train_indices = np.load(split_dir / "train_idx.npy")
    validation_indices = np.load(split_dir / "val_idx.npy")
    test_indices = np.load(split_dir / "test_idx.npy")
    excluded_indices = np.load(split_dir / "excluded_idx.npy")

    parsed = manifest["source_image"].apply(parse_source_image_info)
    manifest["filename"] = [item[0] for item in parsed]
    manifest["prefix"] = [item[1] for item in parsed]
    manifest["frame_num"] = [item[2] for item in parsed]

    partitions = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
        "excluded": excluded_indices,
    }
    run_preflight_checks(features, labels, manifest, partitions)

    check_candidate_output_guard(MODELS_CANDIDATE_DIR)

    print("Building sequence features independently for each partition...")
    X_train_seq, y_train_raw, _ = build_sequence_features_for_partition(
        features, manifest, train_indices, "train"
    )
    X_val_seq, y_val_raw, _ = build_sequence_features_for_partition(
        features, manifest, validation_indices, "validation"
    )
    X_test_seq, y_test_raw, _ = build_sequence_features_for_partition(
        features, manifest, test_indices, "test"
    )

    train_classes = sorted(np.unique(y_train_raw).tolist())
    label_to_index = {label: index for index, label in enumerate(train_classes)}
    unseen_validation = set(np.unique(y_val_raw)) - set(train_classes)
    unseen_test = set(np.unique(y_test_raw)) - set(train_classes)
    if unseen_validation or unseen_test:
        raise ValueError(
            f"Evaluation sequences contain unseen train labels: "
            f"validation={unseen_validation}, test={unseen_test}"
        )

    id_to_name = (
        manifest[["label_id", "class_name"]]
        .drop_duplicates()
        .set_index("label_id")["class_name"]
        .to_dict()
    )
    index_to_name = {index: id_to_name[label] for index, label in enumerate(train_classes)}
    mapping_path = MODELS_CANDIDATE_DIR / "word_label_mapping.json"
    mapping_path.write_text(json.dumps(index_to_name, indent=2), encoding="utf-8")

    y_train = np.asarray([label_to_index[int(label)] for label in y_train_raw])
    y_test = np.asarray([label_to_index[int(label)] for label in y_test_raw])
    report_labels = list(range(len(train_classes)))
    report_names = [index_to_name[index] for index in report_labels]

    print(
        f"Sequences: train={len(X_train_seq)}, validation={len(X_val_seq)}, "
        f"test={len(X_test_seq)}; classes={len(train_classes)}"
    )
    print("Validation is used only for label coverage, not tuning; test is held out.")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=18, class_weight="balanced", random_state=42, n_jobs=-1
    )
    extra_trees = ExtraTreesClassifier(
        n_estimators=300, max_depth=18, class_weight="balanced", random_state=42, n_jobs=-1
    )
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    print("Training candidate word-sequence models...")
    rf.fit(X_train_seq, y_train)
    extra_trees.fit(X_train_seq, y_train)
    xgb_model.fit(X_train_seq, y_train)

    probabilities = (
        rf.predict_proba(X_test_seq)
        + extra_trees.predict_proba(X_test_seq)
        + xgb_model.predict_proba(X_test_seq)
    ) / 3.0
    predictions = np.argmax(probabilities, axis=1)

    print("\nWORD-ONLY SEQUENCE MODEL: RESERVED TEST SET")
    print(f"Accuracy: {np.mean(predictions == y_test) * 100:.2f}%")
    print(
        "Macro F1: "
        f"{f1_score(y_test, predictions, labels=report_labels, average='macro', zero_division=0):.4f}"
    )
    print(
        classification_report(
            y_test,
            predictions,
            labels=report_labels,
            target_names=report_names,
            digits=4,
            zero_division=0,
        )
    )

    joblib.dump(rf, MODELS_CANDIDATE_DIR / "candidate_word_rf.pkl")
    joblib.dump(extra_trees, MODELS_CANDIDATE_DIR / "candidate_word_et.pkl")
    joblib.dump(xgb_model, MODELS_CANDIDATE_DIR / "candidate_word_xgb.pkl")
    print(f"Candidate artifacts saved under {MODELS_CANDIDATE_DIR}; production models untouched.")


if __name__ == "__main__":
    run_word_model_training()
