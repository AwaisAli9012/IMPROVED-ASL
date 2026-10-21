"""Create audited, source-aware train/validation/test splits for raw ASL keypoints."""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from Config import ALPHABET_CLASSES, KEYPOINTS_DIR, SIGN_CLASSES

NOTHING_LABEL_ID = 63
SIGN_LABEL_IDS = set(SIGN_CLASSES)
CLASS_NAMES = {
    **{class_id: name for class_id, name in SIGN_CLASSES.items()},
    **{36 + class_id: name for class_id, name in ALPHABET_CLASSES.items()},
}
SIGN_PREFIX_PATTERN = re.compile(r"^(\d+)_")


def source_group_id(label_id: int, source_image: str) -> str:
    filename = Path(source_image).name
    if label_id in SIGN_LABEL_IDS:
        match = SIGN_PREFIX_PATTERN.match(filename)
        if not match:
            raise ValueError(
                f"Sign image does not match numeric-prefix_frame pattern: {source_image}"
            )
        return f"sign:{label_id}:{match.group(1)}"

    return f"alphabet:{label_id}:{source_image}"


def split_one_class(
    sample_indices: np.ndarray,
    sample_groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = np.unique(sample_groups)
    if len(unique_groups) < 3:
        raise ValueError(
            f"Class has only {len(unique_groups)} source groups; "
            "cannot place it in train, validation, and test."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_local, temporary_local = next(
        splitter.split(sample_indices, groups=sample_groups)
    )

    temporary_indices = sample_indices[temporary_local]
    temporary_groups = sample_groups[temporary_local]
    inner_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=seed + 1
    )
    validation_local, test_local = next(
        inner_splitter.split(temporary_indices, groups=temporary_groups)
    )

    return (
        sample_indices[train_local],
        temporary_indices[validation_local],
        temporary_indices[test_local],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Models_candidate") / "splits",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing candidate split files in the output directory.",
    )
    args = parser.parse_args()

    features_path = KEYPOINTS_DIR / "keypoints_raw.npy"
    labels_path = KEYPOINTS_DIR / "labels_raw.npy"
    manifest_path = KEYPOINTS_DIR / "keypoints_raw_manifest.csv"
    for path in (features_path, labels_path, manifest_path):
        if not path.exists():
            parser.error(f"Required raw extraction file not found: {path}")

    features = np.load(features_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        manifest = list(csv.DictReader(manifest_file))

    if not (len(features) == len(labels) == len(manifest)):
        parser.error(
            "Feature, label, and manifest lengths differ: "
            f"{len(features)}, {len(labels)}, {len(manifest)}"
        )
    if features.ndim != 2 or features.shape[1] != 126:
        parser.error(f"Expected raw features with shape (N, 126), got {features.shape}")

    manifest_labels = np.asarray([int(row["label_id"]) for row in manifest])
    if not np.array_equal(labels, manifest_labels):
        parser.error("labels_raw.npy does not match label_id order in the manifest")

    output_files = (
        "train_idx.npy",
        "val_idx.npy",
        "test_idx.npy",
        "excluded_idx.npy",
        "group_assignments.csv",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in output_files if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        parser.error(
            "Candidate split output already exists; use --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )

    excluded_indices = np.flatnonzero(labels == NOTHING_LABEL_ID)
    train_parts = []
    validation_parts = []
    test_parts = []
    assignments = []

    for label_id in sorted(set(map(int, labels)) - {NOTHING_LABEL_ID}):
        sample_indices = np.flatnonzero(labels == label_id)
        if label_id not in CLASS_NAMES:
            parser.error(f"Raw dataset contains unknown label ID {label_id}")

        groups = np.asarray(
            [source_group_id(label_id, manifest[index]["source_image"]) for index in sample_indices]
        )
        train_idx, validation_idx, test_idx = split_one_class(
            sample_indices, groups, args.seed + label_id * 2
        )
        train_parts.append(train_idx)
        validation_parts.append(validation_idx)
        test_parts.append(test_idx)

        split_by_group = {}
        for split_name, split_indices in (
            ("train", train_idx),
            ("validation", validation_idx),
            ("test", test_idx),
        ):
            for index in split_indices:
                split_by_group[groups[np.searchsorted(sample_indices, index)]] = split_name

        group_sizes = Counter(groups.tolist())
        for group_id, split_name in split_by_group.items():
            assignments.append(
                (split_name, label_id, CLASS_NAMES[label_id], group_id, group_sizes[group_id])
            )

    split_indices = {
        "train_idx.npy": np.sort(np.concatenate(train_parts)).astype(np.int64),
        "val_idx.npy": np.sort(np.concatenate(validation_parts)).astype(np.int64),
        "test_idx.npy": np.sort(np.concatenate(test_parts)).astype(np.int64),
        "excluded_idx.npy": excluded_indices.astype(np.int64),
    }

    split_sets = {name: set(indices.tolist()) for name, indices in split_indices.items()}
    for first, second in (("train_idx.npy", "val_idx.npy"), ("train_idx.npy", "test_idx.npy"), ("val_idx.npy", "test_idx.npy")):
        if split_sets[first] & split_sets[second]:
            raise RuntimeError(f"Sample index leakage between {first} and {second}")

    group_split = {}
    for split_name, label_id, _, group_id, _ in assignments:
        prior_split = group_split.setdefault(group_id, split_name)
        if prior_split != split_name:
            raise RuntimeError(f"Source group {group_id} was split across partitions")

    for filename, indices in split_indices.items():
        np.save(args.output_dir / filename, indices)

    with (args.output_dir / "group_assignments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as audit_file:
        writer = csv.writer(audit_file)
        writer.writerow(("split", "label_id", "class_name", "group_id", "sample_count"))
        writer.writerows(assignments)

    print(f"Raw samples: {len(labels):,}; feature width: {features.shape[1]}")
    print(
        f"Excluded from classifier splits: {len(excluded_indices)} samples "
        f"(nothing label {NOTHING_LABEL_ID}; insufficient valid hand examples)."
    )
    for split_name, filename in (("train", "train_idx.npy"), ("validation", "val_idx.npy"), ("test", "test_idx.npy")):
        indices = split_indices[filename]
        split_counts = Counter(map(int, labels[indices]))
        missing = sorted(set(CLASS_NAMES) - {NOTHING_LABEL_ID} - set(split_counts))
        print(
            f"{split_name:10}: {len(indices):6,} samples "
            f"({len(indices) / (len(labels) - len(excluded_indices)):.1%}), "
            f"{len(split_counts)} classes, missing labels={missing}"
        )
        if missing:
            raise RuntimeError(f"{split_name} is missing trainable class IDs: {missing}")

    print(f"Group assignments: {args.output_dir / 'group_assignments.csv'}")
    print("No augmentation or model training was performed.")
    print("Alphabet groups are individual images; use held-out webcam sessions for real-world evaluation.")


if __name__ == "__main__":
    main()
