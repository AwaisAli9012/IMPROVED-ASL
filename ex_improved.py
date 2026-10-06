"""Extract raw 126-value hand keypoints for the ASL image datasets."""

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from Config import (
    ALPHABETS_DIR,
    KEYPOINTS_DIR,
    SIGN_CLASSES,
    ALPHABET_CLASSES,
    SIGNS_FRAMES_DIR,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FEATURE_COUNT = 126


def list_class_images(folder: Path, limit: int) -> list[Path]:
    if not folder.is_dir():
        return []
    images = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return images[:limit]


def extract_keypoints(image_path: Path, hands_detector) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_image)
    if not results.multi_hand_landmarks:
        return None

    values = []
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            values.extend((landmark.x, landmark.y, landmark.z))

    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    copied_count = min(len(values), FEATURE_COUNT)
    features[:copied_count] = values[:copied_count]
    return features


def build_class_sources(max_images_per_class: int):
    classes = []
    for class_id, class_name in SIGN_CLASSES.items():
        classes.append((class_id, class_name, SIGNS_FRAMES_DIR / class_name))
    for class_id, class_name in ALPHABET_CLASSES.items():
        classes.append((36 + class_id, class_name, ALPHABETS_DIR / class_name))

    return [
        (class_id, class_name, list_class_images(folder, max_images_per_class))
        for class_id, class_name, folder in classes
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=3000,
        help="Maximum source images attempted per class (default: 3000).",
    )
    parser.add_argument(
        "--class-name",
        action="append",
        help="Limit extraction to a class name; may be supplied multiple times for a pilot.",
    )
    parser.add_argument(
        "--output-prefix",
        default="raw",
        help="Output filename prefix (default: raw; writes keypoints_<prefix>.npy, labels_<prefix>.npy, and a manifest).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the raw output files if they already exist.",
    )
    args = parser.parse_args()

    if args.max_images_per_class <= 0:
        parser.error("--max-images-per-class must be greater than zero")
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        parser.error("--output-prefix must be a simple filename prefix")

    KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    keypoints_path = KEYPOINTS_DIR / f"keypoints_{args.output_prefix}.npy"
    labels_path = KEYPOINTS_DIR / f"labels_{args.output_prefix}.npy"
    manifest_path = KEYPOINTS_DIR / f"keypoints_{args.output_prefix}_manifest.csv"
    output_paths = (keypoints_path, labels_path, manifest_path)

    if not args.overwrite and any(path.exists() for path in output_paths):
        parser.error(
            "Raw output already exists. Move it aside or pass --overwrite to replace it."
        )

    class_sources = build_class_sources(args.max_images_per_class)
    if args.class_name:
        requested = set(args.class_name)
        known = {class_name for _, class_name, _ in class_sources}
        unknown = requested - known
        if unknown:
            parser.error(f"Unknown class name(s): {', '.join(sorted(unknown))}")
        class_sources = [
            item for item in class_sources if item[1] in requested
        ]

    total_images = sum(len(images) for _, _, images in class_sources)
    print(f"Attempting up to {total_images:,} source images across {len(class_sources)} classes.")
    print("Only original extracted keypoints are saved; augmentation is not applied.")

    keypoint_chunks = []
    label_chunks = []
    manifest_rows = []
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as detector:
        for class_id, class_name, image_paths in class_sources:
            class_features = []
            class_sources_used = []
            unreadable_count = 0

            for image_path in image_paths:
                features = extract_keypoints(image_path, detector)
                if features is None:
                    unreadable_count += 1
                    continue
                class_features.append(features)
                class_sources_used.append(image_path)

            if class_features:
                feature_array = np.asarray(class_features, dtype=np.float32)
                keypoint_chunks.append(feature_array)
                label_chunks.append(
                    np.full(len(feature_array), class_id, dtype=np.int32)
                )
                manifest_rows.extend(
                    (class_id, class_name, str(path)) for path in class_sources_used
                )

            print(
                f"{class_name:10} label={class_id:2} "
                f"images={len(image_paths):4} extracted={len(class_features):4} "
                f"unreadable/no-hand={unreadable_count:4}"
            )

    if not keypoint_chunks:
        raise RuntimeError("No hand keypoints were extracted; output files were not written.")

    keypoints = np.concatenate(keypoint_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    np.save(keypoints_path, keypoints)
    np.save(labels_path, labels)

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(("label_id", "class_name", "source_image"))
        writer.writerows(manifest_rows)

    print(f"\nSaved {len(labels):,} raw samples with shape {keypoints.shape}.")
    print(f"Keypoints: {keypoints_path}")
    print(f"Labels:    {labels_path}")
    print(f"Manifest:  {manifest_path}")
    print("Existing keypoints.npy and labels.npy were not modified.")


if __name__ == "__main__":
    main()