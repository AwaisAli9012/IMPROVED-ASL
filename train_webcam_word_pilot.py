"""Train and evaluate a small word-sequence pilot across separate webcam sessions.

This is an exploratory session-held-out test, not a production training pipeline.
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
SEQUENCE_ROOT = BASE_DIR / "keypoints_extracted" / "word_sequences"
CANDIDATE_ROOT = BASE_DIR / "Models_candidate"


def clip_to_features(clip_path: Path) -> tuple[np.ndarray, str, str, int]:
    with np.load(clip_path, allow_pickle=False) as clip:
        keypoints = np.asarray(clip["keypoints"], dtype=np.float32)
        frame_numbers = np.asarray(clip["frame_numbers"], dtype=np.float32)
        label_id = int(clip["label_id"])
        class_name = str(clip["class_name"])

    if keypoints.ndim != 2 or keypoints.shape[1] != 126:
        raise ValueError(f"Expected {clip_path.name} keypoints shaped (T, 126)")
    if len(keypoints) == 0 or len(frame_numbers) != len(keypoints):
        raise ValueError(f"Invalid or empty frame data in {clip_path}")
    if np.any(np.diff(frame_numbers) < 0):
        raise ValueError(f"Frame numbers are not chronological in {clip_path}")

    mean = np.mean(keypoints, axis=0)
    std = np.std(keypoints, axis=0)
    minimum = np.min(keypoints, axis=0)
    maximum = np.max(keypoints, axis=0)

    if len(keypoints) > 1:
        frame_gaps = np.diff(frame_numbers)
        frame_gaps = np.where(frame_gaps <= 0, 1.0, frame_gaps)[:, None]
        velocity = np.diff(keypoints, axis=0) / frame_gaps
        mean_velocity = np.mean(np.abs(velocity), axis=0)
        max_velocity = np.max(np.abs(velocity), axis=0)
        displacement = keypoints[-1] - keypoints[0]
        duration = frame_numbers[-1] - frame_numbers[0]
    else:
        mean_velocity = np.zeros_like(mean)
        max_velocity = np.zeros_like(mean)
        displacement = np.zeros_like(mean)
        duration = 1.0

    features = np.concatenate(
        [
            mean,
            std,
            minimum,
            maximum,
            mean_velocity,
            max_velocity,
            displacement,
            np.asarray([len(keypoints), duration], dtype=np.float32),
        ]
    )
    return features, class_name, str(label_id), len(keypoints)


def load_session(session_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    session_dir = SEQUENCE_ROOT / session_id
    manifest_path = session_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Session manifest not found: {manifest_path}")

    feature_rows = []
    labels = []
    clip_ids = []
    seen_clips = set()
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        for row in csv.DictReader(manifest_file):
            if row["session_id"] != session_id:
                raise ValueError(f"Unexpected session ID in {manifest_path}: {row['session_id']}")
            clip_id = row["clip_id"]
            if clip_id in seen_clips:
                raise ValueError(f"Duplicate clip ID in {manifest_path}: {clip_id}")
            seen_clips.add(clip_id)

            clip_path = session_dir / row["clip_file"]
            features, class_name, label_id, frame_count = clip_to_features(clip_path)
            if class_name != row["class_name"] or label_id != row["label_id"]:
                raise ValueError(f"Clip metadata does not match manifest: {clip_path}")
            if frame_count != int(row["frame_count"]):
                raise ValueError(f"Manifest frame count does not match clip: {clip_path}")

            feature_rows.append(features)
            labels.append(class_name)
            clip_ids.append(f"{session_id}/{clip_id}")

    if not feature_rows:
        raise ValueError(f"No clips found in session {session_id}")
    return np.vstack(feature_rows), np.asarray(labels), clip_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-sessions",
        nargs="+",
        default=["session1"],
        help="One or more session IDs used for training (default: session1).",
    )
    parser.add_argument("--test-session", default="session2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CANDIDATE_ROOT / "webcam_word_pilot_session1_session3_to_session2",
    )
    args = parser.parse_args()

    if len(set(args.train_sessions)) != len(args.train_sessions):
        parser.error("Training session IDs must not be repeated")
    if args.test_session in args.train_sessions:
        parser.error("The held-out test session must not appear in --train-sessions")

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"Output directory is not empty; choose another --output-dir: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    training_data = [load_session(session_id) for session_id in args.train_sessions]
    train_features = np.vstack([session[0] for session in training_data])
    train_labels = np.concatenate([session[1] for session in training_data])
    train_clips = [clip_id for session in training_data for clip_id in session[2]]
    test_features, test_labels, test_clips = load_session(args.test_session)

    train_classes = set(train_labels)
    test_classes = set(test_labels)
    if train_classes != test_classes:
        raise ValueError(
            f"Sessions must contain the same classes. Train-only={train_classes - test_classes}; "
            f"test-only={test_classes - train_classes}"
        )

    class_names = sorted(train_classes)
    label_to_index = {name: index for index, name in enumerate(class_names)}
    y_train = np.asarray([label_to_index[name] for name in train_labels], dtype=np.int64)
    y_test = np.asarray([label_to_index[name] for name in test_labels], dtype=np.int64)

    print(f"Training sessions {args.train_sessions}: {len(y_train)} clips")
    print(f"Held-out session {args.test_session}: {len(y_test)} clips")
    print(f"Classes: {class_names}")
    print("Caution: with five clips per class, this is a small exploratory test.")

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    reports = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(train_features, y_train)
        predictions = model.predict(test_features)
        report = classification_report(
            y_test,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        reports[model_name] = report
        print(classification_report(
            y_test,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=3,
            zero_division=0,
        ))
        print("Confusion matrix (rows=true, columns=predicted):")
        print(confusion_matrix(y_test, predictions, labels=list(range(len(class_names)))))
        joblib.dump(model, output_dir / f"{model_name}.pkl")

    mapping = {str(index): name for index, name in enumerate(class_names)}
    (output_dir / "label_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    summary = {
        "train_sessions": args.train_sessions,
        "test_session": args.test_session,
        "train_clip_ids": train_clips,
        "test_clip_ids": test_clips,
        "class_names": class_names,
        "reports": reports,
        "note": "Single held-out session; exploratory only, not a production performance estimate.",
    }
    (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Candidate models and evaluation saved under {output_dir}.")
    print("Production models were not modified.")


if __name__ == "__main__":
    main()
