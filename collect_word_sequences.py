"""Collect labeled webcam word-sign sequences into isolated per-session files.

Controls: Space starts/stops and saves a clip, N/P changes the selected class,
Q quits (an in-progress clip is discarded).
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re

import cv2
import mediapipe as mp
import numpy as np

from Config import KEYPOINTS_DIR, SIGN_CLASSES

FEATURE_COUNT = 126


def extract_hand_keypoints(results) -> np.ndarray | None:
    """Return raw normalized MediaPipe coordinates padded to two 21-point hands."""
    if not results.multi_hand_landmarks:
        return None

    values = []
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            values.extend((landmark.x, landmark.y, landmark.z))

    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    used_values = min(len(values), FEATURE_COUNT)
    features[:used_values] = values[:used_values]
    return features


def parse_classes(value: str | None) -> list[tuple[int, str]]:
    if not value:
        return list(SIGN_CLASSES.items())

    requested = [part.strip() for part in value.split(",") if part.strip()]
    if not requested:
        raise ValueError("--classes must contain at least one sign name")

    name_to_id = {name.lower(): (class_id, name) for class_id, name in SIGN_CLASSES.items()}
    selected = []
    unknown = []
    for name in requested:
        match = name_to_id.get(name.lower())
        if match is None:
            unknown.append(name)
        elif match not in selected:
            selected.append(match)
    if unknown:
        raise ValueError(f"Unknown sign class(es): {', '.join(unknown)}")
    return selected


def make_session_id(value: str | None) -> str:
    session_id = value or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
        raise ValueError("Session ID may contain only letters, digits, underscores, and hyphens")
    return session_id


def save_clip(
    session_dir: Path,
    manifest_path: Path,
    session_id: str,
    clip_number: int,
    label_id: int,
    class_name: str,
    frame_numbers: list[int],
    clip_features: list[np.ndarray],
) -> Path:
    clip_id = f"{session_id}_{clip_number:04d}"
    clip_path = session_dir / f"{clip_id}.npz"
    if clip_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing clip: {clip_path}")

    feature_array = np.asarray(clip_features, dtype=np.float32)
    frame_array = np.asarray(frame_numbers, dtype=np.int32)
    temporary_path = clip_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as clip_file:
        np.savez_compressed(
            clip_file,
            keypoints=feature_array,
            frame_numbers=frame_array,
            label_id=np.asarray(label_id, dtype=np.int32),
            class_name=np.asarray(class_name),
            session_id=np.asarray(session_id),
            clip_id=np.asarray(clip_id),
        )
    temporary_path.replace(clip_path)

    manifest_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        if not manifest_exists:
            writer.writerow(
                ("session_id", "clip_id", "label_id", "class_name", "frame_count", "clip_file")
            )
        writer.writerow(
            (session_id, clip_id, label_id, class_name, len(feature_array), clip_path.name)
        )
        manifest_file.flush()

    return clip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes",
        help="Comma-separated sign names to collect (default: all configured sign words).",
    )
    parser.add_argument("--session-id", help="Session label; defaults to a timestamped ID.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=KEYPOINTS_DIR / "word_sequences",
        help="Root directory for collected clips (default: keypoints_extracted/word_sequences).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="Minimum detected-hand frames required to save a clip (default: 5).",
    )
    args = parser.parse_args()

    if args.min_frames < 1:
        parser.error("--min-frames must be at least 1")
    try:
        classes = parse_classes(args.classes)
        session_id = make_session_id(args.session_id)
    except ValueError as exc:
        parser.error(str(exc))

    session_dir = args.output_dir / session_id
    if session_dir.exists():
        if not session_dir.is_dir() or any(session_dir.iterdir()):
            parser.error(
                f"Session directory already contains data; choose a new session ID: {session_dir}"
            )
    else:
        session_dir.mkdir(parents=True, exist_ok=False)

    manifest_path = session_dir / "manifest.csv"
    capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        session_dir.rmdir()
        raise RuntimeError(
            f"Could not open camera index {args.camera}. Close other apps using the camera and retry."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    class_position = 0
    clip_number = 1
    frame_number = 0
    recording = False
    clip_features: list[np.ndarray] = []
    clip_frame_numbers: list[int] = []
    no_hand_frames = 0
    saved_clips = 0
    mp_hands = mp.solutions.hands

    print(f"Session: {session_id}")
    print(f"Output:  {session_dir}")
    print(f"Classes ({len(classes)}): {', '.join(name for _, name in classes)}")
    print("Space: start/stop+save clip | N: next class | P: previous class | Q: quit")
    print("Record multiple separate performances; use a new --session-id for each session.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands_detector:
            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    print("Camera frame read failed; ending collection.")
                    break

                frame_number += 1
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_detector.process(rgb_frame)

                if recording:
                    features = extract_hand_keypoints(results)
                    if features is None:
                        no_hand_frames += 1
                    else:
                        clip_features.append(features)
                        clip_frame_numbers.append(frame_number)
                        for hand_landmarks in results.multi_hand_landmarks or []:
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                            )

                _, class_name = classes[class_position]
                status = "RECORDING" if recording else "READY"
                cv2.putText(frame, f"{status} | Sign: {class_name}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Detected frames: {len(clip_features)} | No hand: {no_hand_frames}", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"Saved clips this session: {saved_clips}", (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("ASL Word Sequence Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    if recording:
                        print("Active clip discarded on quit.")
                    break
                if key == ord("n") and not recording:
                    class_position = (class_position + 1) % len(classes)
                elif key == ord("p") and not recording:
                    class_position = (class_position - 1) % len(classes)
                elif key == ord(" "):
                    if not recording:
                        recording = True
                        clip_features = []
                        clip_frame_numbers = []
                        no_hand_frames = 0
                        print(f"Recording {class_name}; perform one complete sign, then press Space.")
                    else:
                        recording = False
                        if len(clip_features) >= args.min_frames:
                            label_id, class_name = classes[class_position]
                            saved_path = save_clip(
                                session_dir,
                                manifest_path,
                                session_id,
                                clip_number,
                                label_id,
                                class_name,
                                clip_frame_numbers,
                                clip_features,
                            )
                            print(
                                f"Saved {class_name}: {len(clip_features)} detected frames "
                                f"({no_hand_frames} no-hand frames skipped) -> {saved_path.name}"
                            )
                            clip_number += 1
                            saved_clips += 1
                        else:
                            print(
                                f"Discarded clip: {len(clip_features)} detected frames; "
                                f"need at least {args.min_frames}."
                            )
                        clip_features = []
                        clip_frame_numbers = []
                        no_hand_frames = 0
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Collection ended. Saved {saved_clips} clips in {session_dir}.")


if __name__ == "__main__":
    main()
