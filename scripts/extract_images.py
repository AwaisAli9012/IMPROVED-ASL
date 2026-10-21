"""Landmarks from still images: hand landmarks for letters (static; digits come from the ASL Citizen clips), face blendshapes for emotion.

Run: .venv/bin/python -m scripts.extract_images --task static|emotion [--workers 4]
Writes data/processed/static.npz or data/processed/emotion.npz.
"""
import argparse
import collections
import io
import multiprocessing as mp
import time

import mediapipe as mp_vision
import numpy as np
import pyarrow.parquet as pq
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from PIL import Image

from signbridge.config import EMOTIONS, LETTERS, PROCESSED, RAW, WEB

MODELS = WEB / "static" / "models"
FER_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
FER_SPLIT = {"train": "train", "publicTest": "val", "privateTest": "test"}  # the standard FER2013 protocol
FACE_MIN_SIDE = 192  # FER2013 is 48x48: upscale so the face detector can find it

_landmarker = None


def init(task):
    global _landmarker
    if task == "static":
        opts = vision.HandLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
                                            num_hands=1, min_hand_detection_confidence=0.3)
        _landmarker = vision.HandLandmarker.create_from_options(opts)
    else:
        opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
                                            output_face_blendshapes=True, num_faces=1)
        _landmarker = vision.FaceLandmarker.create_from_options(opts)


def _mp_image(png, min_side=0):
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if min(img.size) < min_side:
        k = min_side / min(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.BICUBIC)
    return mp_vision.Image(image_format=mp_vision.ImageFormat.SRGB, data=np.asarray(img)), img.width / img.height


def detect_hands(batch):
    """-> [(hand (21, 3), aspect) or None] for a list of encoded images."""
    out = []
    for png in batch:
        image, aspect = _mp_image(png)
        res = _landmarker.detect(image).hand_landmarks
        out.append((np.array([[p.x, p.y, p.z] for p in res[0]], dtype=np.float32), aspect) if res else None)
    return out


def detect_faces(batch):
    """-> [(blendshapes (52,), names) or None]."""
    out = []
    for png in batch:
        image, _ = _mp_image(png, FACE_MIN_SIDE)
        res = _landmarker.detect(image).face_blendshapes
        out.append((np.array([c.score for c in res[0]], dtype=np.float32), [c.category_name for c in res[0]]) if res else None)
    return out


def batches(path, label_col="label", size=64):
    """Yield (encoded images, labels) from a parquet file with image{bytes} and a label column."""
    for b in pq.ParquetFile(path).iter_batches(batch_size=size):
        rows = b.to_pylist()
        yield [r["image"]["bytes"] for r in rows], [r[label_col] for r in rows]


def run(pool, fn, path, names, source, keep):
    """Detect over one parquet file. Returns the kept results plus counts for the detection-rate printout."""
    kept, seen, found, t0 = [], 0, 0, time.time()
    jobs = list(batches(path))
    for (imgs, labels), res in zip(jobs, pool.imap(fn, [j[0] for j in jobs])):
        for label, r in zip(labels, res):
            name = names[label]
            if name not in keep:
                continue
            seen += 1
            if r is not None:
                found += 1
                kept.append((name, r))
    print(f"{source}: {found}/{seen} images with a detection ({found / max(seen, 1):.0%}) in {time.time() - t0:.0f}s", flush=True)
    return kept


def build_static(pool):
    hands, aspect, label, source, split = [], [], [], [], []
    for lab, (hand, asp) in run(pool, detect_hands, next((RAW / "asl-letters").glob("*.parquet")), LETTERS, "letters", set(LETTERS)):
        hands.append(hand)
        aspect.append(asp)
        label.append(lab)
        source.append("letters")
        split.append("")
    video = PROCESSED / "digits_video.npz"
    if video.exists():
        d = np.load(video)
        print(f"digit-video: {len(d['hands'])} frames from {len(set(d['participant']))} signers")
        hands += list(d["hands"])
        aspect += list(d["aspect"])
        label += list(d["label"])
        source += ["digit-video"] * len(d["hands"])
        split += list(d["split"])
    np.savez_compressed(PROCESSED / "static.npz", hands=np.stack(hands), aspect=np.array(aspect, dtype=np.float32),
                        label=np.array(label), source=np.array(source), split=np.array(split))
    print(collections.Counter(zip(source, label)).most_common(3), "...", len(hands), "samples")


def build_emotion(pool):
    bs, label, split, names = [], [], [], None
    for fer_split, ours in FER_SPLIT.items():
        path = next((RAW / "fer2013").glob(f"{fer_split}-*.parquet"))
        for lab, (scores, cats) in run(pool, detect_faces, path, FER_NAMES, f"fer2013 {fer_split}", set(EMOTIONS)):
            bs.append(scores), label.append(lab), split.append(ours)
            names = names or cats
    np.savez_compressed(PROCESSED / "emotion.npz", bs=np.stack(bs), label=np.array(label), split=np.array(split), names=np.array(names))
    print(collections.Counter(zip(split, label)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["static", "emotion"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(args.workers, initializer=init, initargs=(args.task,)) as pool:
        (build_static if args.task == "static" else build_emotion)(pool)


if __name__ == "__main__":
    main()
