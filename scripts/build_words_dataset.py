"""Stream the ASL Citizen clips for the chosen words (and digit signs), extract landmarks, delete each clip.

Writes artifacts/vocab_words.json, data/processed/words.npz and data/processed/digits_video.npz.
Run: .venv/bin/python -m scripts.build_words_dataset [--limit N] [--workers N]
"""
import argparse
import collections
import csv
import json
import multiprocessing as mp
import re
import tempfile
import time

import numpy as np

from scripts.download_data import CITIZEN_ZIP
from scripts.remote_zip import open_remote_zip, read_member
from signbridge.config import ARTIFACTS, PROCESSED, RAW
from signbridge.landmarks import prepare_sequence

N_WORDS = 100
DIGIT_GLOSSES = {"ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
                 "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9"}  # ASL Citizen has no FIVE
DIGIT_FRAMES = 10  # still frames kept per digit clip
# Ordered by importance for the course; the first N_WORDS that exist in ASL Citizen are kept.
PRIORITY = """hello bye please thankyou sorry yes no  why where who what how when which  me my you he we
mother father brother sister boy girl man woman child friend teacher baby  happy sad angry sick tired love hate like
drink water milk hungry eat apple pizza bread coffee
go come give see hear talk read write sleep find make have can will open close think wait help finish want need play learn know
now later tomorrow yesterday today morning night time before after  home school work book car phone computer table
red blue green yellow black white  dog cat bird  understand name again more good bad hot cold""".split()



def read_rows():
    rows = []
    for split in ("train", "val", "test"):
        with open(RAW / "asl-citizen" / f"{split}.csv") as f:
            rows += [{**r, "split": split} for r in csv.DictReader(f)]
    return rows


def resolve(words, counts):
    """English word -> gloss: among WORD, WORD1, WORD2..., the one with the most clips (ties: alphabetical)."""
    out = {}
    for w in words:
        pat = re.compile(rf"^{w.upper()}\d*$")
        found = [(-counts[g], g) for g in counts if pat.match(g)]
        if found:
            out[w] = min(found)[1]
    return out


def pick_digit_frames(hands_per_frame, k=DIGIT_FRAMES):
    """k evenly spaced frames from the middle 60% of the frames that show the dominant hand."""
    lh = [h[0] for h in hands_per_frame if h[0] is not None]
    rh = [h[1] for h in hands_per_frame if h[1] is not None]
    seen = rh if len(rh) >= len(lh) else lh
    if len(seen) < 5:
        return []
    mid = seen[int(len(seen) * 0.2):int(len(seen) * 0.8) + 1]
    return [mid[i] for i in np.linspace(0, len(mid) - 1, min(k, len(mid))).round().astype(int)]


def work(task):
    """One clip: download the member, extract landmarks, delete the file. Runs in a worker process."""
    from scripts.video_landmarks import video_landmarks
    data = None
    for attempt in range(3):
        try:
            data = read_member(CITIZEN_ZIP, *task["loc"])
            break
        except Exception:
            time.sleep(2 * (attempt + 1))
    if data is None:
        return {**task, "error": "download"}
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:  # deleted on close: raw videos are never kept
        f.write(data)
        f.flush()
        out = video_landmarks(f.name)
    if out is None:
        return {**task, "error": "decode"}
    if task["kind"] == "word":
        return {**task, "seq": prepare_sequence(out["seq"]).astype(np.float16)}
    frames = pick_digit_frames(out["hands"])
    return {**task, "hands": np.stack(frames) if frames else None, "aspect": out["aspect"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    rows = read_rows()
    counts = collections.Counter(r["Gloss"] for r in rows)
    chosen = resolve(PRIORITY, counts)
    labels = list(chosen)[:N_WORDS]
    print("unresolved words:", [w for w in PRIORITY if w not in chosen])
    assert len(labels) == N_WORDS, f"only {len(labels)} words resolved"
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "vocab_words.json").write_text(json.dumps({"labels": labels, "glosses": {w: chosen[w] for w in labels}}, indent=1))

    gloss_to_word = {chosen[w]: i for i, w in enumerate(labels)}
    index = open_remote_zip(CITIZEN_ZIP)  # parsed once here; workers only get byte offsets
    tasks = []
    for r in rows:
        g = r["Gloss"]
        if g in gloss_to_word:
            kind, label = "word", gloss_to_word[g]
        elif g in DIGIT_GLOSSES:
            kind, label = "digit", DIGIT_GLOSSES[g]
        else:
            continue
        zi = index.getinfo("ASL_Citizen/videos/" + r["Video file"])
        tasks.append({"kind": kind, "label": label, "video": r["Video file"], "split": r["split"],
                      "participant": int(r["Participant ID"].lstrip("P")),
                      "loc": (zi.header_offset, zi.compress_size, zi.compress_type)})
    del index
    tasks = tasks[:args.limit] if args.limit else tasks
    print(len(tasks), "clips:", collections.Counter(t["kind"] for t in tasks))

    words, digits, errors, t0 = [], [], collections.Counter(), time.time()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=100) as pool:
        for n, res in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
            if "error" in res:
                errors[res["error"]] += 1
            elif res["kind"] == "word":
                words.append(res)
            elif res["hands"] is not None:
                digits.append(res)
            else:
                errors["no hand"] += 1
            if n % 100 == 0:
                print(f"{n}/{len(tasks)}  {time.time() - t0:.0f}s  errors {dict(errors)}", flush=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED / "words.npz", X=np.stack([w["seq"] for w in words]), y=np.array([w["label"] for w in words], dtype=np.int16),
        participant=np.array([w["participant"] for w in words], dtype=np.int32),
        split=np.array([w["split"] for w in words]), video=np.array([w["video"] for w in words]))
    if digits:
        np.savez_compressed(
            PROCESSED / "digits_video.npz", hands=np.concatenate([d["hands"] for d in digits]),
            aspect=np.concatenate([[d["aspect"]] * len(d["hands"]) for d in digits]).astype(np.float32),
            label=np.concatenate([[d["label"]] * len(d["hands"]) for d in digits]),
            participant=np.concatenate([[d["participant"]] * len(d["hands"]) for d in digits]).astype(np.int32),
            split=np.concatenate([[d["split"]] * len(d["hands"]) for d in digits]))
    per = collections.Counter(w["label"] for w in words)
    print(f"done in {time.time() - t0:.0f}s: {len(words)} word clips (per class min {min(per.values())} max {max(per.values())}), "
          f"{len(digits)} digit clips, errors {dict(errors)}")


if __name__ == "__main__":
    main()
