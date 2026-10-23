"""Replay held-out test clips through a running server's /api/predict/word and print top-1, top-3 and latency.
Run with the server up: .venv/bin/python -m scripts.check_api [--n 300]"""
import argparse
import json
import time

import httpx
import numpy as np

from signbridge import config
from signbridge.vocab import word_labels


def frames(x: np.ndarray) -> list[dict]:
    block = lambda a: None if np.isnan(a).all() else np.nan_to_num(a).round(4).tolist()  # noqa: E731
    return [{"lh": block(f[config.LH]), "rh": block(f[config.RH]), "pose": block(f[config.POSE]), "lips": block(f[config.LIPS])} for f in x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--url", default=f"http://{config.HOST}:{config.PORT}")
    a = ap.parse_args()
    path = config.PROCESSED / "words.npz"
    if not path.exists():
        return print("data/processed/words.npz is missing: nothing to replay (see the README to rebuild it)")
    d = np.load(path)
    test = set(json.loads((config.REPORTS / "words.json").read_text())["split"]["test_signers"])
    idx = np.flatnonzero(np.isin(d["participant"], list(test)))
    idx = np.random.default_rng(config.SEED).choice(idx, min(a.n, len(idx)), replace=False)
    labels = word_labels()
    X, y = d["X"], d["y"]  # an npz member is decompressed on every access, so read each once
    top1 = top3 = 0
    ms = []
    with httpx.Client(timeout=30) as c:
        for i in idx:
            body = {"frames": frames(X[i].astype(np.float32))}  # built before the clock starts: only the round trip is timed
            t = time.perf_counter()
            r = c.post(f"{a.url}/api/predict/word", json=body)
            r.raise_for_status()
            top = r.json()["top"]
            ms.append((time.perf_counter() - t) * 1000)
            want = labels[int(y[i])]
            top1 += top[0]["label"] == want
            top3 += want in [t["label"] for t in top[:3]]
    print(f"{len(idx)} clips: top-1 {top1 / len(idx):.1%}, top-3 {top3 / len(idx):.1%}, median {np.median(ms):.0f} ms, p95 {np.percentile(ms, 95):.0f} ms")


if __name__ == "__main__":
    main()
