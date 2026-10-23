"""Evaluate a trained model on held-out data and write reports/<task>.json plus confusion and per-class F1 charts.

Run: .venv/bin/python -m scripts.evaluate words|static|emotion
Every number shown anywhere in the app comes from these reports.
"""
import argparse
import datetime
import json
import subprocess
import time

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

from signbridge.config import ARTIFACTS, EMOTIONS, PROCESSED, REPORTS
from signbridge.landmarks import hand_features, sequence_features
from signbridge.models import WordPredictor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_PRECISION = 0.95
MIN_ACCEPTED = 10  # fewest predictions a threshold may keep on val, so one lucky sample cannot pick it


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def accept_threshold(conf, correct):
    """Smallest confidence whose accepted predictions are >= 95% precise on val; else the most precise one."""
    best_t, best_p = None, -1.0
    for t in np.arange(0.30, 0.96, 0.01):
        keep = conf >= t
        if keep.sum() < MIN_ACCEPTED:
            continue
        prec = correct[keep].mean()
        if prec >= TARGET_PRECISION:
            return round(float(t), 2)
        if prec > best_p:
            best_t, best_p = round(float(t), 2), prec
    return best_t


def plots(task, labels, y, pred, per_class_f1):
    n = len(labels)
    cm = confusion_matrix(y, pred, labels=range(n)).astype(float)
    cm /= np.maximum(cm.sum(1, keepdims=True), 1)
    s = max(6, n * 0.18)
    fig, ax = plt.subplots(figsize=(s, s))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=max(4, 9 - n // 20))
    ax.set_yticks(range(n), labels, fontsize=max(4, 9 - n // 20))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (rows sum to 1)")
    fig.tight_layout()
    fig.savefig(REPORTS / f"{task}_confusion.png", dpi=100)
    plt.close(fig)
    order = np.argsort(per_class_f1)
    fig, ax = plt.subplots(figsize=(7, max(4, n * 0.17)))
    ax.barh([labels[i] for i in order], per_class_f1[order], color="#d98f1f")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (test)")
    ax.tick_params(axis="y", labelsize=max(5, 9 - n // 25))
    fig.tight_layout()
    fig.savefig(REPORTS / f"{task}_per_class_f1.png", dpi=100)
    plt.close(fig)


def report(task, model, labels, val, test, split, latency_ms, notes="", per_source=None):
    """val and test are (y_true, probabilities). Writes the JSON and charts, returns the dict."""
    (yv, pv), (y, p) = val, test
    n = len(labels)
    pred = p.argmax(1)
    correct_v, correct = (pv.argmax(1) == yv), (pred == y)
    conf_v, conf = pv.max(1), p.max(1)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y, pred, labels=range(n), zero_division=0)
    t = accept_threshold(conf_v, correct_v)
    keep = conf >= t if t is not None else np.zeros(len(y), bool)
    cm = confusion_matrix(y, pred, labels=range(n))
    off = [(int(cm[i, j]), i, j) for i in range(n) for j in range(n) if i != j and cm[i, j]]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    out = {
        "task": task, "model": model, "trained_at": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
        "n_classes": n, "labels": labels, "split": split, "notes": notes,
        "accuracy": float(accuracy_score(y, pred)),
        "top3": float(top_k_accuracy_score(y, p, k=3, labels=range(n))) if n > 3 else None,
        "top5": float(top_k_accuracy_score(y, p, k=5, labels=range(n))) if n > 5 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
        "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(fw),
        "cohen_kappa": float(cohen_kappa_score(y, pred)), "mcc": float(matthews_corrcoef(y, pred)), "ece_15bins": ece(conf, correct),
        "accept_threshold": t,
        "precision_at_threshold": float(correct[keep].mean()) if keep.any() else None,
        "coverage_at_threshold": float(keep.mean()),
        "latency_ms_p50": float(np.percentile(latency_ms, 50)), "latency_ms_p95": float(np.percentile(latency_ms, 95)),
        "per_class": {labels[i]: {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(fc[i]), "support": int(sc[i])} for i in range(n)},
        "worst_confusions": [[labels[i], labels[j], c] for c, i, j in sorted(off, reverse=True)[:10]],
    }
    if per_source:
        out["per_source"] = per_source
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{task}.json").write_text(json.dumps(out, indent=1))
    plots(task, labels, y, pred, fc)
    print(f"{task}: accuracy {out['accuracy']:.3f}  macro-F1 {out['f1_macro']:.3f}  top3 {out['top3']}  threshold {t}  "
          f"precision@threshold {out['precision_at_threshold']}  coverage {out['coverage_at_threshold']:.2f}")
    return out


def eval_words():
    d = np.load(PROCESSED / "words.npz")
    X, y, split, signer = d["X"].astype(np.float32), d["y"].astype(np.int64), d["split"], d["participant"]
    model = WordPredictor()
    feats = {s: np.stack([sequence_features(x) for x in X[split == s]]) for s in ("val", "test")}
    sets = {s: (y[split == s], model.proba(feats[s])) for s in feats}
    test_x = X[split == "test"]
    times = []
    for x in test_x[np.random.default_rng(0).choice(len(test_x), 200)]:
        t0 = time.perf_counter()
        model.proba(sequence_features(x)[None])
        times.append((time.perf_counter() - t0) * 1000)
    info = {f"{s}_signers": sorted(int(i) for i in set(signer[split == s])) for s in ("train", "val", "test")}
    info.update({f"n_{s}": int((split == s).sum()) for s in ("train", "val", "test")})
    note = ("Test signers never appear in training or validation. Threshold chosen on validation signers. "
            "Source: ASL Citizen (webcam clips), landmarks extracted with MediaPipe.")
    return report("words", f"WordNet x{len(model.models)}", model.labels, sets["val"], sets["test"], info, times, note)


def latency(fn, samples):
    times = []
    for x in samples:
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def eval_static():
    from scripts.train_static import LABELS, load_static, mask_groups
    d = load_static()
    art = joblib.load(ARTIFACTS / "static.joblib")
    model = art["model"]
    sets = {}
    for s in ("val", "test"):
        m = d["split"] == s
        sets[s] = (d["y"][m], mask_groups(model.predict_proba(d["X"][m]), d["y"][m]))
    test = d["split"] == "test"
    pred = sets["test"][1].argmax(1)
    per_source = {src: {"n": int((d["source"][test] == src).sum()),
                        "accuracy": float((pred == sets["test"][0])[d["source"][test] == src].mean()),
                        "f1_macro": float(f1_score(sets["test"][0][d["source"][test] == src], pred[d["source"][test] == src], average="macro"))}
                  for src in sorted(set(d["source"][test]))}
    rows = np.flatnonzero(test)[np.random.default_rng(0).choice(int(test.sum()), 200)]
    times = latency(lambda i: model.predict_proba(hand_features(d["hands"][i], d["aspect"][i])[None]), rows)
    info = {f"n_{s}": int((d["split"] == s).sum()) for s in ("train", "val", "test")}
    info["sources"] = {src: int((d["source"] == src).sum()) for src in sorted(set(d["source"]))}
    note = ("Letters and image digits come from datasets without signer IDs and are split at random per image, so those "
            "scores are optimistic: the same signer can appear in train and test. Digit clips from ASL Citizen are split by signer. "
            "Letters and digits are scored with the other group masked, as in the app's ABC and 123 modes.")
    return report("static", f"{art['kind']} (hand landmarks)", LABELS, sets["val"], sets["test"], info, times, note, per_source)


def eval_emotion():
    from scripts.train_emotion import load_emotion
    bs, y, split, _ = load_emotion()
    art = joblib.load(ARTIFACTS / "emotion.joblib")
    sets = {s: (y[split == s], art["model"].predict_proba(bs[split == s])) for s in ("val", "test")}
    times = latency(lambda i: art["model"].predict_proba(bs[i][None]), np.random.default_rng(0).choice(len(bs), 200))
    info = {f"n_{s}": int((split == s).sum()) for s in ("train", "val", "test")}
    note = ("FER2013 official protocol: train, publicTest as validation, privateTest as test. Images are 48x48 grey and labels are "
            "noisy (human agreement is about 65%), so scores are lower than on high-resolution sets. Disgust and fear are not used.")
    return report("emotion", f"{art['kind']} (52 face blendshapes)", EMOTIONS, sets["val"], sets["test"], info, times, note)


TASKS = {"words": eval_words, "static": eval_static, "emotion": eval_emotion}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=TASKS)
    TASKS[ap.parse_args().task]()
