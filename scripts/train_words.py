"""Train the word model on the ASL Citizen train signers, early-stopping on the val signers. Test signers are never touched.

Run: .venv/bin/python -m scripts.train_words [--seeds 1] [--d-model 192] [--layers 2] [--epochs 80] [--mixup 0]
Writes artifacts/words.pt (state_dicts is a list: one entry per seed, averaged at inference).
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from signbridge.config import ARTIFACTS, LOGS, PROCESSED, SEED, T_MODEL, WORD_FEATURES
from signbridge.landmarks import augment_sequence, sequence_features
from signbridge.models import WordNet

BATCH, PATIENCE, WARMUP = 128, 15, 5


def load():
    d = np.load(PROCESSED / "words.npz")
    X, y, split = d["X"].astype(np.float32), d["y"].astype(np.int64), d["split"]
    return {s: (X[split == s], y[split == s]) for s in ("train", "val", "test")}


def features(X):
    return np.stack([sequence_features(x) for x in X])


def train_one(seed, data, n_classes, args, log):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = WordNet(n_classes, d=args.d_model, layers=args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps_per_epoch = -(-len(data["train"][0]) // BATCH)
    total, warm = args.epochs * steps_per_epoch, WARMUP * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              max(1e-2, 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1)))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xv, yv = features(data["val"][0]), data["val"][1]
    Xt, yt = data["train"]
    best, best_state, stale = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(Xt))
        losses = []
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            xb = torch.from_numpy(np.stack([sequence_features(augment_sequence(Xt[j], rng)) for j in idx]))
            yb = torch.from_numpy(yt[idx])
            if args.mixup:
                lam = float(rng.beta(args.mixup, args.mixup))
                perm = torch.randperm(len(idx))
                logits = model(lam * xb + (1 - lam) * xb[perm])
                loss = lam * loss_fn(logits, yb) + (1 - lam) * loss_fn(logits, yb[perm])
            else:
                loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            pred = model(torch.from_numpy(Xv)).argmax(1).numpy()
        f1, acc = f1_score(yv, pred, average="macro"), accuracy_score(yv, pred)
        log(f"seed {seed} epoch {epoch + 1} loss {np.mean(losses):.3f} val_acc {acc:.3f} val_f1 {f1:.3f} {time.time() - t0:.1f}s")
        if f1 > best:
            best, stale = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best_state, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--mixup", type=float, default=0.0)
    args = ap.parse_args()
    torch.set_num_threads(24)
    LOGS.mkdir(exist_ok=True)
    logf = open(LOGS / "train_words.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    data = load()
    labels = json.loads((ARTIFACTS / "vocab_words.json").read_text())["labels"]
    log(f"args {vars(args)}  train {len(data['train'][0])} val {len(data['val'][0])} test {len(data['test'][0])} clips")
    states, scores = [], []
    for k in range(args.seeds):
        state, best = train_one(SEED + k, data, len(labels), args, log)
        states.append(state)
        scores.append(best)
        log(f"seed {SEED + k} best val_f1 {best:.3f}")
    torch.save({"state_dicts": states, "labels": labels,
                "config": {"T_MODEL": T_MODEL, "WORD_FEATURES": WORD_FEATURES, "d_model": args.d_model, "layers": args.layers}},
               ARTIFACTS / "words.pt")
    log(f"saved artifacts/words.pt  val_f1 per seed {np.round(scores, 3).tolist()}")


if __name__ == "__main__":
    main()
