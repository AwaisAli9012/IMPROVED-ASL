"""
AUTO-SCAN DIAGNOSTIC
====================
Runs the REAL live code (app.py: feature extraction, scaler, ensemble, group masking, cutoff)
on dataset images, so we can tell a model/feature problem apart from a camera/UI problem.

Usage (from the v3 folder):   python diagnose.py
Optional:                     python diagnose.py 40      # images per class (default 25)
"""
import sys
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import app as live                      # the real live code (does not start the server or camera)
import Config
import mediapipe.python.solutions.hands as mp_hands

N_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
random.seed(0)

# static-image MediaPipe, same settings as extract_multimodal.py
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                       model_complexity=1, min_detection_confidence=0.3)

print("\nConfig file:", Config.__file__)
print("Classes in model:", len(live.idx_to_class))
print("Models loaded:   ", sorted(live.candidate_model.keys()))
print("Scaler loaded:   ", live.scaler is not None)


def sample_images(root: Path, classes, n):
    out = []
    for c in classes:
        folder = root / c
        if not folder.exists():
            continue
        files = [p for ext in ("*.jpg", "*.jpeg", "*.png") for p in folder.glob(ext)]
        random.shuffle(files)
        out += [(c, f) for f in files[:n]]
    return out


def run(samples, group_id, flip=False):
    ok = wrong = rejected = nohand = 0
    confs, confusions = [], Counter()
    for label, path in samples:
        img = cv2.imread(str(path))
        if img is None:
            continue
        if flip:
            img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not (res and res.multi_hand_landmarks):
            nohand += 1
            continue
        feats = live.extract_frame_features(res, w, h)
        idx, conf = live.predict_ensemble(feats, group_id=group_id)
        if idx is None:
            rejected += 1
            continue
        pred = live.idx_to_class.get(idx)
        confs.append(conf)
        if pred == label:
            ok += 1
        else:
            wrong += 1
            confusions[(label, pred)] += 1
    seen = ok + wrong + rejected
    return dict(ok=ok, wrong=wrong, rejected=rejected, nohand=nohand, seen=seen,
                mean_conf=float(np.mean(confs)) if confs else 0.0, confusions=confusions)


def report(title, r):
    seen = max(r["seen"], 1)
    print(f"\n--- {title} ---")
    print(f"hand found in {r['seen']} images ({r['nohand']} had none)")
    print(f"correct  {100*r['ok']/seen:5.1f}%   wrong {100*r['wrong']/seen:5.1f}%   "
          f"rejected by cutoff {100*r['rejected']/seen:5.1f}%   mean confidence {r['mean_conf']:.2f}")
    if r["confusions"]:
        print("most common mistakes:", ", ".join(f"{a}->{b} x{n}" for (a, b), n in r["confusions"].most_common(5)))


letters = sample_images(Config.ALPHABETS_DIR, Config.ALL_ALPHABET_NAMES, N_PER_CLASS)
words = sample_images(Config.SIGNS_FRAMES_DIR, Config.ALL_SIGN_NAMES, N_PER_CLASS)
print(f"\nSampled {len(letters)} letter images and {len(words)} word images.")

report("LETTERS  | Alphabet Auto-Scan (ALPHA_ALL) | unflipped, like training", run(letters, "ALPHA_ALL"))
report("LETTERS  | Alphabet Auto-Scan (ALPHA_ALL) | MIRRORED", run(letters, "ALPHA_ALL", flip=True))
a1 = [(c, f) for c, f in letters if c in Config.GROUPS["ALPHA1"]["classes"]]
report("LETTERS A-E | sub-group ALPHA1 (for comparison)", run(a1, "ALPHA1"))
report("WORDS    | Word Auto-Scan (SIGN_ALL) | unflipped", run(words, "SIGN_ALL"))
report("WORDS    | Word Auto-Scan (SIGN_ALL) | MIRRORED", run(words, "SIGN_ALL", flip=True))

print("""
HOW TO READ THIS
- LETTERS auto-scan correct >= ~85% unflipped : model + features + masking are fine. If the live page still
  shows WAITING, the cause is the live camera path (lighting, hand too small, FLIP_FIRST setting), not the model.
- LETTERS auto-scan low but ALPHA1 high      : sub-groups look good only because 5-class renormalising hides
  weak predictions; the live features do not match training.
- MIRRORED much lower than unflipped         : orientation matters, so keep FLIP_FIRST=0 (default).
- 'rejected by cutoff' large                 : probabilities are low; paste this output to me.
""")