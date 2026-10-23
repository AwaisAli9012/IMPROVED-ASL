# How SignBridge is evaluated

Every number the app or the README shows is read from `reports/*.json`, which `scripts/evaluate.py` writes. Nothing is typed in by hand. This page explains what those numbers mean and what they do not prove.

## Protocol

| Task | Data | Split | Why it is a fair test |
|---|---|---|---|
| Words | ASL Citizen clips (webcam, 52 signers) | The dataset's official split by signer: train, validation and test signers never overlap | The model is tested on people it has never seen, which is what a new user is |
| Letters | Public alphabet images | Random split per image | The source has no signer IDs, so this score is **optimistic** and the report says so |
| Digits | Frames from ASL Citizen number clips | Official split by signer | Signer-independent |
| Tone from the face | FER2013 | The standard protocol: train, publicTest for validation, privateTest for testing | Standard benchmark, but 48x48 greyscale with noisy labels |

Rules that keep the numbers honest:

1. Test data is never used for training, early stopping or choosing a threshold. Early stopping and the acceptance threshold use the validation signers only.
2. Training augmentation (small rotation, scaling, shifts, time cropping, noise, dropping a hand) is applied to training clips only.
3. The same functions turn landmarks into features in training and in the live app (`signbridge/landmarks.py`), and the same MediaPipe models produce the landmarks in both (hand, pose lite, face).
4. Hand-slot assignment (which detected hand is the left or right) is one rule, implemented in Python for training and ported to JavaScript for the browser, with a test on each side.

## Metrics reported

Accuracy, top-3 and top-5 accuracy, macro and weighted precision, recall and F1, balanced accuracy, Cohen's kappa, Matthews correlation, expected calibration error (15 bins), per-class precision, recall and F1, the ten most confused pairs, and median and 95th-percentile answer time. Macro-F1 is the headline figure because every sign matters equally, however many clips it has.

**Acceptance threshold.** The app answers only when the model's confidence is above a threshold chosen on the validation signers (the lowest confidence that reaches 95% precision, or the most precise one if none does). The report then measures precision and coverage of that rule on the test signers, so the cost of declining to guess is visible.

## What these numbers do not show

- **Live accuracy on you.** Held-out accuracy is measured on recorded webcam clips. Your camera, lighting, distance and signing style will differ. The Model Inspector in Converse shows the live top-5 so you can see how the model behaves on you.
- **Continuous signing.** The app recognises one sign at a time, with hands lowered between signs. It is not a continuous sign recognizer.
- **Fairness across people.** We have not audited accuracy by skin tone, hand size, age or disability. ASL Citizen volunteers are not a random sample of signers.
- **Grammar.** Eyebrow-based question detection and facial tone are heuristics on top of landmark data, not validated against Deaf signers' judgements.
- **Letters.** See above: random split, not signer-independent.

## Reproducing the results

```bash
.venv/bin/python -m scripts.download_data
.venv/bin/python -m scripts.build_words_dataset
.venv/bin/python -m scripts.extract_images --task static
.venv/bin/python -m scripts.extract_images --task emotion
.venv/bin/python -m scripts.train_words --seeds 3
.venv/bin/python -m scripts.train_static
.venv/bin/python -m scripts.train_emotion
.venv/bin/python -m scripts.evaluate words
.venv/bin/python -m scripts.evaluate static
.venv/bin/python -m scripts.evaluate emotion
```

Training seeds are fixed (`SEED = 42`, seeds 42 to 44 for the ensemble). Running the commands again gives the same reports up to floating-point differences in threading.

## Software tests

`pytest -q` runs the Python tests (features and invariances, evaluation helpers, the sentence builder, the learning engine, the API contract and its input validation) and `ruff` lints the code. In `web/app`, `svelte-check` type-checks the front end and `vitest` tests the browser logic (hand-slot rule, sign segmentation, letter latching). A browser suite (`node e2e/run.mjs`) loads every page with no console errors, runs the intro lesson, a locked-unit redirect, the dictionary deep link, a typed reply and the command palette, runs axe-core on 14 routes (Converse, Learn, the intro lesson, a lesson, review, Progress, Phrases, Captions, Dictionary, Metrics, Settings, About, Certificate and the landing page) in the dark, light and high-contrast themes with no violations, checks that nothing scrolls sideways at 390 px, measures camera speed with a fake camera, and checks reduced motion. Individual quiz steps were not scanned.
