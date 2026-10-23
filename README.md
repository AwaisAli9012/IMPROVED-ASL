# SignBridge

Two-way sign language on one screen. A signer signs to the camera and gets an English sentence and a spoken voice. The other person types or speaks, and a signing figure answers. A full course teaches the language, from setting up the camera to signing whole sentences. Everything runs on your own computer, on a CPU.

## What is in it

- **Converse.** Sign words, letters or numbers. SignBridge builds the sentence, adds the right tone and question form, and speaks it. Type or speak a reply and the signing figure signs it back. The conversation can be saved as text. A **Model Inspector** (press `I`) shows the live top-5 guesses, answer time and camera frame rate.
- **Learn.** A course of about 34 units: setting up, the alphabet, numbers, topic after topic of everyday words, and whole sentences. Each sign is watched, tried on camera, and checked, then quizzed. Spaced repetition brings back signs you are losing, with XP, a daily streak and a printable certificate.
- **Progress.** A heatmap of every sign, the last 14 days, your weak spots, and ten badges.
- **Phrases.** Tap a phrase to see it signed and hear it spoken: greetings, needs, questions, emergencies, and your own.
- **Live captions.** Speech turned into large text as people talk, optionally signed back sentence by sentence.
- **Dictionary.** Every supported sign, searchable by name or topic, played by the signing figure at half or full speed.
- **Metrics.** Every accuracy figure with the confusion matrices and per-class scores, read from the saved evaluation reports.
- **Tone and grammar from the face.** Expression sets the tone of the spoken sentence. Raised eyebrows mark a yes or no question and lowered eyebrows mark what, where and who, as they do in ASL.
- **Settings.** Dark, light and high-contrast themes, four text sizes, reduce motion, voice and speaking speed, camera choice, signing defaults, and your data: download it or delete it.
- **About.** The pipeline, what it knows, privacy, ethics and fairness, accessibility, limits and credits.

Vocabulary: see the Dictionary for the full list. Anything outside it is fingerspelled. The alphabet A to Z and the numbers 0 to 4 and 6 to 9 are supported (there is no number 5 yet: see Known limits).

**Quality checks.** Python tests (pytest, ruff), TypeScript checks (svelte-check) and unit tests (vitest), plus a browser suite (`web/app/e2e`) that loads every page, runs the main flows, and runs axe-core in the dark, light and high-contrast themes with no violations, checks phone width and camera speed; the landing page passes the scroll-craft checks for dead scroll and text contrast at desktop, phone and reduced-motion sizes. How the models are tested is in [docs/EVALUATION.md](docs/EVALUATION.md).

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m scripts.fetch_assets      # MediaPipe models and browser libraries, ~40 MB
cp .env.example .env                          # optional: add GEMINI_API_KEY for richer sentences
.venv/bin/python -m signbridge                # open http://127.0.0.1:5000 in Chrome or Edge
```

The trained models, the evaluation reports and the built web app are in the repository, so nothing needs training or building to run it. Interactive API docs are at `/api/docs`. Without a Gemini key, sentences are built by simple offline rules. Speech input needs Chrome or Edge.

**The signing figure needs one more step.** Its animations are made from real signers' recordings, which the dataset license does not let us republish, so they are not in the repository. Build them once on your own machine (this also needs the data pipeline below):

```bash
.venv/bin/python -m scripts.export_references
```

## Develop the web app and share it

The front end is SvelteKit (Svelte 5, static build) in `web/app`. It needs Node 20.

```bash
cd web/app && npm ci
npm run dev        # http://localhost:5173, proxies /api to the Python server on :5000
npm run check && npm test && npm run build   # the build in web/app/build is committed, and the Python server serves it
node e2e/run.mjs   # browser checks against the running server (uses the system Chrome)
```

To show it to someone else over the internet, run `scripts/tunnel.sh` while the server is up. It starts a Cloudflare quick tunnel and prints a temporary https address (the camera needs https). It needs outbound port 7844, which some networks block: try a phone hotspot.

## Rebuild everything from public data

No accounts or API keys are needed. About two hours on a 28-core CPU, plus downloads.

```bash
.venv/bin/python -m scripts.download_data
.venv/bin/python -m scripts.build_words_dataset          # streams only the ~3,400 clips it needs
.venv/bin/python -m scripts.extract_images --task emotion
.venv/bin/python -m scripts.extract_images --task static
.venv/bin/python -m scripts.train_words && .venv/bin/python -m scripts.evaluate words
.venv/bin/python -m scripts.train_static && .venv/bin/python -m scripts.evaluate static
.venv/bin/python -m scripts.train_emotion && .venv/bin/python -m scripts.evaluate emotion
.venv/bin/python -m scripts.export_references
.venv/bin/python -m scripts.update_readme                # refreshes the table below
```

## Results

Measured on data the models never trained on. The word model is tested on signers who appear in neither training nor validation.

<!-- metrics:start -->
| Task | Test set | Accuracy | Macro F1 | Precision | Recall | Top 3 | Answers in (p95) |
|---|---|---|---|---|---|---|---|
| Words | 3476 clips, 11 unseen signers | 86.1% | 86.2% | 87.7% | 86.1% | 94.7% | 3.4 ms |
| Letters and numbers | 2391 samples | 97.2% | 96.7% | 96.8% | 96.7% | 99.0% | 0.3 ms |
| Tone from the face | 2704 samples | 69.9% | 68.2% | 68.2% | 68.2% | 95.6% | 1.3 ms |
<!-- metrics:end -->

Read the notes on each task in the Metrics page before quoting a number. In particular, letters come from a dataset with no signer information and are split at random per image, so that score is optimistic. Emotion is measured on FER2013, which is low resolution and has noisy labels.

## How it works

```
Browser (SvelteKit app) camera -> MediaPipe in the browser (hands, pose, face)
   words:    segment a sign (hands up to hands down) -> POST /api/predict/word   -> sequence model
   ABC/123:  dominant hand, every frame              -> POST /api/predict/static -> landmark classifier
   every 400 ms: 52 face blendshapes                 -> POST /api/emotion        -> tone
   glosses + tone + question form -> POST /api/sentence -> Gemini or offline rules -> English -> browser voice
Reply: typed or spoken English -> POST /api/gloss -> sign tokens -> signing figure
```

Hand, body and face detection runs on the CPU in the browser at about 20 frames per second (measured in Chrome at 640x360). Add `?gpu=1` to the address to try the GPU on a machine that has one: on a machine without a real GPU it silently falls back to slow software rendering, which is why CPU is the default.

Only landmark numbers leave the browser, and only to the FastAPI server running on the same computer. The API keeps no state, so each request carries everything it needs. Constants live in `signbridge/config.py` and the browser reads them from `/api/config`. Feature code lives in `signbridge/landmarks.py` and is shared by training and live use.

## A three-minute demo

1. Open `/`. Type the evaluator's name: the figure fingerspells it. Scroll to step through the letters.
2. Keep scrolling: the method, the measured numbers, then the scripted conversation (it is labelled as a sample).
3. Open Converse. Sign "hello", then "how" and "you" with raised eyebrows. Press Enter: the sentence is spoken as a question.
4. Let the evaluator type or say a reply. The figure signs it back.
5. Open Learn, start the alphabet, and show the camera check on one letter.
6. Open Metrics and explain why the test signers were held out.

## Data and licences

| Data | Used for | Licence |
|---|---|---|
| ASL Citizen (Microsoft, 83,399 clips, 52 signers) | Words and digits | Microsoft Research License: non-commercial research only. The data and anything derived from the signers' recordings is not redistributed here. Raw videos are streamed, turned into landmarks and deleted. |
| ASL alphabet images (Hugging Face `Marxulia/asl_sign_languages_alphabets_v03`) | Letters | See the dataset page |
| FER2013 (Hugging Face `Aaryan333/fer2013_train_publicTest_privateTest`) | Tone | See the dataset page |
| MediaPipe models | Hands, pose, face | Apache 2.0 |

Trained model weights and the evaluation reports are included. Whether weights trained on ASL Citizen may be shared is a judgement call under that licence: check it before publishing this repository.

## Known limits

- Number 5 is missing: no open dataset has a signed 5. It needs real samples, for example recorded in the app.
- One signer in view, with pauses between signs: lower your hands after each sign. This is taught in the first lesson.
- 280 words is a broad everyday vocabulary, not the whole language. Fingerspelling covers the rest. Accuracy falls as the vocabulary grows: the 100-word core scored 89.7% on the same unseen signers, the 280-word model scores 86.1%.
- Tone is a hint for wording and voice, not a diagnosis of anyone's feelings.
- Live accuracy on a given person and camera differs from the held-out figures.
