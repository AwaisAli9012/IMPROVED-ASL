# SignBridge: Project Guide

**Read this first.** This file explains the whole project in plain language: what it is, how it is built, how to run it, how to share it, what works today, and how to answer an evaluator's questions. It is written so that a person, or an AI assistant such as ChatGPT or Qwen, can read it and understand the project without any other context.

> Where this guide says **[FILL IN]**, only the team knows the answer (who did what, personal experience). Do not leave those blank. Write the true answer before the evaluation.

---

## Contents

1. [What SignBridge is](#1-what-signbridge-is)
2. [Current status](#2-current-status)
3. [Technology stack and why](#3-technology-stack-and-why)
4. [How it works, step by step](#4-how-it-works-step-by-step)
5. [Folder map](#5-folder-map)
6. [First-time setup](#6-first-time-setup)
7. [Run the local server and stop it](#7-run-the-local-server-and-stop-it)
8. [Share it on the internet with Cloudflare](#8-share-it-on-the-internet-with-cloudflare)
9. [Optional Gemini key](#9-optional-gemini-key)
10. [Check that everything works (tests)](#10-check-that-everything-works-tests)
11. [Retrain the models](#11-retrain-the-models)
12. [Models, data and accuracy](#12-models-data-and-accuracy)
13. [Known limits](#13-known-limits)
14. [Rules for anyone changing the code](#14-rules-for-anyone-changing-the-code)
15. [Troubleshooting](#15-troubleshooting)
16. [Evaluator questions and answers](#16-evaluator-questions-and-answers)
17. [Demo script (three minutes)](#17-demo-script-three-minutes)
18. [Glossary](#18-glossary)

---

## 1. What SignBridge is

SignBridge is a web app for **two-way communication between a person who signs (American Sign Language, ASL) and a person who does not**, on one computer.

- **Sign to text and voice.** The signer signs to the webcam. The app recognises words, letters and numbers, builds an English sentence, and speaks it.
- **Text or voice to signing.** The other person types or speaks. A signing figure on screen signs the reply.
- **Learn.** A course of 34 units teaches the language, from setting up the camera to signing whole sentences, with quizzes, spaced repetition, points, a daily streak and a printable certificate.
- **Extras.** Dictionary of every supported sign, phrasebook, live captions for speech, progress page with badges, a metrics page with every accuracy number, settings (themes, text size, reduced motion, voice, camera).

It runs on **one ordinary computer with no graphics card**. The video never leaves the browser tab. Only small lists of numbers (points on the hands, body and lips) go to a server running on the same computer.

**Target users:** Deaf and hard-of-hearing people, their families and colleagues, and people learning ASL.

---

## 2. Current status

Last updated: 23 October 2026. Branch: `v2-sveltekit-fastapi`.

### What works and was checked

| Area | Status | How it was checked |
|---|---|---|
| Backend (FastAPI) | Working | 70 automated tests pass; linter clean |
| Front end (SvelteKit) | Working | Type check: 0 errors, 0 warnings; 4 unit tests pass; production build succeeds |
| All 14 pages | Load with no console errors | Browser test suite (`web/app/e2e/run.mjs`) |
| Accessibility | 0 violations | Automated axe-core scan on all pages in dark, light and high-contrast themes |
| Phone width (390 px) | No sideways scrolling | Browser test suite |
| Camera speed | About 17 to 20 frames per second | Measured with a fake camera on this laptop's CPU |
| Word model on unseen signers | 86.1% accuracy | Saved report `reports/words.json` |
| Server round trip | About 14 ms per sign | `scripts/check_api.py` replay of 300 test clips: 85.3% top-1, 95.0% top-3 |
| Landing page scrolling | No dead scroll, text readable at all sizes | Scroll-craft check at desktop, phone and reduced-motion sizes |

### What is NOT verified yet (be honest about these)

- **Real webcam signing by a real person.** Everything was tested with a fake video feed. Test with your own webcam before the evaluation and note the real accuracy you see.
- **A real phone.** Tested with a phone-sized browser window only.
- **The public Cloudflare link.** The tunnel starts and prints an address, but the connection was blocked on the network used (port 7844). It may work on another network, such as a phone hotspot.
- **Number 5.** No public dataset has a signed 5, so it is not supported yet.

### Code location

The work is committed on the local branch `v2-sveltekit-fastapi`. Push it to GitHub with: `git push fork v2-sveltekit-fastapi` (needs your GitHub login).

---

## 3. Technology stack and why

| Layer | Technology | Why we chose it |
|---|---|---|
| Hand, body and face detection | **MediaPipe** (Google) running **in the browser** | Free, accurate, runs on a normal CPU, and keeps the video private because nothing is uploaded |
| Word model | **PyTorch** neural network (small 1D-convolution plus Transformer) | Signs are movements over time, so the model must read a sequence of frames, not one picture |
| Letter, number and emotion models | **scikit-learn** (small neural network) and **XGBoost** | Fast, accurate on small fixed-size inputs, and tiny to store |
| Backend | **Python 3.12 with FastAPI** and the **uvicorn** server | Fast, typed, automatic API documentation at `/api/docs`, and the same language as the model code |
| Front end | **SvelteKit 2 with Svelte 5 and TypeScript**, built as a static site with **Vite** | Small and fast pages, smooth transitions, and a type checker that catches mistakes before they reach users |
| Database | **SQLite** (a single file) | No server to install, perfect for one computer, stores profiles and learning progress |
| Sentences | Offline rules, optionally **Google Gemini** | Works with no internet or key; Gemini makes sentences read more naturally when a key is added |
| Sharing | **Cloudflare Tunnel** (`cloudflared`) | Gives a public https address without opening router ports; https is required for camera access on other devices |
| Testing | pytest, ruff, svelte-check, vitest, Playwright, axe-core | Backend, types, logic, real browser and accessibility checks |

**Why one computer and a CPU only:** it keeps the project cheap, private, and easy to demonstrate anywhere without special hardware.

---

## 4. How it works, step by step

```
Webcam
  -> MediaPipe in the browser finds 21 points on each hand, 7 body points and 40 lip points,
     plus 52 face-muscle scores
  -> The browser app (SvelteKit) groups frames into one sign (hands up -> hands down)
  -> POST /api/predict/word     -> word model        -> top 5 guesses
     POST /api/predict/static   -> letter/number model -> top 3 guesses
     POST /api/emotion          -> face model          -> tone (happy, sad, ...)
  -> POST /api/sentence -> Gemini or offline rules -> English sentence
  -> The browser speaks the sentence

Reply (typed or spoken English)
  -> POST /api/gloss -> list of signs to show (words in the vocabulary, other words spelled letter by letter)
  -> The signing figure plays recorded movement from real signers
```

Key design rules:

- **The server keeps no memory between requests** (except the SQLite file for learning progress). Each request carries everything it needs, which makes it easy to test.
- **All settings live in one file**, `signbridge/config.py`. The browser reads them from `GET /api/config`, so a number is never copied in two places.
- **The feature math lives in one file**, `signbridge/landmarks.py`, and is used by both training and live use, so the model sees the same kind of numbers in both.
- **Face grammar:** raised eyebrows mark a yes/no question, lowered eyebrows mark a what/where/who question, as in real ASL. The app compares your face with a neutral face captured at set-up.

### API summary (all JSON, errors look like `{"error": "message"}`)

| Method and path | Purpose |
|---|---|
| `GET /api/config` | Settings and the word list for the browser |
| `GET /api/health` | Shows whether models are loaded |
| `POST /api/predict/word` | Sign recognition from up to 120 frames |
| `POST /api/predict/static` | Letter or number from one hand pose |
| `POST /api/emotion` | Tone from 52 face scores |
| `POST /api/sentence` | English sentence from signs, tone and question type |
| `POST /api/gloss` | Signs to show for an English sentence |
| `GET/POST/DELETE /api/profiles` | Learner profiles |
| `GET /api/learn/{id}`, `/review`, `POST /api/learn/attempt`, `/unit_pass` | The course and spaced repetition |
| `GET /api/progress/{id}` | Progress, heatmap and badges |
| `GET /api/reports`, `/api/curriculum`, `/api/phrases` | Metrics, course content, phrasebook |

Interactive documentation of every endpoint: open **http://127.0.0.1:5000/api/docs** while the server runs.

---

## 5. Folder map

```
Sign_Language/
├── Guide.md                 This file
├── README.md                Short overview and results table
├── requirements.txt         Python packages
├── pyproject.toml           Project info, linter and test settings
├── .env.example             Template for private settings (copy to .env)
├── signbridge/              The Python backend
│   ├── app.py               FastAPI app: middleware, errors, serves the web app
│   ├── routes/              API endpoints (meta, recognition, language, learning)
│   ├── services.py          Model loading, input checks, predictions
│   ├── schemas.py           Request body shapes
│   ├── config.py            Every setting and path
│   ├── landmarks.py         Feature math shared by training and live use
│   ├── models.py            The word model and loaders
│   ├── nlg.py               Sentence building (offline rules and Gemini)
│   ├── learn.py             Course progress in SQLite
│   ├── curriculum.json      The 34-unit course
│   └── phrases.json         Phrasebook
├── scripts/                 Download data, train, evaluate, export, check, tunnel
├── web/app/                 The SvelteKit front end
│   ├── src/routes/          Pages (landing, converse, learn, dictionary, ...)
│   ├── src/lib/             Shared code (camera, recogniser, signing figure, components)
│   ├── e2e/run.mjs          Browser test suite
│   └── build/               Built site served by Python (committed)
├── web/static/              Fonts, MediaPipe files, signing animations (some are generated locally)
├── artifacts/               Trained models (committed)
├── reports/                 Accuracy reports and charts (committed)
├── data/                    Downloaded datasets and the SQLite file (never committed)
├── tests/                   Automated Python tests
└── docs/                    DECISIONS.md (choices made) and EVALUATION.md (how accuracy is measured)
```

---

## 6. First-time setup

You need: **Linux, macOS or Windows (WSL)**, **Python 3.12**, **Node.js 20** (only if you change the front end), and **Google Chrome or Edge**.

Open a terminal in the project folder, then run these one at a time:

```bash
# 1. Create a private Python environment
python3 -m venv .venv

# 2. Install the packages (the extra index gives the CPU build of PyTorch)
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Download the MediaPipe models and files the browser needs (about 40 MB)
.venv/bin/python -m scripts.fetch_assets

# 4. Optional: create your private settings file
cp .env.example .env
```

The trained models, accuracy reports and the built web app are already in the repository, so **no training or building is needed to run it**.

**The signing figure needs one extra step.** Its animations are made from real signers' recordings. The dataset licence does not allow republishing them, so they are not in the repository. Generate them once on your machine (needs the data from section 11):

```bash
.venv/bin/python -m scripts.export_references
```

If you skip this, the app still works, but the signing figure may show nothing for some words.

---

## 7. Run the local server and stop it

### Start

From the project folder:

```bash
.venv/bin/python -m signbridge
```

You will see a line like `Uvicorn running on http://127.0.0.1:5000`. **Leave this terminal open**: the server runs as long as it is open.

Then open **http://127.0.0.1:5000** in Chrome or Edge.

- The first time, the browser asks for camera access. Click **Allow**.
- Use `127.0.0.1` or `localhost`. Other addresses need https before the browser allows the camera.
- Useful pages: `/` (landing), `/app` (Converse), `/learn`, `/dictionary`, `/metrics`, `/api/docs`.

### Stop

- **Normal way:** click the terminal that is running the server and press **Ctrl + C**.
- **If the terminal is gone and the server is still running**, open a new terminal and run:

```bash
kill $(pgrep -f 'python -m signbridg[e]')
```

(The odd `[e]` is intentional: it stops the command from matching itself.)

### Check it is running

```bash
curl http://127.0.0.1:5000/api/health
```

A good answer starts with `{"status":"ok"`.

### Change the port

Edit `PORT` in `signbridge/config.py`. If you do, also set `PORT=` when starting the tunnel (section 8).

---

## 8. Share it on the internet with Cloudflare

A Cloudflare Tunnel gives your local server a public **https** address, so a phone or another computer can open it and use its camera. Your computer must stay on with the server running.

### Option A: temporary link (no account, no key)

1. Start the server (section 7) in terminal 1.
2. In terminal 2, from the project folder:

```bash
scripts/tunnel.sh
```

3. Look for an address like `https://something-random.trycloudflare.com` in the output. Open it. The address changes every time.
4. Stop sharing with **Ctrl + C** in terminal 2.

### Option B: your own fixed address (uses a Cloudflare key)

**Where the key goes:** in the file **`.env`** in the project folder (not committed to GitHub), on this line:

```
CLOUDFLARE_TUNNEL_TOKEN=paste-your-token-here
```

How to get the token:

1. Create a free account at **cloudflare.com** and open **Zero Trust**.
2. Go to **Networks, then Tunnels**, and click **Create a tunnel** (type: Cloudflared).
3. Name it, for example `signbridge`. Cloudflare shows an install command that contains a long text after `--token`. **Copy only that long text.** That is the token.
4. In the tunnel's **Public Hostname** settings, add a hostname you own, with **Service** type `HTTP` and URL `127.0.0.1:5000`.
5. Paste the token into `.env` as shown above.
6. Run `scripts/tunnel.sh`. When the token is set, the script uses your named tunnel automatically.

**Keep the token secret.** Anyone with it can run a tunnel to your hostname. Never paste it into chat, a document or GitHub. `.env` is already excluded from Git.

### Notes and problems

- `cloudflared` must be installed. The script looks in `~/.local/bin/cloudflared`; set `CLOUDFLARED=/path/to/cloudflared` to use another location.
- The tunnel needs **outbound port 7844**. Some networks (universities, some Wi-Fi) block it, and the link then shows error 530. Try a **phone hotspot**.
- The script uses the `http2` protocol by default. To try the other one: `PROTOCOL=quic scripts/tunnel.sh`.

---

## 9. Optional Gemini key

Without a key, sentences are built by simple built-in rules and everything works. With a Google Gemini key, sentences read more naturally.

1. Create a key in **Google AI Studio**.
2. Put it in `.env`:

```
GEMINI_API_KEY=your-key
```

3. Restart the server. `GET /api/health` then shows `"gemini": true`.

Only the signed words as text (for example "you hungry") are sent to Google. No video or hand points are sent.

---

## 10. Check that everything works (tests)

| What | Command (from the project folder) | Good result |
|---|---|---|
| Python tests | `.venv/bin/pytest -q` | All pass (70 at the time of writing) |
| Python style check | `.venv/bin/ruff check signbridge scripts tests` | "All checks passed" |
| Front end types | `cd web/app && npm run check` | 0 errors, 0 warnings |
| Front end unit tests | `cd web/app && npm test` | All pass |
| Build the front end | `cd web/app && npm run build` | "done" |
| Browser suite | server running, then `cd web/app && node e2e/run.mjs` | "all browser checks passed" |
| Accuracy replay | server running, then `.venv/bin/python -m scripts.check_api` | About 85% top-1 |

Changing the front end: run `cd web/app && npm ci` once, then `npm run dev` (opens on port 5173 and talks to the Python server on 5000). Run `npm run build` before committing, because the built site is what Python serves.

---

## 11. Retrain the models

No accounts or keys are needed; all data is public. It takes about two hours on a 28-core CPU, plus download time. Run from the project folder:

```bash
.venv/bin/python -m scripts.download_data
.venv/bin/python -m scripts.build_words_dataset          # streams only the clips it needs
.venv/bin/python -m scripts.extract_images --task emotion
.venv/bin/python -m scripts.extract_images --task static
.venv/bin/python -m scripts.train_words && .venv/bin/python -m scripts.evaluate words
.venv/bin/python -m scripts.train_static && .venv/bin/python -m scripts.evaluate static
.venv/bin/python -m scripts.train_emotion && .venv/bin/python -m scripts.evaluate emotion
.venv/bin/python -m scripts.export_references
.venv/bin/python -m scripts.update_readme                # refreshes the results table
```

Each `evaluate` step writes a report into `reports/`. **Every number shown in the app comes from those reports**, never typed by hand.

---

## 12. Models, data and accuracy

### The three models

| Model | Input | Output | Type |
|---|---|---|---|
| **Words** | A whole sign (32 frames of 400 numbers) | One of 280 words | 3 neural networks averaged (an "ensemble"): convolution layers plus a small Transformer |
| **Letters and numbers** | One hand pose (273 numbers) | One of 35 classes (A to Z, 0 to 4, 6 to 9) | Small neural network (MLP) |
| **Tone from the face** | 52 face-muscle scores | happy, sad, angry, surprised, neutral | XGBoost |

### Data (all public)

| Data | Used for | Size |
|---|---|---|
| **ASL Citizen** (Microsoft Research): webcam clips of 52 signers | Words and digits | 280 words, about 8,600 clips used |
| **ASL alphabet images** (Hugging Face) | Letters | About 10,900 images |
| **FER2013** (Hugging Face): 48x48 grey face images | Tone | About 27,000 images |

**Licence note:** ASL Citizen is licensed for **non-commercial research only**, and the data (and animations made from it) must not be redistributed. That is why `data/` and the signing animations are never committed. Videos are streamed, turned into landmark numbers, and deleted.

### Current accuracy (from `reports/*.json`)

| Task | Test set | Accuracy | Macro F1 | Precision | Recall | Top 3 |
|---|---|---|---|---|---|---|
| Words | 3,476 clips from **11 signers never seen in training** | **86.1%** | 86.2% | 87.7% | 86.1% | 94.7% |
| Letters and numbers | 2,391 samples | 97.2% | 96.7% | 96.8% | 96.7% | 99.0% |
| Tone from the face | 2,704 images | 69.9% | 68.2% | 68.2% | 68.2% | 95.6% |

- **Precision** = of the signs the app named, how many were right. **Recall** = of the real signs, how many it found. **F1** = one number combining both.
- **Confidence gate:** when the word model is sure enough (probability at least 0.38), it is right **94.0%** of the time and answers on **82.8%** of signs. When unsure it shows a "Did you mean" list instead of guessing.
- **Letters score is optimistic:** the image dataset has no signer information, so the same person can appear in training and testing. The report says so.
- **Tone score is modest on purpose:** FER2013 images are tiny and labels are noisy (people agree with each other only about 65% of the time).

---

## 13. Known limits

State these openly. Evaluators respect honest limits.

1. **Number 5 is not supported.** No open dataset contains a signed 5.
2. **One signer in view**, with a pause (hands down) between signs.
3. **280 words is a broad everyday vocabulary, not the whole language.** Other words are fingerspelled.
4. **Single signs, not full ASL grammar.** The app reads signs one by one and uses eyebrows for question type, but does not model spatial grammar.
5. **Accuracy on a new person and camera will differ** from the held-out figures.
6. **Tone is a hint for wording and voice**, not a judgement of anyone's feelings.
7. **Not a replacement for a qualified interpreter**, and not for medical, legal or emergency use.
8. **Fairness not audited:** accuracy has not been tested across skin tones, lighting and camera types.
9. **Speech input** works only in Chrome and Edge.

---

## 14. Rules for anyone changing the code

- Run commands from the project folder.
- Settings only in `signbridge/config.py`. Feature math only in `signbridge/landmarks.py`.
- Never commit `data/` or `.env` (keys and tokens live there).
- Never show an accuracy number that is not in `reports/*.json`.
- Keep the API paths and JSON shapes unchanged unless the front end and tests change with them.
- Keep MediaPipe at the bundled **0.10.14** files in `web/static/vendor`. Do not install a newer MediaPipe from npm: the models were trained on points from this version.
- Node 20 only: do not upgrade the front end tools past the versions in `web/app/package.json`.
- Run the checks in section 10 before committing.

---

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| "Camera access is blocked" | Click the camera icon in the address bar, choose Allow, reload. Use `127.0.0.1`, not your computer's network address |
| "No camera was found" | Plug in or enable a camera; in Settings, press "Show my cameras" and pick one |
| "frontend not built" message | Run `cd web/app && npm ci && npm run build` |
| The signing figure shows nothing | Generate the animations: `.venv/bin/python -m scripts.export_references` (needs data from section 11) |
| "model not trained yet" | The files in `artifacts/` are missing; run the training steps in section 11 |
| Port 5000 already in use | Stop the old server (section 7) or change `PORT` |
| Camera feels slow | Close other heavy programs. Camera speed is about 17 to 20 frames per second on a CPU |
| Cloudflare link shows error 530 | The tunnel is not connected; the network blocks port 7844. Try a phone hotspot |
| `npm install` is very slow | The package registry can be slow on some networks. Run it in the background and wait |

---

## 16. Evaluator questions and answers

Short, honest answers your team can say aloud. Adapt the wording, but **keep the facts** and **be able to explain every part you mention**. If you do not know something, say so and explain how you would find out. That scores better than guessing.

### Q1. Do you have any future plans for the project?

Yes, in this order:

1. **Add number 5** by recording samples in the app, since no public dataset has it.
2. **Test on many more people, cameras and lighting**, and report accuracy across skin tones.
3. **Grow the vocabulary** beyond 280 words and add **phrases and sentences as units**, not only single signs.
4. **Continuous signing** without pausing between signs.
5. **Learn from real use:** let users flag a wrong answer so the models can be improved (with permission).
6. **Work with Deaf signers and teachers** to review signs and the course.
7. **A phone app / offline install** and a second sign language.

### Q2. What stack did you use?

- **Browser:** MediaPipe (hand, body and face detection) and a **SvelteKit / Svelte 5 / TypeScript** web app.
- **Server:** **Python 3.12 with FastAPI** (uvicorn).
- **Models:** **PyTorch** (words) and **scikit-learn / XGBoost** (letters, numbers, tone).
- **Database:** **SQLite**.
- **Extras:** Google Gemini (optional), Cloudflare Tunnel for sharing.
- **Quality tools:** pytest, ruff, svelte-check, vitest, Playwright, axe-core.

### Q3. Why did you use that stack?

- **MediaPipe in the browser:** free, runs on a normal CPU, and the **video never leaves the user's computer**, which is a real privacy benefit for a communication tool.
- **PyTorch for words:** signs are movements over time, so we needed a model that reads a sequence. A single-frame classifier (our earlier approach) ignores motion.
- **scikit-learn and XGBoost for the small models:** a letter is one hand pose, so a small fast model is enough and gives 97% accuracy.
- **FastAPI:** fast, type-checked inputs, and automatic API documentation, in the same language as the models.
- **SvelteKit:** small fast pages and smooth animation. The built site is plain files that Python serves, so the whole app runs from one command.
- **SQLite:** zero set-up, one file, ideal for one computer.
- **Cloudflare Tunnel:** public https (needed for camera on other devices) without touching router settings.

### Q4. What database did you use?

**SQLite**, a single file at `data/app.db`, created automatically. It has three tables: **profiles** (name, points, daily streak), **progress** (per sign: level 1 to 5, attempts, next review date) and **unit_pass** (quiz scores). A fourth table logs every attempt for the progress page. It uses a **Leitner spaced-repetition** scheme: a correct answer moves a sign up a box and delays its next review; a wrong answer sends it back to box 1. We chose SQLite because the app runs on one computer and needs no separate database server. For many users we would move to PostgreSQL.

### Q5. How did you train the model?

Describe the real process step by step (it is all in `scripts/`, and you can run it live):

1. **Data:** we used the public **ASL Citizen** dataset (webcam clips of 52 volunteer signers), plus a public alphabet image set and FER2013 for faces. All are public and downloaded by `scripts/download_data.py`.
2. **Features:** we ran **MediaPipe** over every clip to turn video into numbers: 21 points per hand, 7 body points and 40 lip points per frame. The raw video is then deleted, so we keep only numbers. This also matches what the browser sends live.
3. **Preparation:** we make the dominant hand always the same side (so left- and right-handed signers look alike), trim idle frames, resample to a fixed length, and normalise for position and body size.
4. **Split by person:** official split by signer: **34 signers to train, 5 to validate, 11 to test**. Test signers are never seen during training, so the score reflects a new user.
5. **Model:** a small network (convolution layers plus a Transformer) trained with **data augmentation** (random scale, rotation, shift, time cropping, noise, hiding one hand), **label smoothing**, the **AdamW** optimiser, a **learning-rate schedule**, and **early stopping** on the validation signers. We trained **three copies with different random seeds and average them**, which is steadier than one.
6. **Threshold:** the "sure enough" level (0.38) was chosen on validation signers, then measured on the test signers.
7. **Evaluation:** `scripts/evaluate.py` computes accuracy, precision, recall, F1, top-3, a confusion matrix and speed, and writes them to `reports/`.
8. **Training time:** about 1 to 3 hours on a CPU.

**Be ready to explain:** what a training/validation/test split is, why we split by signer, what overfitting is, and what augmentation does.

### Q6. What is the model's current accuracy, why, and how will you improve it?

**Numbers:** words **86.1%** (top-3 **94.7%**, macro F1 **0.86**) on 11 unseen signers; letters and numbers **97.2%**; tone **69.9%**. When the word model is confident, it is right **94%** of the time.

**Why it is not higher:**
- **280 classes** is a hard problem (a random guess is 0.4%). Accuracy dropped from 89.7% to 86.1% when we grew from 100 to 280 words.
- **About 15 clips per sign** for training; more data would help.
- **Similar signs** confuse it (for example "pain" and "hurt", "name" and "call"). The report lists the worst confusions.
- **Webcam quality and different people** vary. We test on people the model never saw, which is the honest but harder measure.
- **Landmarks lose detail:** finger contact and fine face movement are partly lost in 2D points.
- **Tone:** the face dataset is tiny greyscale with noisy labels (humans agree about 65%).

**How we plan to improve:**
1. More training clips per sign, and recordings from our own users.
2. Stronger augmentation and more model copies.
3. Add 3D hand information and finger-bend angles.
4. Targeted help for the most confused pairs.
5. Higher-resolution face data for tone.
6. Test with many real users and tune the confidence threshold.

### Q7. How did you divide the labour? (team of two)

**[FILL IN with the real split.]** Be specific and truthful. You can organise your answer by these areas, and each person must understand all of them at a basic level:

| Area | What it includes | Person |
|---|---|---|
| Data and model training | Datasets, landmark extraction, training scripts, evaluation, accuracy reports | **[FILL IN]** |
| Backend | FastAPI endpoints, SQLite learning engine, sentence building | **[FILL IN]** |
| Front end and design | SvelteKit pages, camera and signing figure, landing page, accessibility | **[FILL IN]** |
| Testing and documentation | Automated tests, browser checks, this guide, the demo | **[FILL IN]** |

Add how you worked together (for example: shared GitHub repository, regular meetings, each reviewing the other's work).

### Q8. What was the most difficult part?

Pick the one that is true for your team. Strong, accurate options:

- **Getting the model to work on people it has never seen.** An earlier approach scored well on its own training data but would not work for a new person. Splitting by signer made the honest score lower and forced us to fix the real problem.
- **Making live recognition match training.** The browser and the training code had to produce exactly the same numbers. We put the feature math in one shared file to guarantee it.
- **Knowing when a sign starts and ends** without a button, using hand position and a pause with hands down.
- **Running smoothly on a CPU.** MediaPipe slowed down when it used too many threads, so we tuned it to reach about 20 frames per second.
- **Data licensing:** the best dataset forbids redistribution, so we stream, convert and delete the videos and keep only the numbers.

**[Add your own story here: a specific bug, how you found it, and how you fixed it.]**

### Q9. What new things did you learn?

**[FILL IN with what each of you really learned.]** Topics this project covers, so you can be truthful about the ones that are new to you:

- Computer-vision landmarks with MediaPipe
- Training a sequence model (convolution and Transformer) and evaluating it properly (precision, recall, F1, confusion matrix, calibration)
- Why testing on unseen people matters (avoiding data leakage)
- Building a typed web API with FastAPI
- Building a modern front end with SvelteKit and TypeScript
- Spaced repetition and a learning database
- Accessibility (keyboard use, screen readers, contrast, reduced motion)
- Privacy-friendly design (video stays in the browser)
- Dataset licences and ethics for sign language and the Deaf community
- Automated testing and browser testing

### Q10. When did you start, and does the timeline fit inside six months?

The Git history shows the project **started on 10 September 2026** (first commits: the sign list and keypoint extraction) and the current version is dated **23 October 2026**. That is **about six weeks, well inside six months**. You can check this yourself with:

```bash
git log --format='%ad %s' --date=short | tail -3     # earliest commits
git log --format='%ad %s' --date=short | head -3     # latest commits
```

A truthful timeline you can present (adjust to what really happened):

| Period | Work |
|---|---|
| Early September | Plan, first version: 65 static signs, single-frame hand model |
| Late September to early October | Emotion experiments, working pipeline, interface fixes |
| Mid October | Rebuilt as SignBridge: motion-based word model, signer-independent evaluation, learning course, sentence building, new interface |
| Late October | Vocabulary grown to 280 words, SvelteKit front end and FastAPI backend, automated tests, documentation |

### Q11. Why is your word model better than a single-picture classifier?

A sign is a movement. Many signs use the same hand shape but move differently, so one frame cannot tell them apart. Our model reads 32 frames in order, so it sees the motion. Our first version used one frame per sign and could not separate such signs.

### Q12. How do you know the accuracy is real and not just memorised?

We test only on **11 signers who never appear in training or validation**, and the confidence threshold is chosen on validation signers, not test. The letters score is flagged as optimistic in the report because that dataset has no signer information. Reports are produced by `scripts/evaluate.py`, and the app reads them, so no number is typed by hand.

### Q13. How do you handle privacy?

The video is processed inside the browser tab and never recorded or uploaded. Only landmark numbers go to the server on the same computer. Progress is stored in a local SQLite file. There are no accounts and no analytics. Users can download or delete their data in Settings. If a Gemini key is used, only the signed words as text are sent.

### Q14. What about ethics and the Deaf community?

The app is an aid, not an interpreter replacement, and it says so. It covers ASL only and individual signs, not full grammar. Results on other people and cameras will differ, and fairness across skin tones has not been audited yet. Sign language belongs to the Deaf community, so we recommend learning from Deaf teachers and plan to work with them to review the vocabulary.

### Q15. What happens when the model is unsure or wrong?

Below the confidence threshold it does not guess. It shows a **"Did you mean" list** of the top three, which the user can pick with keys 1, 2 or 3. Users can undo a sign, clear the sentence, and edit before the sentence is spoken.

### Q16. How does it turn signs into a proper English sentence?

ASL word order differs from English. The recognised signs, the face tone and the question type go to the sentence builder. Without a key, simple rules add capital letters, punctuation and the question mark. With a Gemini key, the model rewrites the signs into natural English and respects the tone. The offline rules mean the app never depends on the internet.

### Q17. How does the learning part work?

The course has 34 units. For each sign the learner watches the figure, tries it on camera (the app checks it), and later takes a six-question quiz. Passing needs 80%. Spaced repetition brings back signs the learner is forgetting. Points, streaks, badges and a certificate keep it motivating.

### Q18. How did you test the project?

Automated tests at several levels: Python tests for features, the API and learning engine; type checking and unit tests for the front end; a real-browser suite that opens every page, runs lessons and flows, checks accessibility in three themes and phone width, and measures camera speed. We also replay 300 held-out clips through the live API to confirm the accuracy and the 14 ms response time.

### Q19. Did you use any tools or AI assistance? (answer honestly)

Check your **university's rules on AI and tool assistance** before the evaluation, and follow them. Whatever you did, **say it truthfully if asked**, and be ready to explain every file and decision you present. Evaluators mainly want to see that you understand the work: be ready to open any file, explain it, and answer follow-up questions. A good way to prepare is to run the project yourself, read `docs/DECISIONS.md`, and practise explaining each section of this guide in your own words.

### Q20. What would you do differently?

Suggested honest points: start collecting our own sign recordings earlier; test with real signers and Deaf users sooner; write the evaluation (signer-independent split) first instead of after the first model; and plan the vocabulary size against how much data exists per sign.

---

## 17. Demo script (three minutes)

1. Open `/`. Type the evaluator's name. The figure fingerspells it. Scroll to see the letters step by step.
2. Keep scrolling: how the app reads you, then the measured numbers, then a scripted sample conversation (labelled as a sample).
3. Open **Converse**. Allow the camera. Sign "hello", then "how" and "you" with raised eyebrows. Press **Enter**: the sentence is spoken as a question.
4. Let the evaluator type or say a reply. The figure signs it back.
5. Open **Learn**, start the alphabet, and show the camera check on one letter.
6. Open **Metrics** and explain why the test signers were held out.

**Before the demo:** start the server, test with your own webcam in good light, sit with shoulders and hands visible, and have the numbers in section 12 ready. Keep a backup screen recording in case the camera fails.

---

## 18. Glossary

| Term | Meaning |
|---|---|
| **ASL** | American Sign Language |
| **Landmarks** | Points a computer finds on a hand, body or face (for example 21 per hand) |
| **MediaPipe** | Google's free tool that finds those points in video |
| **Model** | A program that learned to make predictions from examples |
| **Training / validation / test** | Data to learn from / to tune choices / to measure the final score. Test data is never used for learning |
| **Overfitting** | A model that memorises its training data and fails on new data |
| **Augmentation** | Changing training examples slightly (shift, rotate) so the model copes with variety |
| **Accuracy** | Share of answers that are correct |
| **Precision / recall / F1** | Of what it named, how much was right / of what existed, how much it found / one number combining both |
| **Top-3** | The right answer is among the three best guesses |
| **Confusion matrix** | A table showing which signs get mistaken for which |
| **Ensemble** | Several models whose answers are averaged |
| **API** | The set of web addresses the front end calls to get things done |
| **Endpoint** | One such address, for example `/api/sentence` |
| **SPA** | Single-page application: a web app that changes pages without reloading |
| **Tunnel** | A secure link that gives your local server a public web address |
| **SQLite** | A database stored in one file |
| **Spaced repetition** | Showing items again just before you would forget them |
| **Gloss** | A sign written as an English word, in ASL order |
| **Fingerspelling** | Spelling a word letter by letter with hand shapes |
