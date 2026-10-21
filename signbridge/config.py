"""Every constant and path. The browser reads the live values from GET /api/config."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
WEB = ROOT / "web"
SIGNS_DIR = WEB / "static" / "signs"
DB_PATH = DATA / "app.db"

# Point layout of one frame, shape (N_POINTS, 2). See landmarks.py.
HAND_N = 21
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # nose, L/R shoulder, L/R elbow, L/R wrist (BlazePose)
LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
           78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LH = slice(0, 21)      # anatomical LEFT hand
RH = slice(21, 42)     # anatomical RIGHT hand
POSE = slice(42, 49)   # POSE_IDX order
LIPS = slice(49, 89)   # LIP_IDX order
N_POINTS = 89
T_RAW = 64             # frames stored per training sample
T_MODEL = 32           # frames the word model sees
WORD_FEATURES = 400    # 178 pos + 178 vel + 42 dominant-hand local + 2 presence

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")
EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised"]
SEED = 42

# Live thresholds. WORD_ACCEPT / STATIC_ACCEPT are overridden at load time by
# "accept_threshold" in reports/words.json and reports/static.json when present.
WORD_ACCEPT = 0.55
STATIC_ACCEPT = 0.70
STATIC_STREAK = 8
SPELL_GAP_FRAMES = 25
SEG_START_FRAMES = 3
SEG_END_FRAMES = 10
SEG_MIN_FRAMES = 12
SEG_MAX_FRAMES = 90
BROW_RAISE_DELTA = 0.20
BROW_DOWN_DELTA = 0.20
EMOTION_INTERVAL_MS = 400
EMOTION_EMA = 0.35
