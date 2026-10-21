"""Download the MediaPipe .task models and vendor the tasks-vision JS/WASM bundle.

Run: .venv/bin/python -m scripts.fetch_assets
"""
import shutil
import urllib.request
from importlib.metadata import version

from signbridge.config import WEB

GCS = "https://storage.googleapis.com/mediapipe-models"
CDN = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{version('mediapipe')}"  # JS must match Python version
MODELS = WEB / "static" / "models"
VENDOR = WEB / "static" / "vendor" / "tasks-vision"

FILES = {
    MODELS / "hand_landmarker.task": f"{GCS}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    MODELS / "face_landmarker.task": f"{GCS}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    MODELS / "pose_landmarker_lite.task": f"{GCS}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    VENDOR / "vision_bundle.mjs": f"{CDN}/vision_bundle.mjs",
    **{VENDOR / "wasm" / n: f"{CDN}/wasm/{n}" for n in (
        "vision_wasm_internal.js", "vision_wasm_internal.wasm",
        "vision_wasm_nosimd_internal.js", "vision_wasm_nosimd_internal.wasm")},
}


def fetch(url, path):
    """Download to <name>.part, then rename, so an interrupted download is never mistaken for a finished one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(path)


def main():
    for path, url in FILES.items():
        if not path.exists() or path.stat().st_size == 0:
            print("fetching", url)
            fetch(url, path)
        print(f"{path.relative_to(WEB)}  {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
