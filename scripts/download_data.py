"""Download the public datasets into data/raw/<name>/. No accounts or API keys needed.

ASL Citizen videos are NOT downloaded here: build_words_dataset.py streams only the clips it needs
and deletes each one after landmark extraction (the dataset license asks for that).

Run: .venv/bin/python -m scripts.download_data [--only NAME]
"""
import argparse

from scripts.fetch_assets import fetch
from scripts.remote_zip import open_remote_zip
from signbridge.config import RAW

CITIZEN_ZIP = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
CITIZEN_FILES = ["use.txt", "splits/train.csv", "splits/val.csv", "splits/test.csv"]
HF = "https://huggingface.co/datasets/{id}/resolve/main/{path}"
HF_SOURCES = {
    "asl-letters": ("Marxulia/asl_sign_languages_alphabets_v03", ["data/train-00000-of-00001.parquet"]),
    "fer2013": ("Aaryan333/fer2013_train_publicTest_privateTest", [
        "data/train-00000-of-00001-5eab84e1c6a2fc27.parquet",
        "data/publicTest-00000-of-00001-f41bb7384b8aad6e.parquet",
        "data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet"]),
}


def download_citizen():
    out = RAW / "asl-citizen"
    out.mkdir(parents=True, exist_ok=True)
    z = None
    for name in CITIZEN_FILES:
        dest = out / name.split("/")[-1]
        if not dest.exists():
            z = z or open_remote_zip(CITIZEN_ZIP)
            dest.write_bytes(z.read("ASL_Citizen/" + name))
        print(dest.relative_to(RAW))
    print("NOTE: ASL Citizen is under the Microsoft Research License: non-commercial research only, "
          "no redistribution of the data or derived data, delete personal data when the research ends. "
          f"Read {out / 'use.txt'}.")


def download_hf(name):
    repo, paths = HF_SOURCES[name]
    for path in paths:
        dest = RAW / name / path.split("/")[-1]
        if not dest.exists():
            print("fetching", HF.format(id=repo, path=path))
            fetch(HF.format(id=repo, path=path), dest)
        print(f"{dest.relative_to(RAW)}  {dest.stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["asl-citizen", *HF_SOURCES])
    only = ap.parse_args().only
    if only in (None, "asl-citizen"):
        download_citizen()
    for name in HF_SOURCES:
        if only in (None, name):
            download_hf(name)


if __name__ == "__main__":
    main()
