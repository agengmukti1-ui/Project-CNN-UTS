\
"""Download and extract the lightweight public synthetic fire dataset from Mendeley Data."""
from __future__ import annotations
import json
import shutil
import sys
import zipfile
from pathlib import Path
import requests

DATASET_ID = "mx33vrmxrv"
VERSION = 1
API_BASE = "https://api.data.mendeley.com"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = DATA_DIR / "synthetic_fire_non_fire.zip"
MAX_SIZE_MB = 120  # Published archive is about 89.5 MB; stop if source unexpectedly grows.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def human_size(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_url = f"{API_BASE}/datasets/{DATASET_ID}/zip"
    print("Dataset: Synthetic Image fire and non-fire dataset (Mendeley Data)")
    print("Memeriksa metadata dan ukuran arsip sebelum mengunduh...")
    try:
        response = requests.get(metadata_url, params={"version": VERSION}, timeout=30)
        response.raise_for_status()
        metadata = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Gagal mengambil metadata Mendeley: {exc}")
        print("Cek koneksi atau akses manual: https://data.mendeley.com/datasets/mx33vrmxrv/1")
        return 1

    size = metadata.get("size")
    download_url = metadata.get("url")
    status = metadata.get("status", "unknown")
    if size is not None:
        print(f"Ukuran arsip: {human_size(int(size))} (status: {status})")
        if int(size) > MAX_SIZE_MB * 1024 * 1024:
            print(f"Dibatalkan: ukuran arsip melewati batas aman {MAX_SIZE_MB} MB.")
            print("Periksa halaman dataset terlebih dahulu sebelum melanjutkan.")
            return 1
    if not download_url:
        print(f"URL arsip belum tersedia (status: {status}).")
        print("Coba lagi nanti atau akses: https://data.mendeley.com/datasets/mx33vrmxrv/1")
        return 1

    # Resume is intentionally not attempted; remove partial archives before a fresh download.
    ZIP_PATH.unlink(missing_ok=True)
    print("Mengunduh arsip dataset...")
    try:
        with requests.get(download_url, stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", size or 0))
            downloaded = 0
            with ZIP_PATH.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(f"\rProgress: {downloaded / total * 100:5.1f}% ({human_size(downloaded)})", end="", flush=True)
        print("\nUnduhan selesai.")
    except requests.RequestException as exc:
        ZIP_PATH.unlink(missing_ok=True)
        print(f"Gagal mengunduh arsip: {exc}")
        return 1

    # Validate ZIP before clearing previous extraction.
    try:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            bad = zf.testzip()
            if bad:
                raise zipfile.BadZipFile(f"Anggota ZIP rusak: {bad}")
            destination = RAW_DIR.resolve()
            for member in zf.infolist():
                target = (RAW_DIR / member.filename).resolve()
                if target != destination and destination not in target.parents:
                    raise RuntimeError(f"Path tidak aman di dalam ZIP: {member.filename}")
            if RAW_DIR.exists():
                shutil.rmtree(RAW_DIR)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            zf.extractall(RAW_DIR)
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        print(f"Gagal memvalidasi/mengekstrak ZIP: {exc}")
        return 1

    images = [x for x in RAW_DIR.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        print("Arsip berhasil diekstrak tetapi tidak ada gambar yang ditemukan.")
        print("Periksa struktur arsip di data/raw/ atau halaman dataset.")
        return 1
    print(f"Selesai: ditemukan {len(images)} gambar di {RAW_DIR}")
    print("Langkah berikutnya: python src/prepare_dataset.py")
    print("Arsip unduhan disimpan di data/synthetic_fire_non_fire.zip")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
