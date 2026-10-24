\
"""Split the public dataset's Fire and Non-Fire folders into train/val/test."""
from pathlib import Path
import argparse
import random
import re
import shutil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ALIASES = {
    "fire": "Fire", "fires": "Fire", "fire images": "Fire", "fire image": "Fire",
    "non fire": "Non-Fire", "nonfire": "Non-Fire", "non fire images": "Non-Fire",
    "nonfire images": "Non-Fire", "no fire": "Non-Fire", "no fire images": "Non-Fire",
}


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", value.strip().lower()))


def image_files(directory: Path):
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def locate_class_dirs(root: Path):
    candidates = {"Fire": [], "Non-Fire": []}
    for directory in root.rglob("*"):
        if directory.is_dir():
            label = CLASS_ALIASES.get(normalize_name(directory.name))
            if label:
                count = len(image_files(directory))
                if count:
                    candidates[label].append((count, directory))
    found = {}
    for label, options in candidates.items():
        if not options:
            continue
        # Prefer the candidate folder containing the most images; common for archives with wrapper folders.
        found[label] = sorted(options, key=lambda item: item[0], reverse=True)[0][1]
    missing = set(candidates) - set(found)
    if missing:
        raise SystemExit(
            f"Tidak menemukan folder kelas {', '.join(sorted(missing))} di {root}.\n"
            "Struktur yang diharapkan memiliki folder Fire dan Non-Fire.\n"
            "Jalankan python download_dataset.py terlebih dahulu dan periksa isi data/raw/."
        )
    return found


def main():
    parser = argparse.ArgumentParser(description="Split Fire/Non-Fire images into train, validation and test sets.")
    parser.add_argument("--input_dir", default="data/raw", help="Folder dataset hasil ekstraksi (default: data/raw).")
    parser.add_argument("--output_dir", default="data/processed", help="Folder output split.")
    parser.add_argument("--train", type=float, default=0.70, help="Proporsi train (default 0.70).")
    parser.add_argument("--val", type=float, default=0.15, help="Proporsi validation (default 0.15).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train <= 0 or args.val <= 0 or args.train + args.val >= 1:
        raise SystemExit("Proporsi train dan val harus > 0 dan jumlahnya kurang dari 1.")
    source_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    if not source_root.exists():
        raise SystemExit(f"Folder input tidak ditemukan: {source_root}. Jalankan python download_dataset.py.")
    class_dirs = locate_class_dirs(source_root)
    rng = random.Random(args.seed)
    counts = {split: {"Fire": 0, "Non-Fire": 0} for split in ("train", "val", "test")}
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in counts:
        for label in ("Fire", "Non-Fire"):
            (output_root / split / label).mkdir(parents=True, exist_ok=True)
    for label, class_dir in class_dirs.items():
        images = image_files(class_dir)
        if len(images) < 3:
            raise SystemExit(f"Terlalu sedikit gambar pada {class_dir}: {len(images)}")
        rng.shuffle(images)
        n = len(images)
        n_train = max(1, int(n * args.train))
        n_val = max(1, int(n * args.val))
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1
        split_items = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:],
        }
        for split, items in split_items.items():
            for index, src in enumerate(items):
                dest = output_root / split / label / f"{label.replace('-', '_')}_{index:04d}_{src.name}"
                shutil.copy2(src, dest)
                counts[split][label] += 1
    print(f"Dataset siap: {output_root}")
    print(f"Pembagian: train={args.train:.0%}, validation={args.val:.0%}, test={1-args.train-args.val:.0%}; seed={args.seed}")
    for split, labels in counts.items():
        print(f"{split:10s} | Fire: {labels['Fire']:4d} | Non-Fire: {labels['Non-Fire']:4d} | Total: {sum(labels.values()):4d}")
    print("Catatan: dataset ini berisi gambar sintetis hasil AI; evaluasi tidak otomatis mewakili kondisi kebakaran nyata.")

if __name__ == "__main__":
    main()
