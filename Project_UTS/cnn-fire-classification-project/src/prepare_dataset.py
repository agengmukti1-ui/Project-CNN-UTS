"""Create train/validation/test splits from the original images only.

Expected source folders:
  Real Fire, Smoke, Safe Fire, Artificial Fire
The script searches recursively for these category folder names and copies images
into data/processed/{train,val,test}/{Fire,No Fire}/.
"""
from pathlib import Path
import argparse
import random
import shutil
import re

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_CLASSES = {
    "real fire": "Fire",
    "smoke": "Fire",
    "safe fire": "No Fire",
    "artificial fire": "No Fire",
}

def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", value.strip().lower()))

def locate_class_dirs(root: Path):
    found = {}
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
        key = normalize_name(directory.name)
        if key in SOURCE_CLASSES and key not in found:
            found[key] = directory
    missing = set(SOURCE_CLASSES) - set(found)
    if missing:
        expected = ", ".join(sorted(missing))
        raise SystemExit(
            f"Tidak menemukan folder kategori: {expected}\n"
            f"Folder input: {root}\n"
            "Pastikan Anda mengekstrak arsip Original Images, bukan Augmented Images."
        )
    return found

def main():
    parser = argparse.ArgumentParser(description="Split original fire dataset into train/val/test.")
    parser.add_argument("--input_dir", required=True, help="Folder yang berisi kategori original.")
    parser.add_argument("--output_dir", default="data/processed", help="Folder output split.")
    parser.add_argument("--train", type=float, default=0.70, help="Proporsi training (default 0.70).")
    parser.add_argument("--val", type=float, default=0.15, help="Proporsi validation (default 0.15).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train <= 0 or args.val <= 0 or args.train + args.val >= 1:
        raise SystemExit("Proporsi harus > 0 dan train + val harus kurang dari 1.")

    source_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    if not source_root.exists():
        raise SystemExit(f"Folder input tidak ditemukan: {source_root}")

    class_dirs = locate_class_dirs(source_root)
    rng = random.Random(args.seed)
    counts = {split: {"Fire": 0, "No Fire": 0} for split in ("train", "val", "test")}

    # Start with a clean output to avoid stale files from previous runs.
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in counts:
        for label in ("Fire", "No Fire"):
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    for source_class, label in SOURCE_CLASSES.items():
        class_dir = class_dirs[source_class]
        images = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(images) < 3:
            raise SystemExit(f"Terlalu sedikit gambar di {class_dir}: {len(images)}")
        rng.shuffle(images)

        n = len(images)
        n_train = max(1, int(n * args.train))
        n_val = max(1, int(n * args.val))
        # Ensure at least one sample in test.
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1
        split_items = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, items in split_items.items():
            for index, src in enumerate(items):
                # Prefix with source category to avoid collisions between same-named files.
                safe_name = f"{source_class.replace(' ', '_')}_{index:05d}_{src.name}"
                dest = output_root / split / label / safe_name
                shutil.copy2(src, dest)
                counts[split][label] += 1

    print(f"Dataset split selesai: {output_root}")
    print(f"Seed: {args.seed}; train={args.train:.2f}, val={args.val:.2f}, test={1-args.train-args.val:.2f}")
    for split, labels in counts.items():
        print(f"{split:5s} | Fire: {labels['Fire']:4d} | No Fire: {labels['No Fire']:4d} | total: {sum(labels.values()):4d}")
    print("\nCatatan: gunakan hanya folder Original Images agar augmented copies tidak bocor antar-split.")

if __name__ == "__main__":
    main()
