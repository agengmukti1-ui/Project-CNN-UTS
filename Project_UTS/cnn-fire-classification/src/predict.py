"""Predict Fire/Non-Fire for a single image."""
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("models/best_model.keras")
IMG_SIZE = (180, 180)
CLASS_NAMES = ["Fire", "Non-Fire"]

def main():
    parser = argparse.ArgumentParser(description="Predict whether an image belongs to Fire or Non-Fire.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Gambar tidak ditemukan: {image_path}")
    if not MODEL_PATH.exists():
        raise SystemExit("Model belum ada. Jalankan src/train.py terlebih dahulu.")

    image = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)

    model = tf.keras.models.load_model(MODEL_PATH)
    # Output sigmoid is the probability of class index 1 (Non-Fire), since directory order is Fire, Non-Fire.
    p_non_fire = float(model.predict(array, verbose=0)[0][0])
    p_fire = 1.0 - p_non_fire
    predicted = "Non-Fire" if p_non_fire >= 0.5 else "Fire"

    print(f"Image: {image_path}")
    print(f"Prediction: {predicted}")
    print(f"Fire score: {p_fire:.4f}")
    print(f"Non-Fire score: {p_non_fire:.4f}")
    print("Catatan: skor model bukan probabilitas keselamatan yang tervalidasi.")

if __name__ == "__main__":
    main()
