"""Predict Fire/No Fire for a single image."""
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("models/best_model.keras")
IMG_SIZE = (180, 180)
CLASS_NAMES = ["Fire", "No Fire"]

def main():
    parser = argparse.ArgumentParser(description="Predict whether an image belongs to Fire or No Fire.")
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
    # Output sigmoid is the probability of class index 1 (No Fire), since directory order is Fire, No Fire.
    p_no_fire = float(model.predict(array, verbose=0)[0][0])
    p_fire = 1.0 - p_no_fire
    predicted = "No Fire" if p_no_fire >= 0.5 else "Fire"

    print(f"Image: {image_path}")
    print(f"Prediction: {predicted}")
    print(f"Fire score: {p_fire:.4f}")
    print(f"No Fire score: {p_no_fire:.4f}")
    print("Catatan: skor model bukan probabilitas keselamatan yang tervalidasi.")

if __name__ == "__main__":
    main()
