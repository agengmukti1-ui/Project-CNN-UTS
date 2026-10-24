"""Evaluate the saved CNN on the held-out test split."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

DATA_DIR = Path("data/processed/test")
MODEL_PATH = Path("models/best_model.keras")
RESULTS_DIR = Path("results")
IMG_SIZE = (180, 180)
BATCH_SIZE = 32

def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model tidak ditemukan. Jalankan python src/train.py terlebih dahulu.")
    if not DATA_DIR.exists():
        raise SystemExit("Data test tidak ditemukan. Jalankan src/prepare_dataset.py terlebih dahulu.")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False
    )
    class_names = test_ds.class_names
    if class_names != ["Fire", "No Fire"]:
        raise SystemExit(f"Urutan label tidak sesuai ekspektasi: {class_names}")

    model = tf.keras.models.load_model(MODEL_PATH)
    y_true = np.concatenate([labels.numpy().ravel() for _, labels in test_ds]).astype(int)
    probabilities = model.predict(test_ds, verbose=1).ravel()
    y_pred = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fire": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_fire": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_fire": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "class_names": class_names,
        "threshold": 0.5,
        "test_samples": int(len(y_true)),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS_DIR / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\n=== HASIL EVALUASI DATA TEST ===")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"Hasil disimpan di: {RESULTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
