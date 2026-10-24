"""Train a baseline CNN for Fire vs No Fire image classification."""
from pathlib import Path
import json
import tensorflow as tf
import matplotlib.pyplot as plt

SEED = 42
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

def build_model():
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.08),
        ],
        name="training_augmentation",
    )

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fire_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="cnn_fire_classification")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

def main():
    if not (DATA_DIR / "train").exists() or not (DATA_DIR / "val").exists():
        raise SystemExit("Data split belum ditemukan. Jalankan src/prepare_dataset.py terlebih dahulu.")

    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=True, seed=SEED
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False
    )
    class_names = train_ds.class_names
    print("Class order:", class_names)
    if class_names != ["Fire", "Non-Fire"]:
        raise SystemExit(f"Urutan label tidak sesuai ekspektasi: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "best_model.keras", monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks
    )
    # Save final model with the best weights restored by EarlyStopping.
    model.save(MODEL_DIR / "final_model.keras")

    hist = {key: [float(v) for v in values] for key, values in history.history.items()}
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_history.png", dpi=160)
    plt.close(fig)

    print("\nTraining selesai.")
    print(f"Model terbaik: {MODEL_DIR / 'best_model.keras'}")
    print(f"Grafik training: {RESULTS_DIR / 'training_history.png'}")
    print("Langkah berikutnya: python src/evaluate.py")

if __name__ == "__main__":
    main()
