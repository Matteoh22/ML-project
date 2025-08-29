#!/usr/bin/env python3
"""
Model A (baseline, compact) — balanced splits + saves best/final + plots + report.
- Crea split bilanciati (train/val/test) copiando i file
- P1: 64x64, normalize [0,1], no augmentation
- CNN: Conv(16) → MaxPool → Flatten → Dense(32) → Softmax(3)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random, shutil
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# === CONFIG ===
DATA_DIR  = Path("/Users/matteohasa/Desktop/ML-project1/rps-cv-images")   # dataset originale
SPLIT_DIR = Path("/Users/matteohasa/Desktop/ML-project/model_A/model_A-images_splits_balanced")  # dove copiamo gli split
MODEL_DIR = Path("/Users/matteohasa/Desktop/ML-project/model_A/outputs")  # dove salviamo modelli/plot
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
SEED = 42
IMG_SIZE = (64, 64)
BATCH = 32
EPOCHS = 15

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === 1) Scan images per class ===
exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
class2files = defaultdict(list)
for cls_dir in DATA_DIR.iterdir():
    if cls_dir.is_dir():
        for f in cls_dir.rglob("*"):
            if f.suffix.lower() in exts:
                class2files[cls_dir.name].append(f)

if not class2files:
    raise RuntimeError(f"Nessuna immagine trovata in {DATA_DIR}")

# === 2) Compute balanced count (min across classes) ===
min_count = min(len(files) for files in class2files.values())
print("Balanced split will use", min_count, "images per class")

# === 3) Create output dirs (dataset splits) ===
for split in SPLITS.keys():
    for cls in class2files.keys():
        (SPLIT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# === 4) Split per class (downsample + copy) ===
for cls, files in class2files.items():
    files = random.sample(files, min_count)  # downsample if necessary
    n_train = int(SPLITS["train"] * min_count)
    n_val   = int(SPLITS["val"]   * min_count)
    n_test  = min_count - n_train - n_val

    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:]

    for f in train_files:
        shutil.copy(f, SPLIT_DIR / "train" / cls / f.name)
    for f in val_files:
        shutil.copy(f, SPLIT_DIR / "val" / cls / f.name)
    for f in test_files:
        shutil.copy(f, SPLIT_DIR / "test" / cls / f.name)

print("Done! Balanced splits saved in", SPLIT_DIR)

# === 5) Build tf.data datasets from the BALANCED folders ===
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    SPLIT_DIR / "train", labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    SPLIT_DIR / "val", labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, shuffle=False
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    SPLIT_DIR / "test", labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, shuffle=False
)
class_names = train_ds.class_names  # ora riferito agli split bilanciati

# normalize to [0,1] + performance
def norm(ds):
    return ds.map(lambda x,y:(tf.cast(x,tf.float32)/255.0,y)).cache().prefetch(tf.data.AUTOTUNE)

train_ds, val_ds, test_ds = norm(train_ds), norm(val_ds), norm(test_ds)

# === 6) Model ===
inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs, name="model_A_baseline_compact")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# === 7) Callbacks (save best) ===
ckpt_path = (MODEL_DIR / "model_A_best.keras").as_posix()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
]

# === 8) Train ===
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

# === 9) Save final model ===
final_path = (MODEL_DIR / "model_A_final.keras").as_posix()
model.save(final_path)
print(f"Saved best to: {ckpt_path}\nSaved final to: {final_path}")

# === 10) Plots → files ===
plt.figure(); plt.plot(history.history["loss"]); plt.plot(history.history["val_loss"])
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (train/val)"); plt.legend(["train","val"])
plt.tight_layout(); plt.savefig((MODEL_DIR / "training_loss.png").as_posix()); plt.close()

plt.figure(); plt.plot(history.history["accuracy"]); plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (train/val)"); plt.legend(["train","val"])
plt.tight_layout(); plt.savefig((MODEL_DIR / "training_accuracy.png").as_posix()); plt.close()

# === 11) Test eval + report → files ===
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

y_true, y_pred = [], []
for x, y in test_ds:
    p = model.predict(x, verbose=0)
    y_true.extend(y.numpy()); y_pred.extend(np.argmax(p, axis=1))

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
with open(MODEL_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write("Classification Report (Model A compact, balanced splits)\n"); f.write(report)
print("\nClassification report saved.")

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix (Model A)")
plt.colorbar(); ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha="right"); plt.yticks(ticks, class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout(); plt.savefig((MODEL_DIR / "confusion_matrix.png").as_posix()); plt.close()

with open(MODEL_DIR / "README_Model_A_compact_balanced.txt", "w", encoding="utf-8") as f:
    f.write(
        "Model A (compact) with balanced splits:\n"
        "- P1: 64x64, normalize [0,1], no augmentation\n"
        "- Conv(16)-MaxPool-Flat-Dense(32)-Softmax\n"
        "- Balanced train/val/test from original dataset\n"
        "- Saved: model_A_best.keras, model_A_final.keras, plots, report\n"
    )
print("All outputs saved in:", MODEL_DIR.resolve())