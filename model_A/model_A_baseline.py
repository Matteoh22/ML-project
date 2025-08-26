#!/usr/bin/env python3
"""
Model A (baseline, compact) — saves best/final model + plots + report.
- P1: 64x64, normalize [0,1], no augmentation
- CNN: Conv(16) → MaxPool → Flatten → Dense(32) → Softmax(3)
- Split: train / val / test tramite image_dataset_from_directory (+ split val/test 50/50)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# ==== Config ====
DATA_DIR = Path("/Users/matteohasa/Desktop/ML-project/rps-cv-images")
OUT_DIR = Path("./outputs/model_A_compact"); OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE = (64, 64); BATCH = 32; EPOCHS = 15; SEED = 42

# ==== Datasets ====
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED,
    validation_split=0.30, subset="training", shuffle=True
)
temp_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, labels="inferred", label_mode="int",
    image_size=IMG_SIZE, batch_size=BATCH, seed=SEED,
    validation_split=0.30, subset="validation", shuffle=True
)
class_names = train_ds.class_names
temp_batches = temp_ds.cardinality().numpy()
val_ds  = temp_ds.take(temp_batches // 2)
test_ds = temp_ds.skip(temp_batches // 2)

# normalize to [0,1] + performance
def norm(ds): return ds.map(lambda x,y:(tf.cast(x,tf.float32)/255.0,y)).cache().prefetch(tf.data.AUTOTUNE)
train_ds, val_ds, test_ds = norm(train_ds), norm(val_ds), norm(test_ds)

# ==== Model ====
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

# ==== Callbacks (save best) ====
ckpt_path = (OUT_DIR / "model_A_best.keras").as_posix()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
]

# ==== Train ====
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

# ==== Save final model ====
final_path = (OUT_DIR / "model_A_final.keras").as_posix()
model.save(final_path)
print(f"Saved best to: {ckpt_path}\nSaved final to: {final_path}")

# ==== Plots → files ====
plt.figure(); plt.plot(history.history["loss"]); plt.plot(history.history["val_loss"])
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (train/val)"); plt.legend(["train","val"])
plt.tight_layout(); plt.savefig((OUT_DIR / "training_loss.png").as_posix()); plt.close()

plt.figure(); plt.plot(history.history["accuracy"]); plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (train/val)"); plt.legend(["train","val"])
plt.tight_layout(); plt.savefig((OUT_DIR / "training_accuracy.png").as_posix()); plt.close()

# ==== Test eval + report → files ====
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

y_true, y_pred = [], []
for x, y in test_ds:
    p = model.predict(x, verbose=0)
    y_true.extend(y.numpy()); y_pred.extend(np.argmax(p, axis=1))

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write("Classification Report (Model A compact)\n"); f.write(report)
print("\nClassification report saved.")

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix (Model A)")
plt.colorbar(); ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha="right"); plt.yticks(ticks, class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout(); plt.savefig((OUT_DIR / "confusion_matrix.png").as_posix()); plt.close()

# ==== Quick summary file ====
with open(OUT_DIR / "README_Model_A_compact.txt", "w", encoding="utf-8") as f:
    f.write(
        "Model A (compact) summary:\n"
        "- P1: 64x64, normalize [0,1], no augmentation\n"
        "- Conv(16)-MaxPool-Flat-Dense(32)-Softmax\n"
        "- Saved: model_A_best.keras, model_A_final.keras, plots, report\n"
    )
print("All outputs saved in:", OUT_DIR.resolve())