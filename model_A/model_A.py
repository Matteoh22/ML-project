import os, shutil, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Config
DATA_DIR = Path("/Users/matteohasa/Desktop/ML-project/rps-cv-images")
OUT_DIR  = Path("/Users/matteohasa/Desktop/ML-project/model_A/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (64, 64)
BATCH = 32
EPOCHS = 25
SEED = 42

# Dataset split 70% train / 30% validation-test
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED,
    validation_split=0.30, subset="training", shuffle=True
)
temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED,
    validation_split=0.30, subset="validation", shuffle=True
)

class_names = train_ds.class_names
print("Classi:", class_names)

# Divide 30% in 15% validation and 15% test
temp_batches = tf.data.experimental.cardinality(temp_ds).numpy()
val_ds  = temp_ds.take(temp_batches // 2)
test_ds = temp_ds.skip(temp_batches // 2)

# Preprocess: normalization + cache/prefetch
def preprocess(ds):
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)) \
             .cache().prefetch(tf.data.AUTOTUNE)

train_ds, val_ds, test_ds = map(preprocess, [train_ds, val_ds, test_ds])

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=(*IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Callbacks: Best Value and Early Stopping
ckpt_path = OUT_DIR / "model_A_best.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

# Training
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=EPOCHS, callbacks=callbacks, verbose=1)

# Save final
final_path = OUT_DIR / "model_A_final.keras"
model.save(final_path)
print(f"Best model: {ckpt_path}\nFinal model: {final_path}")

# Plots
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.legend(); plt.title("Accuracy")
plt.savefig(OUT_DIR / "accuracy.png"); plt.close()

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend(); plt.title("Loss")
plt.savefig(OUT_DIR / "loss.png"); plt.close()

# Test + Report
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test acc = {test_acc:.4f}")

y_true, y_pred = [], []
for x, y in test_ds:
    p = model.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(p, axis=1))

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
with open(OUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix")
plt.colorbar(); plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png"); plt.close()

# Saving Errors
errors_dir = OUT_DIR / "errors"
if errors_dir.exists(): shutil.rmtree(errors_dir)
for true, pred in zip(y_true, y_pred):
    if true != pred:
        t, p = class_names[true], class_names[pred]
        (errors_dir / t / f"pred_{p}").mkdir(parents=True, exist_ok=True)

print("Done! Errors saved in:", errors_dir)