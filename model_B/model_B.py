import os, shutil
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from shutil import copy2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import regularizers

# Config
DATA_DIR = Path("/Users/matteohasa/Desktop/ML-project/rps-cv-images")
OUT_DIR  = Path("/Users/matteohasa/Desktop/ML-project/model_B/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH    = 32
EPOCHS   = 25
SEED     = 42

TEST_RATIO = 0.15
VAL_RATIO  = 0.15
AUTOTUNE   = tf.data.AUTOTUNE

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Files & labels
exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
name2lab = {n: i for i, n in enumerate(class_names)}

files, labs = [], []
for cname in class_names:
    for f in (DATA_DIR / cname).rglob("*"):
        if f.suffix.lower() in exts:
            files.append(str(f)); labs.append(name2lab[cname])

if not files:
    raise RuntimeError(f"No images found in {DATA_DIR}")

print("Classes:", class_names)
print(f"Total images: {len(files)}")

# Stratified split 70/15/15
X_tmp, X_test, y_tmp, y_test = train_test_split(
    files, labs, test_size=TEST_RATIO, stratify=labs, random_state=SEED
)
val_rel = VAL_RATIO / (1.0 - TEST_RATIO)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=SEED
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Compute mean/std on train (after resize)
def _read_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, IMG_SIZE, antialias=True)
    return img

means, sqmeans = [], []
for p in X_train:
    im = _read_resize(p)
    means.append(tf.reduce_mean(im, axis=[0,1]).numpy())
    sqmeans.append(tf.reduce_mean(tf.square(im), axis=[0,1]).numpy())

mean = np.mean(np.stack(means), axis=0)
sqm  = np.mean(np.stack(sqmeans), axis=0)
std  = np.sqrt(np.maximum(sqm - mean**2, 1e-8))
np.save(OUT_DIR / "train_mean.npy", mean)
np.save(OUT_DIR / "train_std.npy",  std)
print("Train mean:", mean); print("Train std :", std)

# Pipelines
aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_aug")

mean_t = tf.constant(mean, dtype=tf.float32)
std_t  = tf.constant(std,  dtype=tf.float32)

def _map_fn(path, label, augment=False):
    img = _read_resize(path)
    if augment:
        img = aug(tf.expand_dims(img,0), training=True)[0]
    img = (img - mean_t) / tf.maximum(std_t, 1e-6)
    return img, label

def make_ds(paths, labels, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, np.array(labels, np.int32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p,y: _map_fn(p,y,augment), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

train_ds = make_ds(X_train, y_train, shuffle=True,  augment=True)
val_ds   = make_ds(X_val,   y_val,   shuffle=False, augment=False)
test_ds  = make_ds(X_test,  y_test,  shuffle=False, augment=False)

# small subset for search
train_ds_search = train_ds.take(35)  # ~35 batches
val_ds_search   = val_ds.take(10)    # ~10 batches

# Hyperparameter tuning
def conv_block(x, filters, k, use_bn, wd):
    x = tf.keras.layers.Conv2D(filters, k, padding="same",
                               use_bias=not use_bn,
                               kernel_regularizer=regularizers.l2(wd) if wd>0 else None)(x)
    if use_bn: x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    return x

def build_model(num_classes, c1, c2, c3, k1, k2, k3,
                use_bn, dense_units, dp1, dp2, wd):
    inp = tf.keras.Input(shape=(*IMG_SIZE,3))
    x = inp
    x = conv_block(x, c1, k1, use_bn, wd)
    x = conv_block(x, c2, k2, use_bn, wd)
    x = conv_block(x, c3, k3, use_bn, wd)
    x = tf.keras.layers.Flatten()(x)
    if dp1 > 0: x = tf.keras.layers.Dropout(dp1)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu",
                              kernel_regularizer=regularizers.l2(wd) if wd>0 else None)(x)
    if dp2 > 0: x = tf.keras.layers.Dropout(dp2)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="model_B_tuned")

# Search space
space = {
    "c1": [32, 48, 64],
    "c2": [64, 96, 128],
    "c3": [128, 160, 192],
    "k1": [3, 5],
    "k2": [3, 5],
    "k3": [3],
    "use_bn": [True, False],
    "dense_units": [128, 192, 256],
    "dp1": [0.0, 0.3, 0.4],
    "dp2": [0.0, 0.3],
    "wd":  [0.0, 1e-5, 1e-4],  
    "lr":  [1e-3, 7e-4, 5e-4]
}

TRIALS = 8
SEARCH_EPOCHS = 6

best_val = -np.inf
best_cfg = None

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

for t in range(TRIALS):
    cfg = {k: random.choice(v) for k,v in space.items()}
    m = build_model(len(class_names),
                    cfg["c1"], cfg["c2"], cfg["c3"],
                    cfg["k1"], cfg["k2"], cfg["k3"],
                    cfg["use_bn"], cfg["dense_units"],
                    cfg["dp1"], cfg["dp2"], cfg["wd"])
    m.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(train_ds_search, validation_data=val_ds_search,
              epochs=SEARCH_EPOCHS, verbose=0, callbacks=[es])
    val_acc = max(h.history["val_accuracy"])
    print(f"[Trial {t+1}/{TRIALS}] {cfg} -> best val_acc={val_acc:.4f}")
    if val_acc > best_val:
        best_val, best_cfg = val_acc, cfg

print("\nSelected config:", best_cfg, f"(val_acc={best_val:.4f})")

# Build final model with best cfg (train on full dataset)
model = build_model(len(class_names),
                    best_cfg["c1"], best_cfg["c2"], best_cfg["c3"],
                    best_cfg["k1"], best_cfg["k2"], best_cfg["k3"],
                    best_cfg["use_bn"], best_cfg["dense_units"],
                    best_cfg["dp1"], best_cfg["dp2"], best_cfg["wd"])
model.compile(optimizer=tf.keras.optimizers.Adam(best_cfg["lr"]),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Optional class weights 
y_train_np = np.asarray(y_train, dtype=int)
counts = np.bincount(y_train_np, minlength=len(class_names))
print("Train distribution:", {class_names[i]: int(c) for i,c in enumerate(counts)})
dev = np.max(np.abs(counts - counts.mean()) / np.maximum(counts.mean(), 1e-8))
class_weight = None
lbls = np.unique(y_train_np)
if dev > 0.15 and len(lbls) == len(class_names):
    w = compute_class_weight(class_weight="balanced", classes=lbls, y=y_train_np)
    class_weight = {int(c): float(wi) for c, wi in zip(lbls, w)}
    print("Using class_weight:", class_weight)

# Training 
ckpt_path = (OUT_DIR / "model_B_best.keras").as_posix()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.005, patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=EPOCHS, callbacks=callbacks,
                    class_weight=class_weight, verbose=1)

# Save final
final_path = (OUT_DIR / "model_B_final.keras").as_posix()
model.save(final_path)
print(f"Best model:  {ckpt_path}\nFinal model: {final_path}")

# Plots
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.legend(); plt.title("Accuracy (Model B)")
plt.savefig(OUT_DIR / "accuracy.png"); plt.close()

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend(); plt.title("Loss (Model B)")
plt.savefig(OUT_DIR / "loss.png"); plt.close()

# Test + report + confusion matrix
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test acc = {test_acc:.4f}")

y_true, y_pred = [], []
for xb, yb in test_ds:
    probs = model.predict(xb, verbose=0)
    y_true.extend(yb.numpy())
    y_pred.extend(np.argmax(probs, axis=1))

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
plt.imshow(cm, interpolation="nearest", cmap="Blues"); plt.title("Confusion Matrix (Model B)")
plt.colorbar(); ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha="right"); plt.yticks(ticks, class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png"); plt.close()

# Save errors 
errors_dir = OUT_DIR / "errors"
if errors_dir.exists(): shutil.rmtree(errors_dir)
errors_dir.mkdir(parents=True, exist_ok=True)

for idx, (t, p) in enumerate(zip(y_test, y_pred)):
    if int(t) != int(p):
        true_name = class_names[int(t)]; pred_name = class_names[int(p)]
        dst = errors_dir / true_name / f"pred_{pred_name}"
        dst.mkdir(parents=True, exist_ok=True)
        try:
            copy2(X_test[idx], dst / Path(X_test[idx]).name)
        except Exception:
            pass

print("Done! Errors saved in:", errors_dir)
print("Outputs saved in:", OUT_DIR.resolve())