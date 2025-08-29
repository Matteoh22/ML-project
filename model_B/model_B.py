#!/usr/bin/env python3
"""
Model B (compact + scientific): intermediate CNN
- P2: 128x128, per-channel standardization (train mean/std), moderate augmentation (train only)
- Strict stratified split (sklearn) + fixed seeds
- Saves: best & final model, plots, confusion matrix, classification report
- Tiny LR sweep (optional) to demonstrate hyperparameter tuning
"""

from pathlib import Path
import random, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# ==== Config ====
DATA_DIR = Path("/Users/matteohasa/Desktop/ML-project/rps-cv-images")
OUT_DIR  = Path("/Users/matteohasa/Desktop/ML-project/model_B/outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE = (128, 128); BATCH = 32
EPOCHS = 25; SEED = 42
TEST_RATIO = 0.15; VAL_RATIO = 0.15
BASE_LR = 1e-3
AUTOTUNE = tf.data.AUTOTUNE

# ==== Seeds ====
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ==== 1) Scan files & labels ====
exts = {".jpg",".jpeg",".png",".bmp",".gif"}
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
name2lab = {n:i for i,n in enumerate(class_names)}
files, labs = [], []
for cname in class_names:
    for f in (DATA_DIR/cname).rglob("*"):
        if f.suffix.lower() in exts:
            files.append(str(f)); labs.append(name2lab[cname])
if not files: raise RuntimeError(f"No images found in {DATA_DIR}")

# ==== 2) Stratified split: test, then val from remaining ====
X_tmp, X_test, y_tmp, y_test = train_test_split(files, labs, test_size=TEST_RATIO,
                                                stratify=labs, random_state=SEED)
val_rel = VAL_RATIO / (1.0 - TEST_RATIO)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_rel,
                                                  stratify=y_tmp, random_state=SEED)

# ==== 3) Compute train mean/std (after resize to 128) ====
def _read_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)   # [0,1]
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
np.save(OUT_DIR/"train_mean.npy", mean); np.save(OUT_DIR/"train_std.npy", std)

# ==== 4) tf.data pipelines (augment only on train) ====
aug_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")

mean_t = tf.constant(mean, dtype=tf.float32)
std_t  = tf.constant(std,  dtype=tf.float32)

def _map_fn(path, label, augment=False):
    img = _read_resize(path)
    if augment:
        img = aug_layer(tf.expand_dims(img,0), training=True)[0]
    img = (img - mean_t) / tf.maximum(std_t, 1e-6)
    return img, label

def make_ds(paths, labels, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, np.array(labels, np.int32)))
    if shuffle: ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p,y: _map_fn(p,y,augment), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

ds_train = make_ds(X_train, y_train, shuffle=True,  augment=True)
ds_val   = make_ds(X_val,   y_val,   shuffle=False, augment=False)
ds_test  = make_ds(X_test,  y_test,  shuffle=False, augment=False)

# ==== 5) Model B (Conv-BN-ReLU blocks + Dropout) ====
def build_model(num_classes:int)->tf.keras.Model:
    inp = tf.keras.Input(shape=(*IMG_SIZE,3))
    x = tf.keras.layers.Conv2D(32,3,padding="same",use_bias=False)(inp); x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x); x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64,3,padding="same",use_bias=False)(x);  x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x); x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128,3,padding="same",use_bias=False)(x); x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x); x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    return tf.keras.Model(inp, out, name="model_B_compact_scientific")

# ==== 6) (Optional) tiny LR sweep for tuning ====
def lr_sweep(lrs):
    best_lr, best_val = None, -1
    for lr in lrs:
        m = build_model(len(class_names))
        m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = m.fit(ds_train, validation_data=ds_val, epochs=5, verbose=0)
        cur = max(h.history["val_accuracy"])
        print(f"[LR SWEEP] lr={lr:.5f} → best val_acc={cur:.4f}")
        if cur > best_val: best_val, best_lr = cur, lr
    print(f"[LR SWEEP] selected lr={best_lr}")
    return best_lr

best_lr = lr_sweep([BASE_LR/3, BASE_LR, BASE_LR*3])

# ==== 7) Train final model ====
model = build_model(len(class_names))
model.compile(optimizer=tf.keras.optimizers.Adam(best_lr),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# optional class weights if imbalance >15%
y_train_np = np.asarray(y_train, dtype=int)
counts = np.bincount(y_train_np, minlength=len(class_names))
print("Train counts per class:", {class_names[i]: int(c) for i, c in enumerate(counts)})

dev = np.max(np.abs(counts - counts.mean()) / np.maximum(counts.mean(), 1e-8))
class_weight = None

# Calcola pesi SOLO se davvero sbilanciato e se tutte le classi sono presenti
labels_present = np.unique(y_train_np)
if dev > 0.15 and len(labels_present) == len(class_names):
    w = compute_class_weight(
        class_weight="balanced",
        classes=labels_present,   # usa SOLO le classi presenti
        y=y_train_np
    )
    class_weight = {int(c): float(wi) for c, wi in zip(labels_present, w)}
    print("Using class weights:", class_weight)
elif dev > 0.15:
    print("⚠️ Imbalance rilevato ma manca almeno una classe nel train: salto class_weight.")

ckpt = (OUT_DIR/"model_B_best.keras").as_posix()
cbs = [
    tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]
hist = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=cbs,
                 class_weight=class_weight, verbose=1)

final_path = (OUT_DIR/"model_B_final.keras").as_posix()
model.save(final_path)
print(f"Saved best to: {ckpt}\nSaved final to: {final_path}")

# ==== 8) Plots ====
plt.figure(); plt.plot(hist.history["loss"]); plt.plot(hist.history["val_loss"])
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (train/val)"); plt.legend(["train","val"])
plt.tight_layout(); plt.savefig((OUT_DIR/"training_loss.png").as_posix()); plt.close()

plt.figure(); plt.plot(hist.history["accuracy"]); plt.plot(hist.history["val_accuracy"])
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (train/val)"); plt.legend(["train","val"])
plt.tight_layout(); plt.savefig((OUT_DIR/"training_accuracy.png").as_posix()); plt.close()

# ==== 9) Test + report ====
test_loss, test_acc = model.evaluate(ds_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

y_true, y_pred = [], []
for x,y in ds_test:
    p = model.predict(x, verbose=0)
    y_true.extend(y.numpy()); y_pred.extend(np.argmax(p, axis=1))
rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
with open(OUT_DIR/"classification_report.txt","w",encoding="utf-8") as f:
    f.write("Classification Report (Model B compact + scientific)\n"); f.write(rep)

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix (Model B)")
plt.colorbar(); ticks=np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha="right"); plt.yticks(ticks, class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,str(cm[i,j]),ha="center",va="center")
plt.tight_layout(); plt.savefig((OUT_DIR/"confusion_matrix.png").as_posix()); plt.close()

with open(OUT_DIR/"README_Model_B_compact_scientific.txt","w",encoding="utf-8") as f:
    f.write("Model B compact + scientific summary:\n"
            "- 128x128, per-channel standardization (train mean/std), moderate aug.\n"
            "- 3x [Conv-BN-ReLU-MaxPool] → Flatten → Dropout → Dense(128) → Dropout → Softmax.\n"
            "- EarlyStopping, ReduceLROnPlateau, checkpoint; tiny LR sweep.\n")
print("All outputs saved in:", OUT_DIR.resolve())