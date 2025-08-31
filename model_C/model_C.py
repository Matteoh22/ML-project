import os, json, shutil
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from shutil import copy2

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, efficientnet

# Config
DATA_DIR = Path("/Users/matteohasa/Desktop/ML-project/rps-cv-images")
OUT_DIR  = Path("/Users/matteohasa/Desktop/ML-project/model_C/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH    = 32
SEED     = 42
TEST_RATIO = 0.15
VAL_RATIO  = 0.15
EPOCHS_P1  = 10     # phase 1: head
EPOCHS_P2  = 15     # phase 2: fine-tune
LR_P1      = 1e-3
LR_P2      = 1e-4
AUTOTUNE   = tf.data.AUTOTUNE

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# === Dataset scan & split ===
exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
name2lab = {n: i for i, n in enumerate(class_names)}

files, labs = [], []
for cname in class_names:
    for f in (DATA_DIR / cname).rglob("*"):
        if f.suffix.lower() in exts:
            files.append(str(f))
            labs.append(name2lab[cname])

print("Classes:", class_names, "| Total:", len(files))

# Stratified 70/15/15
X_tmp, X_test, y_tmp, y_test = train_test_split(
    files, labs, test_size=TEST_RATIO, stratify=labs, random_state=SEED
)
val_rel = VAL_RATIO / (1 - TEST_RATIO)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=SEED
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Data augumentation + pipelines
aug = tf.keras.Sequential([
    layers.Resizing(256, 256),
    layers.RandomCrop(*IMG_SIZE),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
])

def _read_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def _map_fn(path, label, augment=False):
    img = _read_img(path)
    if augment:
        img = aug(tf.expand_dims(img,0), training=True)[0]
    else:
        img = tf.image.resize(img, IMG_SIZE, antialias=True)
    return img, label

def make_ds(paths, labels, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, np.array(labels, np.int32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p,y: _map_fn(p,y,augment), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

ds_train = make_ds(X_train, y_train, shuffle=True,  augment=True)
ds_val   = make_ds(X_val,   y_val,   shuffle=False, augment=False)
ds_test  = make_ds(X_test,  y_test,  shuffle=False, augment=False)

# Build model
def build_model(num_classes, trainable_backbone=False):
    inp = layers.Input(shape=(*IMG_SIZE,3))
    x = efficientnet.preprocess_input(inp * 255.0)
    backbone = EfficientNetB0(include_top=False, weights="imagenet",
                              input_tensor=x, pooling="avg")
    backbone.trainable = trainable_backbone
    x = backbone.output
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out), backbone

model, backbone = build_model(len(class_names), trainable_backbone=False)

# Phase 1: train head
model.compile(optimizer=tf.keras.optimizers.Adam(LR_P1),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
ckpt1 = OUT_DIR / "model_C_phase1_best.keras"
hist1 = model.fit(ds_train, validation_data=ds_val,
                  epochs=EPOCHS_P1, verbose=1,
                  callbacks=[
                      tf.keras.callbacks.ModelCheckpoint(ckpt1, monitor="val_loss", save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.03, patience=3,
                                                       restore_best_weights=True, start_from_epoch=3)
                  ])

# Phase 2: fine-tune last layers
backbone.trainable = True
freeze_until = len(backbone.layers) - 50
for l in backbone.layers[:max(freeze_until,0)]:
    l.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(LR_P2),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
ckpt2 = OUT_DIR / "model_C_phase2_best.keras"
hist2 = model.fit(ds_train, validation_data=ds_val,
                  epochs=EPOCHS_P2, verbose=1,
                  callbacks=[
                      tf.keras.callbacks.ModelCheckpoint(ckpt2, monitor="val_loss", save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3,
                                                       restore_best_weights=True, start_from_epoch=3),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                           patience=3, min_lr=1e-6),
                  ])

# Save final
final_path = OUT_DIR / "model_C_final.keras"
model.save(final_path)
with open(OUT_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f)

# Plots
def plot_hist(hist, tag):
    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="val")
    plt.legend(); plt.title(f"{tag} Accuracy")
    plt.savefig(OUT_DIR / f"{tag}_acc.png"); plt.close()

    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.legend(); plt.title(f"{tag} Loss")
    plt.savefig(OUT_DIR / f"{tag}_loss.png"); plt.close()

plot_hist(hist1, "phase1")
plot_hist(hist2, "phase2")

# Test + Report + Confusion matrix
test_loss, test_acc = model.evaluate(ds_test, verbose=0)
print(f"Test acc = {test_acc:.4f}")

y_true, y_pred = [], []
for x, y in ds_test:
    probs = model.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(probs, axis=1))

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
with open(OUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix (Model C)")
plt.colorbar(); plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png"); plt.close()

print("Done! Outputs in:", OUT_DIR.resolve())