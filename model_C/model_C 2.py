#!/usr/bin/env python3
"""
Model C (compact + scientific): transfer learning with EfficientNetB0
- P3: 224x224, stronger augmentation (train only), RandomResizedCrop-like
- Strict stratified split (sklearn) + fixed seeds
- Two phases: (1) train head with frozen backbone, (2) fine-tune top layers
- Saves: best (per phase) & final model, plots (phase 1 & 2), confusion matrix, classification report
"""

from pathlib import Path
import random, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

# ==== Config ====
DATA_DIR = Path("/Users/matteohasa/Desktop/ML-project1/rps-cv-images")
OUT_DIR  = Path("/Users/matteohasa/Desktop/ML-project/model_C/outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE = (224, 224); BATCH = 32
SEED = 42; TEST_RATIO = 0.15; VAL_RATIO = 0.15
EPOCHS_P1 = 10   # head
EPOCHS_P2 = 15   # fine-tune
LR_P1 = 1e-3; LR_P2 = 1e-4
AUTOTUNE = tf.data.AUTOTUNE

# ==== Seeds ====
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ==== 1) Scan & split (stratified) ====
exts = {".jpg",".jpeg",".png",".bmp",".gif"}
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
name2lab = {n:i for i,n in enumerate(class_names)}
files, labs = [], []
for cname in class_names:
    for f in (DATA_DIR/cname).rglob("*"):
        if f.suffix.lower() in exts:
            files.append(str(f)); labs.append(name2lab[cname])
if not files: raise RuntimeError(f"No images found in {DATA_DIR}")

X_tmp, X_test, y_tmp, y_test = train_test_split(files, labs, test_size=TEST_RATIO,
                                                stratify=labs, random_state=SEED)
val_rel = VAL_RATIO / (1.0 - TEST_RATIO)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_rel,
                                                  stratify=y_tmp, random_state=SEED)

# ==== 2) Augmentation & pipelines ====
aug_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
    layers.Resizing(256, 256),
    layers.RandomCrop(*IMG_SIZE),
], name="data_augmentation")

def _read_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, IMG_SIZE, antialias=True)
    return img

def _map_fn(path, label, augment=False):
    img = _read_resize(path)
    if augment:
        img = aug_layer(tf.expand_dims(img,0), training=True)[0]
    # EfficientNetB0 expects [0,1] inputs; it has internal normalization
    return img, label

def make_ds(paths, labels, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, np.array(labels, np.int32)))
    if shuffle: ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p,y: _map_fn(p,y,augment), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

ds_train = make_ds(X_train, y_train, shuffle=True,  augment=True)
ds_val   = make_ds(X_val,   y_val,   shuffle=False, augment=False)
ds_test  = make_ds(X_test,  y_test,  shuffle=False, augment=False)

# ==== 3) Build model (EfficientNetB0 backbone) ====
def build_model(num_classes:int, trainable_backbone:bool=False):
    inp = layers.Input(shape=(*IMG_SIZE,3))
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                    input_tensor=inp, pooling="avg")
    backbone.trainable = trainable_backbone
    x = backbone.output
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="model_C"), backbone

model, backbone = build_model(len(class_names), trainable_backbone=False)

# ==== 4) Phase 1: train head ====
model.compile(optimizer=tf.keras.optimizers.Adam(LR_P1),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
ckpt1 = (OUT_DIR/"model_C_phase1_best.keras").as_posix()
hist1 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_P1, verbose=1,
                  callbacks=[
                      tf.keras.callbacks.ModelCheckpoint(ckpt1, monitor="val_loss", save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", min_delta=0.005, patience=3, restore_best_weights=True, start_from_epoch=3, verbose=1),
                  ])

# ==== 5) Phase 2: fine-tune top layers ====
backbone.trainable = True
freeze_until = len(backbone.layers) - 50  # keep lower layers frozen
for l in backbone.layers[:max(freeze_until,0)]: l.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(LR_P2),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
ckpt2 = (OUT_DIR/"model_C_phase2_best.keras").as_posix()
hist2 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_P2, verbose=1,
                  callbacks=[
                      tf.keras.callbacks.ModelCheckpoint(ckpt2, monitor="val_loss", save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", mode="min",
                        min_delta=0.003,
                        patience=3,
                        restore_best_weights=True,
                        start_from_epoch=3,
                        verbose=1
                        ),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                  ])

# ==== 6) Save final model ====
final_path = (OUT_DIR/"model_C_final.keras").as_posix()
model.save(final_path)
print(f"Saved phase1 best to: {ckpt1}\nSaved phase2 best to: {ckpt2}\nSaved final to: {final_path}")

# ==== 7) Plots (phase 1 & 2) ====
def plot_hist(h, tag):
    plt.figure(); plt.plot(h.history["loss"]); plt.plot(h.history["val_loss"])
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{tag} Loss"); plt.legend(["train","val"])
    plt.tight_layout(); plt.savefig((OUT_DIR/f"{tag.replace(' ','_').lower()}_loss.png").as_posix()); plt.close()

    plt.figure(); plt.plot(h.history["accuracy"]); plt.plot(h.history["val_accuracy"])
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{tag} Accuracy"); plt.legend(["train","val"])
    plt.tight_layout(); plt.savefig((OUT_DIR/f"{tag.replace(' ','_').lower()}_accuracy.png").as_posix()); plt.close()

plot_hist(hist1, "Phase 1"); plot_hist(hist2, "Phase 2")

# ==== 8) Test + report ====
test_loss, test_acc = model.evaluate(ds_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

y_true, y_pred = [], []
for x,y in ds_test:
    p = model.predict(x, verbose=0)
    y_true.extend(y.numpy()); y_pred.extend(np.argmax(p, axis=1))
rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
with open(OUT_DIR/"classification_report_final.txt","w",encoding="utf-8") as f:
    f.write("Classification Report (Model C compact + scientific - final)\n"); f.write(rep)

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix (Model C - final)")
plt.colorbar(); ticks=np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha="right"); plt.yticks(ticks, class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,str(cm[i,j]),ha="center",va="center")
plt.tight_layout(); plt.savefig((OUT_DIR/"confusion_matrix_final.png").as_posix()); plt.close()

with open(OUT_DIR/"README_Model_C.txt","w",encoding="utf-8") as f:
    f.write("Model C compact + scientific summary:\n"
            "- 224x224, stronger augmentation (flip/rotation/translation/zoom/contrast + random crop).\n"
            "- EfficientNetB0 (ImageNet) → head training → fine-tune top layers.\n"
            "- EarlyStopping, ReduceLROnPlateau, checkpoint per fase.\n")
print("All outputs saved in:", OUT_DIR.resolve())