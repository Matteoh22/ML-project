#!/usr/bin/env python3
"""
Model B (intermediate, compact):
- P2: 128x128, normalize with mean/std, moderate augmentation
- CNN: Conv(32) → Conv(64) → MaxPool → Dropout →
       Conv(128) → MaxPool → Flatten → Dense(128) → Softmax(3)
- Split stratificato (train/val/test)
- Output: best/final model + plots + report + confusion matrix
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import random

# ==== CONFIG ====
DATA_DIR = Path("/Users/matteohasa/Desktop/ML-project/rps-cv-images")
OUT_DIR  = Path("/Users/matteohasa/Desktop/ML-project/outputs/model_B")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH = 32
SEED = 42
EPOCHS = 20
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==== 1) Dataset: collect paths + labels ====
exts = {".jpg",".jpeg",".png"}
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
cls2lab = {c:i for i,c in enumerate(classes)}

X, y = [], []
for cls in classes:
    for f in (DATA_DIR/cls).rglob("*"):
        if f.suffix.lower() in exts:
            X.append(str(f)); y.append(cls2lab[cls])
X = np.array(X); y = np.array(y)

# Stratified split
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, stratify=y, random_state=SEED
)
val_rel = VAL_RATIO / (1.0 - TEST_RATIO)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=SEED
)

# ==== 2) Precompute mean/std on train set ====
def read_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32) # [0,1]
    img = tf.image.resize(img, IMG_SIZE)
    return img

imgs = np.stack([read_img(p).numpy() for p in X_train])
mean = imgs.mean(axis=(0,1,2))
std  = imgs.std(axis=(0,1,2))
print("Train mean:", mean, "std:", std)

# ==== 3) Pipeline builder ====
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def map_fn(path, label, train=False):
    img = read_img(path)
    if train:
        img = augment(tf.expand_dims(img,0), training=True)[0]
    img = (img - mean) / std
    return img, label

def make_ds(paths, labels, train=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p,y: map_fn(p,y,train), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

ds_train = make_ds(X_train, y_train, train=True)
ds_val   = make_ds(X_val, y_val)
ds_test  = make_ds(X_test, y_test)

# ==== 4) Model ====
inputs = layers.Input(shape=(*IMG_SIZE,3))
x = layers.Conv2D(32,3,activation="relu",padding="same")(inputs)
x = layers.Conv2D(64,3,activation="relu",padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(128,3,activation="relu",padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128,activation="relu")(x)
outputs = layers.Dense(len(classes), activation="softmax")(x)

model = models.Model(inputs, outputs, name="model_B_compact")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==== 5) Training ====
ckpt_path = (OUT_DIR/"model_B_best.keras").as_posix()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
]

hist = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=1)

# ==== 6) Save final ====
final_path = (OUT_DIR/"model_B_final.keras").as_posix()
model.save(final_path)

# ==== 7) Plots ====
plt.figure(); plt.plot(hist.history["loss"]); plt.plot(hist.history["val_loss"])
plt.legend(["train","val"]); plt.title("Loss"); plt.savefig(OUT_DIR/"loss.png"); plt.close()

plt.figure(); plt.plot(hist.history["accuracy"]); plt.plot(hist.history["val_accuracy"])
plt.legend(["train","val"]); plt.title("Accuracy"); plt.savefig(OUT_DIR/"accuracy.png"); plt.close()

# ==== 8) Eval ====
test_loss, test_acc = model.evaluate(ds_test, verbose=0)
print(f"\nTest loss={test_loss:.4f}, acc={test_acc:.4f}")

y_true, y_pred = [], []
for xb,yb in ds_test:
    p = model.predict(xb, verbose=0)
    y_true.extend(yb.numpy()); y_pred.extend(np.argmax(p,axis=1))

report = classification_report(y_true, y_pred, target_names=classes, digits=4)
with open(OUT_DIR/"classification_report.txt","w") as f: f.write(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(); plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix")
plt.xticks(range(len(classes)), classes, rotation=45); plt.yticks(range(len(classes)), classes)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,cm[i,j],ha="center",va="center")
plt.tight_layout(); plt.savefig(OUT_DIR/"confusion_matrix.png"); plt.close()