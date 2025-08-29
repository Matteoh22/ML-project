#!/usr/bin/env python3
"""
Model C (compact + scientific, FIXED):
- P3: 224x224, aug forte (solo train), RandomCrop-like
- Split stratificato (sklearn) + seed fissi
- EfficientNetB0 con normalizzazione INCAPSULATA nel modello:
    preprocess_input(x * 255) via Lambda → coerenza train/eval
- Training in 2 fasi: head → fine-tuning (ultimi blocchi)
- Salva: class_names.json, best per fase, final; grafici; report; confusion matrix
- Valutazione: riallinea le classi usando class_names.json per evitare colonne “sfasate”
"""

from pathlib import Path
import random, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import efficientnet

# ==== CONFIG ====
DATA_DIR = Path("/Users/matteohasa/Desktop/ML-project/rps-cv-images")
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
    # output rimane [0,1]; la normalizzazione vera è dentro al modello (Lambda)
    return img, label

def make_ds(paths, labels, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, np.array(labels, np.int32)))
    if shuffle: ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p,y: _map_fn(p,y,augment), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

ds_train = make_ds(X_train, y_train, shuffle=True,  augment=True)
ds_val   = make_ds(X_val,   y_val,   shuffle=False, augment=False)
ds_test  = make_ds(X_test,  y_test,  shuffle=False, augment=False)

# ==== 3) Build model (EfficientNetB0 backbone) with IN-MODEL preprocessing ====
def build_model(num_classes:int, trainable_backbone:bool=False):
    inp = layers.Input(shape=(*IMG_SIZE,3))
    # Incapsulo la normalizzazione: da [0,1] → preprocess_input (che si aspetta 0..255)
    x = layers.Lambda(lambda t: efficientnet.preprocess_input(t * 255.0), name="effnet_preprocess")(inp)
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                    input_tensor=x, pooling="avg")
    backbone.trainable = trainable_backbone
    x = backbone.output
    x = layers.Dropout(0.5)(x)   # un po' più alto per robustezza
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="model_C_compact_scientific_fixed"), backbone

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
freeze_until = len(backbone.layers) - 50  # fine-tune solo gli ultimi ~50 layers
for l in backbone.layers[:max(freeze_until,0)]: l.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(LR_P2),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
ckpt2 = (OUT_DIR/"model_C_phase2_best.keras").as_posix()
hist2 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_P2, verbose=1,
                  callbacks=[
                      tf.keras.callbacks.ModelCheckpoint(ckpt2, monitor="val_loss", save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", mode="min",
                        min_delta=0.002,
                        patience=3,
                        restore_best_weights=True,
                        start_from_epoch=3,
                        verbose=1
                        ),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                  ])

# ==== 6) Save final model + class order ====
final_path = (OUT_DIR/"model_C_final.keras").as_posix()
model.save(final_path)
with open(OUT_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f)
print(f"Saved phase1 best to: {ckpt1}\nSaved phase2 best to: {ckpt2}\nSaved final to: {final_path}")
print(f"Saved class names to: {OUT_DIR/'class_names.json'}")

# ==== 7) Plots (phase 1 & 2) ====
def plot_hist(h, tag):
    plt.figure(); plt.plot(h.history["loss"]); plt.plot(h.history["val_loss"])
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{tag} Loss"); plt.legend(["train","val"])
    plt.tight_layout(); plt.savefig((OUT_DIR/f"{tag.replace(' ','_').lower()}_loss.png").as_posix()); plt.close()

    plt.figure(); plt.plot(h.history["accuracy"]); plt.plot(h.history["val_accuracy"])
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{tag} Accuracy"); plt.legend(["train","val"])
    plt.tight_layout(); plt.savefig((OUT_DIR/f"{tag.replace(' ','_').lower()}_accuracy.png").as_posix()); plt.close()

plot_hist(hist1, "Phase 1"); plot_hist(hist2, "Phase 2")

# ==== 8) Test + report (con riallineamento classi da JSON) ====
# Carico class_names del modello (appena salvato) e costruisco indice di riordino
with open(OUT_DIR / "class_names.json", "r", encoding="utf-8") as f:
    model_class_names = json.load(f)

eval_class_names = class_names  # lo split è stato creato dal medesimo elenco; comunque creiamo la mappa
name2idx_model = {n:i for i,n in enumerate(model_class_names)}
reorder_idx = np.array([name2idx_model[n] for n in eval_class_names], dtype=int)

# Predizioni test
test_loss, test_acc = model.evaluate(ds_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

y_true, y_pred = [], []
for x, y in ds_test:
    # x è [0,1]; la normalizzazione corretta è INSIDE il modello
    probs = model.predict(x, verbose=0)
    # (qui l'ordine è già coerente col modello; eval_class_names == class_names,
    #  ma se vuoi essere ultra-robusto, potresti applicare probs = probs[:, reorder_idx])
    pred = np.argmax(probs, axis=1)
    y_true.extend(y.numpy()); y_pred.extend(pred)

# Report
report = classification_report(y_true, y_pred, target_names=eval_class_names, digits=4)
with open(OUT_DIR/"classification_report_final.txt","w",encoding="utf-8") as f:
    f.write("Classification Report (Model C compact + scientific - FIXED)\n"); f.write(report)
print("\nClassification report:\n", report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(eval_class_names))))
plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix (Model C - final)")
plt.colorbar(); ticks=np.arange(len(eval_class_names))
plt.xticks(ticks, eval_class_names, rotation=45, ha="right"); plt.yticks(ticks, eval_class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,str(cm[i,j]),ha="center",va="center")
plt.tight_layout(); plt.savefig((OUT_DIR/"confusion_matrix_final.png").as_posix()); plt.close()

# ==== 9) README ====
with open(OUT_DIR/"README_Model_C_compact_scientific_FIXED.txt","w",encoding="utf-8") as f:
    f.write(
        "Model C (compact+scientific) — FIXED normalization & class order:\n"
        "- In-model preprocessing: EfficientNet preprocess_input(x*255) via Lambda.\n"
        "- Saved class_names.json to lock class order across train/eval.\n"
        "- Two-phase training: head (frozen) → fine-tune top layers.\n"
        "- Outputs: best checkpoints, final model, training plots, confusion matrix, classification report.\n"
    )
print("All outputs saved in:", OUT_DIR.resolve())