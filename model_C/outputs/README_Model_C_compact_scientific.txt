Model C compact + scientific summary:
- 224x224, stronger augmentation (flip/rotation/translation/zoom/contrast + random crop).
- EfficientNetB0 (ImageNet) → head training → fine-tune top layers.
- EarlyStopping, ReduceLROnPlateau, checkpoint per fase.
