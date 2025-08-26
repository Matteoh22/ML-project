Model A (compact) with balanced splits:
- P1: 64x64, normalize [0,1], no augmentation
- Conv(16)-MaxPool-Flat-Dense(32)-Softmax
- Balanced train/val/test from original dataset
- Saved: model_A_best.keras, model_A_final.keras, plots, report
