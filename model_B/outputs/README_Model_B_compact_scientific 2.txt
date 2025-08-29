Model B compact + scientific summary:
- 128x128, per-channel standardization (train mean/std), moderate aug.
- 3x [Conv-BN-ReLU-MaxPool] → Flatten → Dropout → Dense(128) → Dropout → Softmax.
- EarlyStopping, ReduceLROnPlateau, checkpoint; tiny LR sweep.
