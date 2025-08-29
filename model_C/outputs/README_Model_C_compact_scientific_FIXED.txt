Model C (compact+scientific) — FIXED normalization & class order:
- In-model preprocessing: EfficientNet preprocess_input(x*255) via Lambda.
- Saved class_names.json to lock class order across train/eval.
- Two-phase training: head (frozen) → fine-tune top layers.
- Outputs: best checkpoints, final model, training plots, confusion matrix, classification report.
