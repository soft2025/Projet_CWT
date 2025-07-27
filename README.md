# Projet CWT

This repository reorganizes the original notebook into reusable Python modules.

## Usage

Prepare your dataset of CWT images arranged in sub‑folders for each class. Then
run the training script:

```bash
python main.py --data-dir path/to/images_cwt_224x224_rgb --epochs 25
```

The script will train a Swin Transformer on the dataset and evaluate it on a
held‑out test set.
