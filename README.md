# Projet CWT

This repository reorganizes the original notebook into reusable Python modules.

## Installation

Install the required Python packages with pip:

```bash
pip install -r requirements.txt
```

## Usage

### Generate the CWT image dataset

If you start from the raw *hawk* measurements you can create the image dataset
using the CLI provided in `scripts/generate_dataset.py`:

```bash
python scripts/generate_dataset.py /path/to/hawk_data images_cwt_224x224_rgb
```

### Train the model

Prepare your dataset of CWT images arranged in sub‑folders for each class. Then
run the training script:

```bash
python main.py \
  --data-dir path/to/images_cwt_224x224_rgb \
  --output-dir training_runs/run1 \
  --epochs 25
```

The dataset directory must follow this structure:

```
images_cwt_224x224_rgb/
├── class_a/
│   ├── img1.png
│   └── ...
├── class_b/
│   └── ...
└── class_n/
    └── ...
```

The trained model is saved to `<output-dir>/model.pth`.

The script will train a Swin Transformer on the dataset and evaluate it on a
held‑out test set.

### Denoising model checkpoint

The dataset generation step relies on a pretrained denoising network. Download the checkpoint before running the scripts:

```bash
./scripts/download_denoising_checkpoint.sh
```

The script stores the file at `checkpoints/denoising_model.pth`. You can also download the file manually from `https://example.com/path/to/denoising_model.pth` and place it at that location.
