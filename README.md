# Projet CWT

This repository reorganizes the original notebook into reusable Python modules.

## Installation

Install the required Python packages with pip:

```bash
pip install -r requirements.txt
```

### Providing `hawk-data` locally

The `hawk-data` package is only needed when running
`scripts/generate_dataset.py`.  In an offline environment you can supply this
dependency in one of two ways:

1. **Git submodule**

   Add the repository as a submodule and install it in editable mode:

   ```bash
   git submodule add https://github.com/MDCHAMP/hawk-data external/hawk-data
   pip install -e external/hawk-data
   ```

2. **Pre‑built wheel**

   Build or download a wheel for `hawk-data`, place it under `wheels/` and run
   `setup.sh`.  The script will install the wheel along with the other local
   packages.

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

