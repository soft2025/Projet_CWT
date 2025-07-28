# Projet CWT

This repository reorganizes the original notebook into reusable Python modules.

## Installation

Install the required Python packages with pip:

```bash
pip install -r requirements.txt
```

You can also perform an offline installation using the provided
`setup.sh` script:

```bash
./setup.sh
```

This script installs packages from wheel files located in the `wheels/`
directory. Make sure all the required wheels are available in this folder
before running it.

After installation, run the test suite to verify the environment:

```bash
pytest
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


### Pretrained denoising model

The dataset generation utilities rely on a small denoising network whose
weights are **not** tracked in this repository.  To use the provided scripts you
need the file `checkpoints/denoising_model.pth`.

You can download it from the project's release page:

```bash
bash scripts/download_denoising_model.sh
```

This script downloads the checkpoint from the release and stores it in the
`checkpoints/` directory.  If the file is missing, code that uses
`load_denoising_model` will fall back to an untrained network and print a
warning.

