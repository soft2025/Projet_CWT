#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(dirname "$0")

pip install --no-index --find-links="$SCRIPT_DIR/wheels" \
    numpy torch torchvision timm scikit-learn matplotlib scikit-image \
    pywavelets scipy seaborn pytest
