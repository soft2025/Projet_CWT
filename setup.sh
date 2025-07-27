#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(dirname "$0")

pip install --no-index --find-links="$SCRIPT_DIR/wheels" \
    numpy torch torchvision timm scikit-learn matplotlib scikit-image \
    pywavelets scipy seaborn pytest

# install hawk-data if a wheel is provided
if ls "$SCRIPT_DIR"/wheels/hawk_data-*.whl > /dev/null 2>&1; then
    pip install --no-index --find-links="$SCRIPT_DIR/wheels" hawk_data
fi
