#!/usr/bin/env bash
# Download the pretrained denoising model used by dataset generation.
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
CHECKPOINT_DIR="$SCRIPT_DIR/../checkpoints"
mkdir -p "$CHECKPOINT_DIR"

URL="https://github.com/MDCHAMP/Projet_CWT/releases/download/v1.0/denoising_model.pth"
if [ ! -f "$CHECKPOINT_DIR/denoising_model.pth" ]; then
    echo "Downloading denoising model from $URL"
    curl -L "$URL" -o "$CHECKPOINT_DIR/denoising_model.pth"
    echo "Saved to $CHECKPOINT_DIR/denoising_model.pth"
else
    echo "Denoising model already exists at $CHECKPOINT_DIR/denoising_model.pth"
fi
