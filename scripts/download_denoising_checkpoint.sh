#!/usr/bin/env bash
# Simple helper script to download the denoising model checkpoint
# Usage: ./scripts/download_denoising_checkpoint.sh [DEST]
# DEST defaults to checkpoints/denoising_model.pth
set -euo pipefail
DEST="${1:-checkpoints/denoising_model.pth}"
URL="https://example.com/path/to/denoising_model.pth"
mkdir -p "$(dirname "$DEST")"
if command -v curl >/dev/null; then
    curl -L "$URL" -o "$DEST"
elif command -v wget >/dev/null; then
    wget "$URL" -O "$DEST"
else
    echo "Neither curl nor wget found. Please download $URL manually to $DEST" >&2
    exit 1
fi
printf 'Downloaded checkpoint to %s\n' "$DEST"

