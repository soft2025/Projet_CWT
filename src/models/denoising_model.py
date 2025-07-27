import os
import torch
from torch import nn


class SimpleDenoiser(nn.Module):
    """Tiny convolutional network for image denoising."""

    def __init__(self, channels: int = 1, features: int = 64, num_layers: int = 5) -> None:
        super().__init__()
        layers = [nn.Conv2d(channels, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def load_denoising_model(checkpoint_path: str = "checkpoints/denoising_model.pth") -> nn.Module:
    """Load and return the pretrained denoising model."""
    model = SimpleDenoiser()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "See README.md for download instructions."
        )

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model
