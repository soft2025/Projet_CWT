import timm
import torch
from torch import nn

from .cbam import CBAM


def get_swin_tiny_partial_finetune(num_classes: int = 4, pretrained: bool = True,
                                   unfreeze_stages=(3,)):
    """Return a Swin-T model where specific stages are unfrozen for fine-tuning."""
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=pretrained,
        num_classes=num_classes,
    )
    model.num_classes = num_classes
    for param in model.parameters():
        param.requires_grad = False
    for stage_idx in unfreeze_stages:
        if 0 <= stage_idx < len(model.layers):
            stage = model.layers[stage_idx]
            for param in stage.parameters():
                param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True
    return model


class SwinWithAttention(nn.Module):
    """Swin-Tiny backbone followed by CBAM attention and a classifier."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.cbam = CBAM(self.backbone.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
