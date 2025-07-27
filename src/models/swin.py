import timm


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
