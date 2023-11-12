from enum import Enum

import timm

class VisionModelName(str, Enum):
    SWIN_TRANSFORMER = "timm/swin_small_patch4_window7_224.ms_in22k_ft_in1k"
    DEIT = "timm/deit_tiny_patch16_224.fb_in1k"


def get_model(model_name: str, num_classes: int = 2, pretrained: bool = True, unfreeze_head: bool = False, unfreeze_blocks_number: int = 0):
    model_enum = VisionModelName[model_name.upper()]
    model = timm.create_model(model_enum.value, pretrained=pretrained, num_classes=num_classes)

    trainable_parameters = []

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze and collect head parameters if requested
    if unfreeze_head:
        head_params = []
        for param in model.head.parameters():
            param.requires_grad = True
            head_params.append(param)
        trainable_parameters.append({'params': head_params})

    # Unfreeze and collect block parameters if requested
    if unfreeze_blocks_number > 0:
        block_params = []
        for block in model.blocks[-unfreeze_blocks_number:]:
            for param in block.parameters():
                param.requires_grad = True
                block_params.append(param)
        trainable_parameters.append({'params': block_params})

    return model, trainable_parameters
