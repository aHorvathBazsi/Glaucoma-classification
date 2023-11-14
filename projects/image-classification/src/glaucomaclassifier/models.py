from enum import Enum

import timm
from glaucomaclassifier.constants import CLASS_NAME_ID_MAP


class VisionModelName(str, Enum):
    SWIN_TRANSFORMER = "timm/swin_tiny_patch4_window7_224.ms_in1k"
    DEIT = "timm/deit_tiny_patch16_224.fb_in1k"


def get_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    unfreeze_head: bool = False,
    unfreeze_blocks_number: int = 0,
):
    if num_classes != len(CLASS_NAME_ID_MAP):
        raise ValueError(
            f"Number of classes doesn't match settings with constant which is: {len(CLASS_NAME_ID_MAP)}"
        )
    model_enum = VisionModelName[model_name.upper()]
    model = timm.create_model(
        model_enum.value, pretrained=pretrained, num_classes=num_classes
    )

    if unfreeze_head:
        trainable_parameters = []
        # Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze and collect head parameters if requested
        head_params = []
        for param in model.head.parameters():
            param.requires_grad = True
            head_params.append(param)
        trainable_parameters.append({"params": head_params})

        # Unfreeze and collect block parameters if requested
        if unfreeze_blocks_number > 0 and model_name == "deit":
            block_params = []
            for block in model.blocks[-unfreeze_blocks_number:]:
                for param in block.parameters():
                    param.requires_grad = True
                    block_params.append(param)
            trainable_parameters.append({"params": block_params})
    else:
        trainable_parameters = model.parameters()

    return model, trainable_parameters


if __name__ == "__main__":
    model = get_model(
        model_name="swin_transformer",
        num_classes=2,
        pretrained=True,
        unfreeze_head=True,
        unfreeze_blocks_number=0,
    )
