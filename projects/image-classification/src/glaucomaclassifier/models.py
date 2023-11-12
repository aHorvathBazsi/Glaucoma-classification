from enum import Enum

import timm

class VisionModelName(str, Enum):
    SWIN_TRANSFORMER = "timm/swin_small_patch4_window7_224.ms_in22k_ft_in1k"
    DEIT = "timm/deit_small_patch16_224.fb_in1k"


def get_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    model_enum = VisionModelName[model_name.upper()]
    return timm.create_model(model_enum.value, pretrained=pretrained, num_classes=num_classes)
