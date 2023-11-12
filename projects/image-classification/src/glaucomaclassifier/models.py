from enum import Enum

import timm


class VisionModelName(str, Enum):
    swin_transformer = "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k"
    deit = "timm/deit_base_patch16_224.fb_in1k"

def get_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    model_enum = VisionModelName(model_name)
    if "timm" in model_name:
        return timm.create_model(model_enum.value, pretrained=pretrained, num_classes=num_classes)
