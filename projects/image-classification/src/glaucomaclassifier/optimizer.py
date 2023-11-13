from enum import Enum
from typing import Any

import torch.optim as optim


class OptimizerType(Enum):
    ADAM = "Adam"
    ADAMW = "AdamW"


def get_optimizer(
    optimizer_type: str, parameters: Any, base_learning_rate: float, weight_decay: float
) -> optim.Optimizer:
    if optimizer_type == OptimizerType.ADAM.value:
        return optim.Adam(parameters, lr=base_learning_rate, weight_decay=weight_decay)
    elif optimizer_type == OptimizerType.ADAMW.value:
        return optim.AdamW(parameters, lr=base_learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. Please check enum values."
        )
