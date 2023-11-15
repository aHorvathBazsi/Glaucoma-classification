from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingRunConfig:
    run_name: str = "test_run"
    model_name: str = "deit"
    batch_size: int = 32
    train_val_ratio: float = 0.8
    max_rotation_angle: int = 20
    num_epochs: int = 15
    use_weighted_sampler: bool = True
    unfreeze_head: bool = True
    unfreeze_blocks_number: int = 1
    optimizer_type: str = "Adam"
    base_learning_rate: float = 0.001
    weight_decay: float = 0.01
    step_size: int = 3
    gamma: float = 0.97

    def __post_init__(self):
        if self.optimizer_type not in ["Adam", "AdamW"]:
            raise ValueError(
                f"Unknown optimizer type: {self.optimizer_type}. Please check enum values in optimizer.py"
            )
        if not self.unfreeze_head and self.unfreeze_blocks_number != 0:
            raise ValueError(
                "If head is frozen, then unfreeze_blocks_number should be 0."
            )
        if self.model_name not in ["deit", "swin_transformer"]:
            raise ValueError(
                f"Unknown model name: {self.model_name}. Please check enum values in models.py"
            )
