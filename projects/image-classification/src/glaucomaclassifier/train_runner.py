import torch
import os
from torch import nn, optim

from glaucomaclassifier.dataloader import get_data_loaders
from glaucomaclassifier.models import get_model
from glaucomaclassifier.optimizer import get_optimizer
from glaucomaclassifier.train import train_model
from glaucomaclassifier.training_run_config import TrainingRunConfig

def run_training(config: TrainingRunConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        train_data_loader,
        val_data_loader,
        train_dataset_size,
        val_dataset_size,
        class_weigths,
    ) = get_data_loaders(
        train_val_ratio=config.train_val_ratio,
        max_rotation_angle=config.max_rotation_angle,
        batch_size=config.batch_size,
        use_weighted_sampler=config.use_weighted_sampler,
    )

    if not config.use_weighted_sampler:
        class_weigths = torch.FloatTensor(class_weigths).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weigths)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    model, trainable_parameters = get_model(
        model_name=config.model_name,
        num_classes=2,
        pretrained=True,
        unfreeze_head=config.unfreeze_head,
        unfreeze_blocks_number=config.unfreeze_blocks_number,
    )
    model.to(device)

    optimizer = get_optimizer(
        optimizer_type=config.optimizer_type,
        parameters=trainable_parameters,
        base_learning_rate=config.base_learning_rate,
        weight_decay=config.weight_decay,
    )
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.step_size, gamma=config.gamma
    )

    model = train_model(
        training_run_config=config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=exp_lr_scheduler,
        dataloaders={"train": train_data_loader, "val": val_data_loader},
        dataset_sizes={"train": train_dataset_size, "val": val_dataset_size},
        device=device,
        num_epochs=config.num_epochs,
        wandb_track_enabled=True,
    )
    # torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    training_config = TrainingRunConfig(
        run_name="deit-test-no-tuning",
        model_name="deit",
        batch_size=32,
        train_val_ratio=0.8,
        max_rotation_angle=20,
        num_epochs=2,
        use_weighted_sampler=True,
        unfreeze_head=True,
        unfreeze_blocks_number=1,
        optimizer_type="Adam",
        base_learning_rate=0.001,
        weight_decay=0.01,
        step_size=3,
        gamma=0.97,
    )
    run_training(config=training_config)
