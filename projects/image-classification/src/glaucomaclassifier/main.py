import torch
from glaucomaclassifier.dataloader import get_data_loaders
from glaucomaclassifier.models import get_model
from glaucomaclassifier.train import train_model
from torch import nn, optim


def run_training(
    batch_size=32,
    train_val_ratio=0.8,
    max_rotation_angle=20,
    num_epochs=10,
    use_weighted_sampler=True,
    unfreeze_head=True,
    unfreeze_blocks_number=0
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_loader, val_data_loader, train_dataset_size, val_dataset_size, class_weigths = get_data_loaders(
        train_val_ratio=train_val_ratio,
        max_rotation_angle=max_rotation_angle,
        batch_size=batch_size,
        use_weighted_sampler=use_weighted_sampler
    )
    
    if not use_weighted_sampler:
        class_weigths = torch.FloatTensor(class_weigths).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weigths)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    model, trainable_parameters = get_model(
        model_name="deit",
        num_classes=2,
        pretrained=True,
        unfreeze_head=unfreeze_head,
        unfreeze_blocks_number=unfreeze_blocks_number
    )
    model.to(device)

    optimizer = optim.Adam(trainable_parameters, lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=exp_lr_scheduler,
        dataloaders={"train": train_data_loader, "val": val_data_loader},
        dataset_sizes={"train": train_dataset_size, "val": val_dataset_size},
        device=device,
        num_epochs=num_epochs,
    )

if __name__ == "__main__":
    run_training()
