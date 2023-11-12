import os

import pandas as pd
import torch
from torch import optim
from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader

from constants import SAMPLED_IMAGE_DIR, SAMPLED_LABEL_CSV
from glaucomaclassifier.dataset import CustomImageDataset
from glaucomaclassifier.transforms import (get_image_transform,
                                           get_label_transform)
from glaucomaclassifier.models import get_model
from glaucomaclassifier.train import train_model
from imagedatahandler.data_sampling import split_label_dataframe

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_SIZE = (224, 224)

def main():
    label_dataframe = pd.read_csv(filepath_or_buffer=os.path.join(THIS_DIR, SAMPLED_LABEL_CSV))
    image_directory = os.path.join(THIS_DIR, SAMPLED_IMAGE_DIR)
    train_label_dataframe, val_label_dataframe = split_label_dataframe(label_dataframe=label_dataframe, fraction=0.8)

    train_image_transform = get_image_transform(is_train=True, input_size=INPUT_SIZE, max_rotation_angle=30)
    val_image_transform = get_image_transform(is_train=False, input_size=INPUT_SIZE, max_rotation_angle=30)
    label_transform = get_label_transform()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CustomImageDataset(
        label_dataframe=train_label_dataframe,
        image_directory=image_directory,
        image_transform=train_image_transform,
        label_transform=label_transform,
    )

    val_dataset = CustomImageDataset(
        label_dataframe=val_label_dataframe,
        image_directory=image_directory,
        image_transform=val_image_transform,
        label_transform=label_transform,
    )

    compensate_with_weighted_loss = False
    if compensate_with_weighted_loss:
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        class_weights = torch.FloatTensor(train_dataset.normalized_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # Use sampler for data balancing
        sample_weights = [train_dataset.normalized_weights[label] for label in train_dataset.label_dataframe['encoded_label']]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=train_dataset.label_dataframe['encoded_label'].value_counts().min().item() * 2,
            replacement=False
        )
        train_data_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        criterion = nn.CrossEntropyLoss()
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

    model = get_model(model_name="deit", num_classes=2, pretrained=True)
    model.to(device)

    transfer_learning = True
    if transfer_learning:
        # Freeze all parameters in the model
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the parameters in the head (classifier)
        for param in model.head.parameters():
            param.requires_grad = True

    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=exp_lr_scheduler,
        dataloaders={"train": train_data_loader, "val": val_data_loader},
        dataset_sizes={"train": len(train_dataset), "val": len(val_dataset)},
        device=device,
        num_epochs=10,
    )

if __name__ == "__main__":
    main()
