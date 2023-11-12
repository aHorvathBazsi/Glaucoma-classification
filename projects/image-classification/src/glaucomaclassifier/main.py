import os

import pandas as pd
import torch
from torch import optim
from torch import nn

from constants import SAMPLED_IMAGE_DIR, SAMPLED_LABEL_CSV
from glaucomaclassifier.dataloader import get_data_loader
from glaucomaclassifier.dataset import CustomImageDataset
from glaucomaclassifier.transforms import (get_image_transform,
                                           get_label_transform)
from glaucomaclassifier.models import get_model
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

    train_data_loader = get_data_loader(dataset=train_dataset, batch_size=32, is_train=True)
    val_data_loader = get_data_loader(dataset=val_dataset, batch_size=32, is_train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name="deit", num_classes=2, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)


if __name__ == "__main__":
    main()
