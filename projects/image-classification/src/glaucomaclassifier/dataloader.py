import os

import pandas as pd
from glaucomaclassifier.constants import (
    INPUT_SIZE,
    SAMPLED_IMAGE_DIR,
    SAMPLED_LABEL_CSV,
)
from glaucomaclassifier.dataset import CustomImageDataset
from glaucomaclassifier.transforms import get_image_transform, get_label_transform
from imagedatahandler.data_sampling import split_label_dataframe
from torch.utils.data import DataLoader, WeightedRandomSampler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_data_loaders(
    train_val_ratio=0.8,
    max_rotation_angle=20,
    batch_size=32,
    use_weighted_sampler=False,
):
    label_dataframe = pd.read_csv(
        filepath_or_buffer=os.path.join(THIS_DIR, SAMPLED_LABEL_CSV)
    )
    image_directory = os.path.join(THIS_DIR, SAMPLED_IMAGE_DIR)
    train_label_dataframe, val_label_dataframe = split_label_dataframe(
        label_dataframe=label_dataframe, fraction=train_val_ratio
    )

    train_image_transform = get_image_transform(
        is_train=True, input_size=INPUT_SIZE, max_rotation_angle=max_rotation_angle
    )
    val_image_transform = get_image_transform(
        is_train=False, input_size=INPUT_SIZE, max_rotation_angle=max_rotation_angle
    )
    label_transform = get_label_transform()

    train_dataset = CustomImageDataset(
        label_dataframe=train_label_dataframe,
        image_directory=image_directory,
        image_transform=train_image_transform,
        label_transform=label_transform,
    )
    train_dataset_size = len(train_dataset)

    val_dataset = CustomImageDataset(
        label_dataframe=val_label_dataframe,
        image_directory=image_directory,
        image_transform=val_image_transform,
        label_transform=label_transform,
    )
    val_dataset_size = len(val_dataset)

    train_loader_options = {"batch_size": batch_size, "shuffle": True}
    if use_weighted_sampler:
        train_dataset_size = (
            train_dataset.label_dataframe["encoded_label"].value_counts().min().item()
            * 2
        )
        sample_weights = [
            train_dataset.normalized_weights[label]
            for label in train_dataset.label_dataframe["encoded_label"]
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=train_dataset_size, replacement=False
        )
        train_loader_options["sampler"] = sampler
        train_loader_options["shuffle"] = False  # Disable shuffle when using a sampler

    train_data_loader = DataLoader(train_dataset, **train_loader_options)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    class_weigths = None if use_weighted_sampler else train_dataset.normalized_weights

    return (
        train_data_loader,
        val_data_loader,
        train_dataset_size,
        val_dataset_size,
        class_weigths,
    )
