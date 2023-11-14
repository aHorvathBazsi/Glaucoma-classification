import os
from copy import deepcopy

import numpy as np
import pandas as pd
import timm
import torch
from glaucomaclassifier.constants import ALL_IMAGE_DIR
from glaucomaclassifier.dataset import CustomImageDataset
from glaucomaclassifier.transforms import get_label_transform
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

KEEP_RATIO = 0.1
BATCH_SIZE = 32
SAMPLING_STEP = 32
TOTAL_BATCHES = 96
SAMPLE_PER_STEP = int((BATCH_SIZE * SAMPLING_STEP) * KEEP_RATIO)
TRAIN_LABELS_CSV = "train_labels.csv"
IMAGE_DATA_DIR = "image-data"
PRETRAINED_MODEL_NAME = "deit_tiny_patch16_224.fb_in1k"


def get_pretrained_model():
    model = timm.create_model(PRETRAINED_MODEL_NAME, pretrained=True, num_classes=0)
    return model


def get_image_transforms(model):
    data_config = timm.data.resolve_model_data_config(model)
    image_transforms = timm.data.create_transform(**data_config, is_training=False)
    return image_transforms


def sample_dissimilar_images(
    image_embeddings: torch.Tensor, num_samples: int, index_offset: int
):
    # Convert to numpy for compatibility with sklearn
    image_embeddings = image_embeddings.cpu().numpy()

    # Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(image_embeddings)
    # Invert the similarity matrix to represent dissimilarity
    dissimilarity_matrix = 1 - similarity_matrix
    # Average dissimilarity for each image
    average_dissimilarity = np.mean(dissimilarity_matrix, axis=1)

    # Select the indices of the 1000 most distinct images
    selected_indices = np.argsort(average_dissimilarity)[-num_samples:]

    return selected_indices + index_offset


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_pretrained_model()
    model = model.to(device)
    model.eval()

    image_transforms = get_image_transforms(model)
    label_transforms = get_label_transform()

    labels_dataframe = pd.read_csv("majority_class_labels.csv")
    majority_class_dataset = CustomImageDataset(
        label_dataframe=labels_dataframe,
        image_directory=os.path.join(THIS_DIR, ALL_IMAGE_DIR),
        image_transform=image_transforms,
        label_transform=label_transforms,
    )
    data_loader = DataLoader(
        majority_class_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    selected_ids = []
    with torch.no_grad():
        image_embeddings = None
        for batch_number, (inputs, labels) in tqdm(
            iterable=enumerate(data_loader), total=TOTAL_BATCHES + 1
        ):
            if batch_number > TOTAL_BATCHES:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model.forward_features(inputs)
            output = model.forward_head(output, pre_logits=True)
            if image_embeddings is not None:
                image_embeddings = torch.cat((image_embeddings, output), dim=0)
            else:
                image_embeddings = deepcopy(output)

            if not (batch_number + 1) % SAMPLING_STEP:
                selected_ids.extend(
                    sample_dissimilar_images(
                        image_embeddings=image_embeddings,
                        num_samples=SAMPLE_PER_STEP,
                        index_offset=(batch_number - SAMPLING_STEP + 1) * BATCH_SIZE,
                    )
                )
                image_embeddings = None
    sampled_labels_dataframe = labels_dataframe.iloc[selected_ids]
    sampled_labels_dataframe.reset_index(inplace=True, drop=True)
    sampled_labels_dataframe.to_csv("similarity_based_sampled_labels.csv", index=False)


if __name__ == "__main__":
    main()
