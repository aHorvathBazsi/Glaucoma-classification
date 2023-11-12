import glob
import os

import pandas as pd
from constants import (CLASS_NAME_ID_MAP, IMAGE_COLUMN_IDX,
                       IMAGE_FILE_EXTENSION, LABEL_COLUMN_IDX)
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CustomImageDataset(Dataset):
    def __init__(
        self,
        label_dataframe: pd.DataFrame,
        image_directory: str,
        image_transform: transforms.Compose = None,
        label_transform: transforms.Compose = None,
        image_file_extension: str = IMAGE_FILE_EXTENSION,
    ):
        self.label_dataframe = label_dataframe
        self.image_directory = image_directory
        self.image_file_extension = image_file_extension
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.__validate_image_data()
        self.__validate_label_data()

    def __validate_image_data(self):
        image_filenames_from_dataframe = self.label_dataframe.iloc[
            :, IMAGE_COLUMN_IDX
        ].tolist()
        image_paths = glob.glob(
            os.path.join(self.image_directory, f"*{self.image_file_extension}")
        )

        image_filenames_no_extension = [
            os.path.splitext(os.path.basename(image_filename))[0]
            for image_filename in image_paths
        ]

        if not set(image_filenames_from_dataframe).issubset(
            set(image_filenames_no_extension)
        ):
            raise ValueError(
                "Image filenames in dataframe is not a subset of image filenames in image directory"
            )

    def __validate_label_data(self):
        label_data = self.label_dataframe.iloc[:, LABEL_COLUMN_IDX].tolist()
        if not set(label_data).issubset(set(CLASS_NAME_ID_MAP.keys())):
            raise ValueError(
                "Label data is not a subset of class names"
            )

    def __len__(self):
        return len(self.label_dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.image_directory,
            self.label_dataframe.iloc[idx, IMAGE_COLUMN_IDX]
            + self.image_file_extension,
        )
        image = Image.open(img_path)
        label = self.label_dataframe.iloc[idx, LABEL_COLUMN_IDX]
        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label

if __name__ == "__main__":

    label_dataframe = pd.read_csv("sampled_train_labels.csv")
    image_directory = os.path.join(os.getcwd(), "sampled-image-data")
    dataset = CustomImageDataset(
        label_dataframe=label_dataframe, image_directory=image_directory
    )

    print(dataset[0])
