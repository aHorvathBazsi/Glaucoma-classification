import os

import cv2
import pandas as pd
from imagedatahandler.data_sampling import sample_data_custom_ratio_per_class
from imagedatahandler.image_operations import remove_padding, save_image

from glaucomaclassifier.constants import (
    ALL_IMAGE_DIR,
    ALL_LABEL_CSV,
    IMAGE_COLUMN_IDX,
    IMAGE_FILE_EXTENSION,
    SAMPLE_PER_CLASS_DICT,
    SAMPLED_IMAGE_DIR,
    SAMPLED_LABEL_CSV,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_image_data():
    label_df = pd.read_csv(ALL_LABEL_CSV)
    image_dir = os.path.join(THIS_DIR, ALL_IMAGE_DIR)

    sampled_dataframe = sample_data_custom_ratio_per_class(
        dataframe=label_df, sample_per_class_dict=SAMPLE_PER_CLASS_DICT
    )
    sampled_dataframe.to_csv(SAMPLED_LABEL_CSV, index=False)

    for image_id in sampled_dataframe.iloc[:, IMAGE_COLUMN_IDX].tolist():
        image_filename = f"{image_id}{IMAGE_FILE_EXTENSION}"
        image_path = os.path.join(image_dir, image_filename)
        cropped_images = remove_padding(cv2.imread(image_path))
        save_image(
            image=cropped_images,
            save_path=os.path.join(THIS_DIR, SAMPLED_IMAGE_DIR, image_filename),
        )


if __name__ == "__main__":
    prepare_image_data()
