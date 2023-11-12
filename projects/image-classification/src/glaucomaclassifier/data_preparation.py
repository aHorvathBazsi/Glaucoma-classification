import os
import cv2
import pandas as pd
from imagedatahandler.data_sampling import sample_data_custom_ratio_per_class
from imagedatahandler.image_operations import remove_padding, save_image

def prepare_image_data():
    label_df = pd.read_csv("train_labels.csv")
    image_dir = os.path.join(os.getcwd(), "image-data")
    sample_per_class_dict = {"NRG": 5000, "RG": 2500}

    sampled_dataframe = sample_data_custom_ratio_per_class(
        dataframe=label_df,
        sample_per_class_dict=sample_per_class_dict
    )
    sampled_dataframe.to_csv("sampled_train_labels.csv", index=False)

    for image_id in sampled_dataframe["challenge_id"].tolist():
        image_filename = f"{image_id}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        cropped_images = remove_padding(cv2.imread(image_path))
        save_image(
            image=cropped_images,
            save_path=os.path.join(os.getcwd(), "sampled-image-data", image_filename)
        )
if __name__ == "__main__":
    prepare_image_data()