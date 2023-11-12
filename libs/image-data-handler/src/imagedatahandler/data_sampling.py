import pandas as pd
from imagedatahandler.constants import LABEL_COLUMN_NAME


def sample_data_balanced_per_class(dataframe, sample_per_class: int):
    if not isinstance(sample_per_class, int):
        raise ValueError("sample_per_class must be an integer")
    return dataframe.groupby(LABEL_COLUMN_NAME).apply(
        lambda x: x.sample(min(sample_per_class, len(x)))
    )


def sample_data_custom_ratio_per_class(
    dataframe, sample_per_class_dict: dict[str, int]
):
    sampled_dataframe = pd.DataFrame(columns=dataframe.columns)

    if not isinstance(sample_per_class_dict, dict):
        raise ValueError(
            "sample_per_class must be a dictionary with class names as keys and sample sizes as values."
        )
    for class_name, class_dataframe in dataframe.groupby(LABEL_COLUMN_NAME):
        if class_name not in sample_per_class_dict:
            raise ValueError(f"Class {class_name} not found in sample_per_class_dict")
        sample_size = min(sample_per_class_dict[class_name], len(class_dataframe))
        sampled_dataframe = pd.concat(
            objs=[class_dataframe.sample(sample_size), sampled_dataframe]
        )
    return sampled_dataframe

def split_label_dataframe(label_dataframe: pd.DataFrame, fraction: float = 0.8, random_state: int = 42):
    train_data = pd.concat(
        [grouped_dataframe.sample(frac=fraction, random_state=random_state) for _, grouped_dataframe in label_dataframe.groupby(LABEL_COLUMN_NAME)]
    )
    val_data = label_dataframe.drop(train_data.index)
    return train_data, val_data
