import pandas as pd


def sample_data_balanced_per_class(dataframe, sample_per_class: int):
    if not isinstance(sample_per_class, int):
        raise ValueError("sample_per_class must be an integer")
    return dataframe.groupby("class").apply(lambda x: x.sample(min(sample_per_class, len(x))))

def sample_data_custom_ratio_per_class(dataframe, sample_per_class_dict: dict[str, int]):
    sampled_dataframe = pd.DataFrame(columns=dataframe.columns)

    if not isinstance(sample_per_class_dict, dict):
        raise ValueError("sample_per_class must be a dictionary with class names as keys and sample sizes as values.")
    for class_name, class_dataframe in dataframe.groupby("class"):
        if class_name not in sample_per_class_dict:
            raise ValueError(f"Class {class_name} not found in sample_per_class_dict")
        sample_size = min(sample_per_class_dict[class_name], len(class_dataframe))
        sampled_dataframe = pd.concat(objs=[class_dataframe.sample(sample_size), sampled_dataframe])
    return sampled_dataframe
