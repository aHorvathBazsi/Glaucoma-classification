from torch.utils.data import DataLoader
from glaucomaclassifier.dataset import CustomImageDataset

def get_data_loader(dataset: CustomImageDataset, batch_size: int, is_train: bool = False) -> DataLoader:
    shuffle = True if is_train else False
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
