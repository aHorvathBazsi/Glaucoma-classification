from typing import Tuple

from torchvision import transforms

from glaucomaclassifier.constants import CLASS_NAME_ID_MAP


def get_label_transform():
    return transforms.Lambda(lambda label: CLASS_NAME_ID_MAP[label])


def get_image_transform(
    input_size: Tuple[int, int], max_rotation_angle: int, is_train: bool = True
):
    image_transform = [
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if is_train:
        image_transform.insert(1, transforms.RandomHorizontalFlip())
        image_transform.insert(2, transforms.RandomRotation(max_rotation_angle))

    return transforms.Compose(image_transform)


if __name__ == "__main__":
    image_transform = get_image_transform(is_train=True)
    print("HEY")
