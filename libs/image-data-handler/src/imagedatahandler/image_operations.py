from pathlib import Path

import cv2
import numpy as np


def save_image(image: np.ndarray, save_path: str) -> None:
    save_path = Path(save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    cv2.imwrite(str(save_path), image)


def remove_padding(image, lower_binary_threshold: int = 10, upper_binary_threshold: int = 255) -> np.ndarray:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image
    _, binary_image = cv2.threshold(image_gray, lower_binary_threshold, upper_binary_threshold, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    # Get the bounding rectangle of the largest contour
    upper_left_x, upper_left_y, width, height = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding rectangle
    cropped_image = image[upper_left_y:upper_left_y + height, upper_left_x:upper_left_x + width]
    return cropped_image

