import numpy as np
from imagedatahandler.image_operations import remove_padding


def test_remove_padding():
    # Test case 1: Image with padding on all sides
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:90, 10:90, :] = 255
    cropped_image = remove_padding(image)
    assert cropped_image.shape == (80, 80, 3)

    # Test case 2: Image with padding on top and left sides
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:90, 10:100, :] = 255
    cropped_image = remove_padding(image)
    assert cropped_image.shape == (80, 90, 3)
