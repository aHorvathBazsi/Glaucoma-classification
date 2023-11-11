import pandas as pd
from imagedatahandler.data_sampling import sample_data_custom_ratio_per_class

def test_sample_data_custom_ratio_per_class():
    # Test case 1: Sample 2 images per class
    df = pd.DataFrame({
        "filename": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        "class": ["cat", "cat", "dog", "dog"]
    })
    sampled_df = sample_data_custom_ratio_per_class(df, {"cat": 2, "dog": 2})
    assert len(sampled_df) == 4
    assert (sampled_df["class"].value_counts() == 2).all()

    # Test case 4: Invalid input
    df = pd.DataFrame({
        "filename": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        "class": ["cat", "cat", "dog", "dog"]
    })
    try:
        sampled_df = sample_data_custom_ratio_per_class(df, 2)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for invalid input"