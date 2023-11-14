# Glaucoma Classification

This project focuses on classifying retinal images into healthy or glaucoma-affected categories using deep learning models.

## Setting Up Virtual Environment

1. Install `virtualenvwrapper` as per the [documentation](https://virtualenvwrapper.readthedocs.io/en/latest/install.html).
2. Create a virtual environment using: `mkvirtualenv env-name --python python3.11`.
3. Install `poetry` for package management with: `pip install poetry`. We use poetry for editable installs, so any updates to `glaucomaclassifier` or `imagedatahandler` do not require reinstallation.
4. Run `poetry install` to install all project dependencies.



## Project Structure

- `constants.py`: Defines constants used across the project, including data paths and configurations.
- `evaluate.py`: Script for evaluating the trained models on a test dataset.
- `hyperparameter_tuning.py`: Script to perform hyperparameter search using Weights and Biases.
- `run_experiment.py`: Script to run training experiments.

## Data Sampling

There are two approaches for data sampling:

- **Random Sampling**: Select a random subset of images from your dataset.
- **Similarity-Based Sampling**: Utilize image embeddings to sample images based on similarity, ensuring a diverse set of images for training.

## Data Structure

Organize your image data as specified in `constants.py`. Ensure that the image paths and labels are correctly set up for training and evaluation.


## Evaluating a model

To evaluate a trained model:
1. Ensure the model weights file is in the project directory.
2. Run `python evaluate.py`. This script will load the model, run it on the test dataset, and output evaluation metrics.

## Hyperparameter Tuning

Use `hyperparameter_tuning.py` to perform sweeps for hyperparameter optimization:
1. Configure your sweep parameters in a YAML file (sweep_config.yaml)
2. Run `python hyperparameter_tuning.py` to start the tuning process.

## Running Individual Experiments

To train and evaluate models:
1. Set up your experiment parameters in `run_experiment.py`.
2. Execute `python run_experiment.py` to start the training and evaluation process.

## Contact

For any queries or contributions, feel free to reach out to me.

- Balázs Horváth ([@aHorvathBazsi](https://github.com/aHorvathBazsi))