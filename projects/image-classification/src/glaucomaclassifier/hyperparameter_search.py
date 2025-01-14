import os

import torch
import wandb
import yaml
from glaucomaclassifier.dataloader import get_test_data_loader
from glaucomaclassifier.evaluate import evaluate_model
from glaucomaclassifier.train_runner import run_training
from glaucomaclassifier.training_run_config import TrainingRunConfig

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def wandb_run_experiment():
    # Initialize a wandb run
    wandb.init(project="glaucoma-classification")

    # Get configuration from wandb
    config = wandb.config

    # Create a TrainingRunConfig object using the wandb config
    training_run_config = TrainingRunConfig(
        run_name=wandb.run.name,
        model_name=config.model_name,
        batch_size=config.batch_size,
        unfreeze_blocks_number=config.unfreeze_blocks_number,
        optimizer_type=config.optimizer_type,
        base_learning_rate=config.base_learning_rate,
        gamma=config.gamma,
    )

    # Run the training
    model = run_training(training_run_config)

    test_data_loader = get_test_data_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_state_dict_path = os.path.join(
        THIS_DIR, f"{training_run_config.run_name}.pth"
    )
    evaluate_model(
        model_state_dict_path=model_state_dict_path,
        model=model,
        data_loader=test_data_loader,
        device=device,
        run_name=wandb.run.name,
        wandb_track_enabled=True,
    )

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    with open("sweep_config.yaml", "r") as file:
        sweep_configuration = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="glaucoma-classification")
    wandb.agent(sweep_id=sweep_id, function=wandb_run_experiment, count=8)
