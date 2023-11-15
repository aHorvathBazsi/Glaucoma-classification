import argparse

import wandb
from glaucomaclassifier.train_runner import run_training
from glaucomaclassifier.training_run_config import TrainingRunConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, help="Name of the run", required=True)
    parser.add_argument(
        "--run-path",
        help="Restart a run using a wandb run config.",
    )
    return parser.parse_args()


def run_experiment(config: dict):
    training_run_config = TrainingRunConfig(
        run_name=config["run_name"],
        model_name=config["model_name"],
        batch_size=config["batch_size"],
        unfreeze_blocks_number=config["unfreeze_blocks_number"],
        optimizer_type=config["optimizer_type"],
        base_learning_rate=config["base_learning_rate"],
        gamma=config["gamma"],
        num_epochs=20,
    )
    _ = run_training(training_run_config)


def main():
    args = parse_arguments()
    if args.run_path is not None:
        api = wandb.Api()
        run = api.run("horvathbazsi/glaucoma-classification/g6hoaed2")
        config = run.config
    else:
        config = {
            "gamma": 0.807476065399156,
            "batch_size": 32,
            "model_name": "deit",
            "optimizer_type": "AdamW",
            "base_learning_rate": 0.0010997898678008486,
            "unfreeze_blocks_number": 2,
        }
    config["run_name"] = args.run_name
    run_experiment(config)

if __name__ == "__main__":
    main()
