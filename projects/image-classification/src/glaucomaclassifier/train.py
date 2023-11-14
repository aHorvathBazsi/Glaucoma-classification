import os
import time

import numpy as np
import torch
import wandb
from glaucomaclassifier.training_run_config import TrainingRunConfig
from tqdm import tqdm

import logging

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def train_model(
    training_run_config: TrainingRunConfig,
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    device,
    wandb_track_enabled=False,
    num_epochs=10,
):
    if wandb_track_enabled:
        wandb.init(
            project="glaucoma-classification",
            name=training_run_config.run_name,
            config=training_run_config.__dict__,
        )
    since = time.time()
    best_loss = np.inf

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch}/{num_epochs - 1}")
        logging.info("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            # Create a tqdm iterator
            progress_bar = tqdm(
                dataloaders[phase], desc=f"Epoch {epoch}/{num_epochs - 1} {phase}"
            )
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update tqdm description with the current batch loss
                progress_bar.set_description(
                    f"Epoch {epoch}/{num_epochs - 1} {phase} Loss: {loss.item():.4f}"
                )

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logging.info("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            if wandb_track_enabled:
                wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc})

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                model_state_dict_path = os.path.join(
                    THIS_DIR, f"{training_run_config.run_name}.pth"
                )
                torch.save(model.state_dict(), model_state_dict_path)

    if wandb_track_enabled:
        artifact = wandb.Artifact(
            f"glaucoma-classifier-{training_run_config.run_name}", type="model"
        )
        artifact.add_file(model_state_dict_path)
        wandb.log_artifact(artifact)

    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logging.info("Best Val Loss: {:.4f}".format(best_loss))
    return model
