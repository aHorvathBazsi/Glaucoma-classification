import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import os
import logging

import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from glaucomaclassifier.constants import CLASS_NAME_ID_MAP
from glaucomaclassifier.dataloader import get_test_data_loader
from glaucomaclassifier.models import get_model
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_predictions(data_loader, model, device):
    # Collect all labels and model outputs
    original_labels = []
    predicted_labels = []
    glaucoma_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(iterable=data_loader, total=len(data_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)

            original_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
            glaucoma_probs.extend(
                probs.cpu().numpy()[:, 1]
            )  # Probability for class '1'

    return original_labels, predicted_labels, glaucoma_probs


def plot_confusion_matrix(original_labels, predicted_labels, class_names, save_path):
    cm = confusion_matrix(original_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    plt.close()


def plot_precision_recall_curve(original_labels, glaucoma_probs, save_path):
    precision, recall, _ = precision_recall_curve(original_labels, glaucoma_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, lw=2, color="darkorange", label="Precision-Recall curve"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    plt.close()


def compute_additional_metrics(original_labels, predicted_labels):
    precision = precision_score(original_labels, predicted_labels)
    recall = recall_score(original_labels, predicted_labels)
    f1 = f1_score(original_labels, predicted_labels)

    return precision, recall, f1


def get_roc_curve(original_labels, glaucoma_probs, roc_curve_save_path):
    fpr, tpr, thresholds = roc_curve(original_labels, glaucoma_probs)
    roc_auc = auc(fpr, tpr)
    logging.info(f"ROC AUC: {roc_auc:.2f}")

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_save_path)
    plt.close()
    return fpr, tpr, thresholds

def sensitivity_at_certain_specificity(target_specificity, false_positive_rates, true_positive_rates):
    # Calculate specificities from FPRs
    specificities = 1 - false_positive_rates

    # Find the index where specificity is closest to the target specificity
    closest_index = np.abs(specificities - target_specificity).argmin()

    # Get the sensitivity (TPR) at this index
    sensitivity_at_target_spec = true_positive_rates[closest_index]

    # Logging the result for clarity
    print(f"Sensitivity at {target_specificity*100:.0f}% Specificity: {sensitivity_at_target_spec:.2f}")

    return sensitivity_at_target_spec


def evaluate_model(
    model_state_dict_path, model, data_loader, device, wandb_track_enabled, run_name
):
    # Load the model
    model.load_state_dict(torch.load(model_state_dict_path))
    model.to(device)
    model.eval()

    # Get predictions
    original_labels, predicted_labels, glaucoma_probs = get_predictions(
        data_loader, model, device
    )

    # Plot and compute metrics
    confusion_matrix_save_path = os.path.join(THIS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(
        original_labels,
        predicted_labels,
        list(CLASS_NAME_ID_MAP.keys()),
        save_path=confusion_matrix_save_path,
    )
    pr_curve_save_path = os.path.join(THIS_DIR, "precision_recall_curve.png")
    plot_precision_recall_curve(original_labels, glaucoma_probs, pr_curve_save_path)

    precision, recall, f1 = compute_additional_metrics(
        original_labels, predicted_labels
    )

    roc_curve_save_path = os.path.join(THIS_DIR, "roc_curve.png")
    fpr, tpr, thresholds = get_roc_curve(
        original_labels, glaucoma_probs, roc_curve_save_path
    )

    sensitivity_at_high_specificity = sensitivity_at_certain_specificity(
        0.95, fpr, tpr
    )  # Change 0.95 to your target specificity
    if wandb_track_enabled:
        wandb.init(project="glaucoma-classification", name=run_name)
        wandb.log(
            {
                "confusion_matrix": wandb.Image(confusion_matrix_save_path),
                "precision_recall_curve": wandb.Image(pr_curve_save_path),
                "roc_curve": wandb.Image(roc_curve_save_path),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "sensitivity_at_95_specificity": sensitivity_at_high_specificity,
            }
        )


def main():
    model, _ = get_model(model_name="deit", num_classes=2, pretrained=True)
    test_data_loader = get_test_data_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_model(
        model_state_dict_path="best-model.pth",
        model=model,
        data_loader=test_data_loader,
        device=device,
        wandb_track_enabled=False,
        run_name="evaluate-long-run-experiment",
    )


if __name__ == "__main__":
    main()
