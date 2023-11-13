import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
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

from glaucomaclassifier.constants import CLASS_NAME_ID_MAP
from glaucomaclassifier.dataloader import get_data_loaders
from glaucomaclassifier.models import get_model


def get_predictions(val_loader, model, device):
    # Collect all labels and model outputs
    original_labels = []
    predicted_labels = []
    glaucoma_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(iterable=val_loader, total=len(val_loader)):
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


def plot_confusion_matrix(original_labels, predicted_labels, class_names):
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


def plot_precision_recall_curve(original_labels, glaucoma_probs):
    precision, recall, _ = precision_recall_curve(original_labels, glaucoma_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, lw=2, color="darkorange", label="Precision-Recall curve"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig("precision_recall_curve.png")
    plt.close()


def compute_additional_metrics(original_labels, predicted_labels):
    precision = precision_score(original_labels, predicted_labels)
    recall = recall_score(original_labels, predicted_labels)
    f1 = f1_score(original_labels, predicted_labels)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def get_roc_curve(original_labels, glaucoma_probs):
    fpr, tpr, thresholds = roc_curve(original_labels, glaucoma_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.2f}")

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
    plt.savefig("roc_curve.png")
    plt.close()


def sensitivity_at_certain_specificity(target_specificity, fpr, tpr):
    closest_specificity_index = np.argmin(
        np.abs(tpr + (1 - fpr) - (1 + target_specificity))
    )
    sensitivity_at_specificity = tpr[closest_specificity_index]
    print(
        f"Sensitivity at {target_specificity*100:.0f}% Specificity: {sensitivity_at_specificity:.2f}"
    )


def main():
    # Load the model
    model, _ = get_model(model_name="deit", num_classes=2, pretrained=True)
    model.load_state_dict(torch.load("best_model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _, val_data_loader, _, _, _ = get_data_loaders(
        train_val_ratio=0.8,
        max_rotation_angle=20,
        batch_size=32,
        use_weighted_sampler=False,
    )

    # Get predictions
    original_labels, predicted_labels, glaucoma_probs = get_predictions(
        val_data_loader, model, device
    )

    # Plot and compute metrics
    plot_confusion_matrix(
        original_labels, predicted_labels, list(CLASS_NAME_ID_MAP.keys())
    )
    plot_precision_recall_curve(original_labels, glaucoma_probs)
    compute_additional_metrics(original_labels, predicted_labels)
    get_roc_curve(original_labels, glaucoma_probs)

    # Calculate sensitivity at a certain specificity
    fpr, tpr, thresholds = roc_curve(original_labels, glaucoma_probs)
    sensitivity_at_certain_specificity(
        0.95, fpr, tpr
    )  # Change 0.95 to your target specificity


if __name__ == "__main__":
    main()
