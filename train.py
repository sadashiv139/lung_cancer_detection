import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torchvision import transforms as T

from dataset import LungDataset
from model import HybridCNNTransformer
from utils import compute_metrics, plot_confusion_matrix, print_classification_report


DATASET_PATH = "dataset"
CLASSES: List[str] = ["Bengin", "Malignant", "Normal"]


def load_image_paths_and_labels(
    dataset_path: str, classes: List[str]
) -> Tuple[list, list]:
    image_paths = []
    labels = []

    for i, c in enumerate(classes):
        class_path = os.path.join(dataset_path, c)
        if not os.path.isdir(class_path):
            raise RuntimeError(f"Expected folder not found: {class_path}")

        for img in os.listdir(class_path):
            img_path = os.path.join(class_path, img)
            if not os.path.isfile(img_path):
                continue
            image_paths.append(img_path)
            labels.append(i)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under {dataset_path}.")

    return image_paths, labels


def add_gaussian_noise(x: torch.Tensor, std: float = 0.03) -> torch.Tensor:
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0.0, 1.0)


def get_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Data augmentation details:
    # - rotation: RandomRotation(15 degrees)
    # - zoom: RandomResizedCrop with scale 0.8 to 1.0
    # - brightness: ColorJitter brightness +/-20%
    # - noise: gaussian noise with std=0.03
    train_transform = T.Compose(
        [
            T.ToPILImage(),
            T.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),  # gentle zoom
            T.RandomRotation(degrees=8),  # gentle rotation
            T.ColorJitter(brightness=0.15),  # mild brightness jitter
            T.ToTensor(),
            T.Lambda(lambda x: add_gaussian_noise(x, std=0.01)),  # mild noise
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    val_transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_transform, val_transform


def train_and_evaluate(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    k_folds: int = 5,
    patience: int = 5,
) -> None:
    # Device selection (Apple Silicon MPS support)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # Load dataset
    image_paths, labels = load_image_paths_and_labels(DATASET_PATH, CLASSES)
    labels_np = np.array(labels)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    train_transform, val_transform = get_transforms()

    fold_metrics = []
    best_fold_acc = -1.0
    best_fold_cm = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels_np), start=1):
        print(f"\n========== Fold {fold}/{k_folds} ==========")

        train_x = [image_paths[i] for i in train_idx]
        train_y = [labels[i] for i in train_idx]
        val_x = [image_paths[i] for i in val_idx]
        val_y = [labels[i] for i in val_idx]

        train_dataset = LungDataset(train_x, train_y, transform=train_transform)
        val_dataset = LungDataset(val_x, val_y, transform=val_transform)

        # Balance classes in each fold with weighted sampling
        class_counts = np.bincount(train_y, minlength=len(CLASSES)).astype(np.float32)
        class_weights = 1.0 / np.maximum(class_counts, 1.0)
        sample_weights = [class_weights[y] for y in train_y]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        model = HybridCNNTransformer(
            num_classes=len(CLASSES), pretrained=True, strict_pretrained=False
        ).to(device)

        # Weight loss by inverse class frequency
        ce_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=ce_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

        best_val_acc_for_fold = -1.0
        best_state_for_fold = None
        no_improve_epochs = 0

        # Training loop per fold
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            loop = tqdm(
                train_loader,
                desc=f"Fold {fold} Epoch {epoch + 1}/{epochs}",
                unit="batch",
            )
            for imgs, labels_batch in loop:
                imgs = imgs.to(device)
                labels_batch = labels_batch.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)
            print(f"Fold {fold} Epoch {epoch + 1} Loss: {avg_loss:.4f}")

            # Quick validation each epoch for scheduler + early stopping
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, labels_batch in val_loader:
                    imgs = imgs.to(device)
                    labels_batch = labels_batch.to(device)
                    outputs = model(imgs)
                    preds = torch.argmax(outputs, dim=1)
                    val_total += labels_batch.size(0)
                    val_correct += (preds == labels_batch).sum().item()

            val_acc = val_correct / max(val_total, 1)
            print(f"Fold {fold} Epoch {epoch + 1} Val Acc: {val_acc * 100:.2f}%")
            scheduler.step(val_acc)

            if val_acc > best_val_acc_for_fold:
                best_val_acc_for_fold = val_acc
                best_state_for_fold = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        # Restore best epoch for this fold before final fold validation
        if best_state_for_fold is not None:
            model.load_state_dict(best_state_for_fold)

        # Validation for current fold
        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for imgs, labels_batch in tqdm(
                val_loader, desc=f"Fold {fold} Evaluating", unit="batch"
            ):
                imgs = imgs.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels_batch.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        metrics, cm = compute_metrics(y_true, y_pred)
        fold_metrics.append(metrics)

        print(
            f"Fold {fold} Metrics -> "
            f"Acc: {metrics['accuracy'] * 100:.2f}% | "
            f"Prec: {metrics['precision'] * 100:.2f}% | "
            f"Rec: {metrics['recall'] * 100:.2f}% | "
            f"F1: {metrics['f1_score'] * 100:.2f}%"
        )

        if metrics["accuracy"] > best_fold_acc:
            best_fold_acc = metrics["accuracy"]
            best_fold_cm = cm
            torch.save(model.state_dict(), "lung_cancer_model.pth")
            print("Saved best fold model to lung_cancer_model.pth")

        if fold == k_folds:
            print_classification_report(y_true, y_pred, CLASSES)

    # Aggregate CV metrics
    avg_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
    avg_prec = float(np.mean([m["precision"] for m in fold_metrics]))
    avg_rec = float(np.mean([m["recall"] for m in fold_metrics]))
    avg_f1 = float(np.mean([m["f1_score"] for m in fold_metrics]))

    print("\n========== Cross-Validation Summary ==========")
    print(f"Average Accuracy:  {avg_acc * 100:.2f}%")
    print(f"Average Precision: {avg_prec * 100:.2f}%")
    print(f"Average Recall:    {avg_rec * 100:.2f}%")
    print(f"Average F1-score:  {avg_f1 * 100:.2f}%")

    # Plot confusion matrix from best fold
    if best_fold_cm is not None:
        try:
            plot_confusion_matrix(best_fold_cm, CLASSES, title="Best Fold Confusion Matrix")
        except Exception as e:
            print(f"Could not display confusion matrix plot: {e}")


if __name__ == "__main__":
    train_and_evaluate()