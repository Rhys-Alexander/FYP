import os
import random

import wandb
import torch

import numpy as np
import pandas as pd
import nibabel as nib
import torchio as tio
import torch.nn as nn
import torch.optim as optim
import torchvision.models.video as models

from tqdm import tqdm
from nilearn import plotting
from torchinfo import torchinfo
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    precision_recall_curve,
)
from IPython.display import display

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

if device == "mps":
    # Empty CUDA cache periodically during training to avoid memory fragmentation
    def empty_cache():
        try:
            # For newer PyTorch versions with MPS cache management
            torch.mps.empty_cache()
        except:
            print("MPS cache management not available")
            pass  # Ignore if this function doesn't exist"


def visualize_scans(dataset_path, split="train", num_samples=3):
    """
    Visualize the first few MRI scans from AD and CN directories.

    Args:
        dataset_path (str): Path to the dataset directory.
        split (str): Dataset split to visualize ('train', 'val', 'test').
        num_samples (int): Number of samples to visualize from each class.
    """
    # Define directories for AD and CN
    ad_dir = os.path.join(dataset_path, split, "AD")
    cn_dir = os.path.join(dataset_path, split, "CN")

    # Get the first few files from each directory
    ad_files = [
        os.path.join(ad_dir, f) for f in os.listdir(ad_dir) if f.endswith(".nii.gz")
    ][:num_samples]
    cn_files = [
        os.path.join(cn_dir, f) for f in os.listdir(cn_dir) if f.endswith(".nii.gz")
    ][:num_samples]

    # Plot the first few AD scans
    print("AD Scans:")
    for file in ad_files:
        plotting.plot_anat(file, title=os.path.basename(file))
    plotting.show()

    # Plot the first few CN scans
    print("CN Scans:")
    for file in cn_files:
        plotting.plot_anat(file, title=os.path.basename(file))
    plotting.show()


class MRIDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        apply_augmentation=False,
        target_size=(128, 128, 128),
    ):
        self.root_dir = root_dir
        self.split = split
        self.samples = []
        self.labels = []
        self.apply_augmentation = apply_augmentation
        self.target_size = target_size

        # Validate inputs
        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Split must be one of 'train', 'val', 'test', got {split}"
            )

        # Get all files from AD and CN directories
        ad_dir = os.path.join(root_dir, split, "AD")
        cn_dir = os.path.join(root_dir, split, "CN")

        if not os.path.exists(ad_dir):
            raise FileNotFoundError(f"AD directory not found at {ad_dir}")
        if not os.path.exists(cn_dir):
            raise FileNotFoundError(f"CN directory not found at {cn_dir}")

        self._load_samples(ad_dir, cn_dir)
        self._setup_transforms()

        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Augmentation applied: {apply_augmentation}")

    def _load_samples(self, ad_dir, cn_dir):
        """Load samples from AD and CN directories"""
        # Load AD samples (label 1)
        ad_files = [f for f in os.listdir(ad_dir) if f.endswith(".nii.gz")]
        for file in ad_files:
            self.samples.append(os.path.join(ad_dir, file))
            self.labels.append(1)  # AD class

        # Load CN samples (label 0)
        cn_files = [f for f in os.listdir(cn_dir) if f.endswith(".nii.gz")]
        for file in cn_files:
            self.samples.append(os.path.join(cn_dir, file))
            self.labels.append(0)  # CN class

        if len(self.samples) == 0:
            raise ValueError(f"No .nii.gz files found in {ad_dir} or {cn_dir}")

    def _setup_transforms(self):
        if self.apply_augmentation:
            self.transforms = tio.Compose(
                [
                    # Randomly flip images along the left-right axis (axis 1 relative to channel-first data)
                    tio.RandomFlip(axes=(1,), p=0.5),
                    # Apply slight affine transformations: modest scaling, rotation (±5°),
                    # and translation limited to the 3-voxel padding.
                    tio.RandomAffine(
                        scales=(0.95, 1.05),
                        degrees=5,
                        translation=3.0,  # in mm; max translation matches the 3 voxel padding
                        p=0.75,
                    ),
                    # Add slight noise reflecting scanner variability.
                    tio.RandomNoise(mean=0.0, std=(0, 0.1), p=0.3),
                    # Adjust intensity minimally using gamma correction.
                    tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.3),
                    # Normalize intensities to zero mean and unit variance.
                    tio.ZNormalization(),
                ]
            )
        else:
            self.transforms = tio.Compose([tio.ZNormalization()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load the .nii.gz file
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
            # Load image using nibabel
            img = nib.load(img_path)
            img_data = img.get_fdata()

            # Validate image dimensions
            expected_d, expected_h, expected_w = self.target_size
            current_d, current_h, current_w = img_data.shape

            if (
                current_d != expected_d
                or current_h != expected_h
                or current_w != expected_w
            ):
                raise ValueError(
                    f"Expected image size {expected_d}x{expected_h}x{expected_w} "
                    f"but got {current_d}x{current_h}x{current_w} for {img_path}"
                )

            # Add channel dimension to numpy array
            img_data = np.expand_dims(img_data, axis=0)

            # Apply transforms
            img_data = self.transforms(img_data)

            # Convert to tensor if not already a tensor
            if not isinstance(img_data, torch.Tensor):
                img_data = torch.tensor(img_data, dtype=torch.float32)

            # Ensure the label is also a tensor
            label = torch.tensor(label, dtype=torch.long)

            return img_data, label

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a default or raise the exception
            raise


def create_datasets_and_loaders(data_root, batch_size, use_augmentation):
    """Create datasets and data loaders."""
    # Create datasets
    train_dataset = MRIDataset(
        data_root, split="train", apply_augmentation=use_augmentation
    )
    val_dataset = MRIDataset(data_root, split="val", apply_augmentation=False)
    test_dataset = MRIDataset(data_root, split="test", apply_augmentation=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    dataset_stats = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_AD_samples": train_dataset.labels.count(1),
        "train_CN_samples": train_dataset.labels.count(0),
        "val_AD_samples": val_dataset.labels.count(1),
        "val_CN_samples": val_dataset.labels.count(0),
    }

    return (
        train_dataset,
        train_loader,
        val_loader,
        test_loader,
        dataset_stats,
    )


# Modified 3D ResNet model with layer freezing
class MRIModel(nn.Module):
    def __init__(self, num_classes=2, freeze_layers=True, architecture="r3d_18"):
        super(MRIModel, self).__init__()
        # Using a video ResNet and modifying it for 3D MRI
        if architecture == "r3d_18":
            self.resnet = models.r3d_18(weights=models.R3D_18_Weights.KINETICS400_V1)
        elif architecture == "mc3_18":
            self.resnet = models.mc3_18(weights=models.MC3_18_Weights.KINETICS400_V1)
        elif architecture == "r2plus1d_18":
            self.resnet = models.r2plus1d_18(
                weights=models.R2Plus1D_18_Weights.KINETICS400_V1
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Replace the first layer to accept single-channel input instead of 3
        if architecture in ["r3d_18", "mc3_18"]:
            self.resnet.stem[0] = nn.Conv3d(
                1,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            )
        elif architecture == "r2plus1d_18":
            # R2Plus1D uses a slightly different stem structure
            self.resnet.stem[0] = nn.Conv3d(
                1,
                45,  # This is the mid-channel count for R2Plus1D
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            )

        # Replace the final fully connected layer for binary classification
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

        # Freeze specific layers if requested
        if freeze_layers:
            self._freeze_layers()

    def _freeze_layers(self):
        """Freeze most layers of the ResNet model, leaving only layer4 and fc unfrozen"""
        # Freeze stem and layers 1-3
        # TODO loook at model in more detail and see where to freeze
        for name, param in self.resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    def count_trainable_params(self):
        """Count and return trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self):
        """Count and return total parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # Input: (B, 1, D, H, W)
        return self.resnet(x)


def setup_model(freeze_layers, device, architecture):
    """Initialize and configure the model."""
    model = MRIModel(
        num_classes=2, freeze_layers=freeze_layers, architecture=architecture
    )
    model = model.to(device)

    # Get parameter statistics
    trainable_params = model.count_trainable_params()
    total_params = model.count_total_params()
    frozen_params = total_params - trainable_params

    model_stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "frozen_percentage": frozen_params / total_params,
    }

    return model, model_stats


def display_model_summary(model, input_size=(1, 1, 128, 128, 128), detailed=True):
    """
    Display a comprehensive summary of the model architecture and parameters.

    Args:
        model: The PyTorch model to analyze
        input_size: The input tensor size (batch_size, channels, depth, height, width)
        detailed: Whether to show detailed layer information
    """
    # Get basic model summary using torchinfo
    summary = torchinfo.summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=0,
    )

    print(f"MODEL ARCHITECTURE SUMMARY:")
    print("=" * 80)
    print(summary)

    # Count parameters by layer type
    layer_counts = {}
    for name, module in model.named_modules():
        layer_type = module.__class__.__name__
        if layer_type not in layer_counts:
            layer_counts[layer_type] = {"count": 0, "params": 0, "trainable_params": 0}

        layer_counts[layer_type]["count"] += 1
        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable_params = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )

        layer_counts[layer_type]["params"] += params
        layer_counts[layer_type]["trainable_params"] += trainable_params

    # Create detailed layer information dataframe
    if detailed:
        layers_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )

                layers_info.append(
                    {
                        "Layer": name,
                        "Type": module.__class__.__name__,
                        "Parameters": params,
                        "Trainable": trainable,
                        "Frozen": params - trainable,
                    }
                )

        # Create and display DataFrame
        df = pd.DataFrame(layers_info)
        if not df.empty:
            print("\nDETAILED LAYER INFORMATION:")
            print("=" * 80)
            display(df)

    # Show frozen vs trainable stats
    total_params = model.count_total_params()
    trainable_params = model.count_trainable_params()
    frozen_params = total_params - trainable_params

    print("\nPARAMETER STATISTICS:")
    print("=" * 80)
    print(f"Total parameters:    {total_params:,}")
    print(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)"
    )
    print(
        f"Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)"
    )

    # Display model architecture as text
    print("\nMODEL ARCHITECTURE DETAILS:")
    print("=" * 80)
    print(model)

    # Return summary for potential further use
    return summary


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train model for one epoch with optimized PyTorch practices.

    Args:
        model: PyTorch model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Computation device (CPU/GPU/MPS)
        epoch: Current epoch number

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """

    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        # Move data to device with non_blocking for potential performance gain
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100 * running_corrects / len(dataloader.dataset)

    # Log epoch-level metrics
    wandb.log(
        {
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "epoch": epoch,
        }
    )

    if device.type == "mps":
        empty_cache()

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate the model and return performance metrics.

    Args:
        model: The PyTorch model to validate
        dataloader: DataLoader containing validation data
        criterion: Loss function
        device: Device to run validation on (cuda/cpu)
        epoch: Current training epoch

    Returns:
        Tuple of (validation loss, validation accuracy, metrics dictionary)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    if dataloader.dataset.split == "val":
        prefix = "val"
        desc = f"Validation Epoch {epoch+1}"
    else:
        prefix = "test"
        epoch = None
        desc = "Testing"

    # Collect predictions
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=desc):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics
    loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    acc = metrics["accuracy"] * 100  # Convert to percentage

    log_to_wandb_dashboard(
        all_labels, all_preds, all_probs, loss, metrics, epoch, prefix=prefix
    )

    return loss, acc


def compute_metrics(labels, preds, probs):
    """Calculate classification metrics using scikit-learn."""

    # Basic metrics
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
    }

    # Class-specific accuracy
    for cls in np.unique(labels):
        mask = labels == cls
        if np.any(mask):
            metrics[f"class_{cls}_acc"] = accuracy_score(labels[mask], preds[mask])
        else:
            metrics[f"class_{cls}_acc"] = 0

    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    metrics.update(
        {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    )

    # Confusion matrix and derived metrics
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, _, _ = cm.ravel()
        metrics["specificity"] = tn / (tn + fp + 1e-10)

    # ROC AUC and PR AUC (with error handling)
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except Exception:
        metrics["roc_auc"] = 0

    try:
        metrics["avg_precision"] = average_precision_score(labels, probs)
    except Exception:
        metrics["avg_precision"] = 0

    return metrics


def log_to_wandb_dashboard(labels, preds, probs, loss, metrics, epoch, prefix="val"):
    """Create visualizations and log metrics to W&B."""

    log_dict = {
        f"{prefix}_loss": loss,
        f"{prefix}_acc": metrics["accuracy"] * 100,
        f"{prefix}_balanced_acc": metrics["balanced_accuracy"] * 100,
        f"{prefix}_CN_acc": metrics["class_0_acc"] * 100,
        f"{prefix}_AD_acc": metrics["class_1_acc"] * 100,
        f"{prefix}_precision": metrics["precision"],
        f"{prefix}_recall": metrics["recall"],
        f"{prefix}_specificity": metrics.get("specificity", 0),
        f"{prefix}_f1": metrics["f1_score"],
        f"{prefix}_roc_auc": metrics["roc_auc"],
        f"{prefix}_avg_precision": metrics["avg_precision"],
    }

    # Add epoch to log_dict only if it's not None
    if epoch is not None:
        log_dict["epoch"] = epoch

    # Create confusion matrix visualization
    conf_matrix = wandb.plot.confusion_matrix(
        preds=preds, y_true=labels, class_names=["CN", "AD"]
    )
    log_dict["confusion_matrix"] = conf_matrix

    # ROC Curve
    try:
        roc_curve_plot = wandb.plot.roc_curve(
            y_true=labels,
            y_probas=np.stack([1 - probs, probs], axis=1),
            labels=["CN", "AD"],
            classes_to_plot=[1],
        )
        log_dict["roc_curve"] = roc_curve_plot
    except Exception as e:
        print(f"Error creating ROC curve: {e}")
        pass

    # PR Curve
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
        pr_data = [[x, y] for x, y in zip(recall_vals, precision_vals)]
        pr_table = wandb.Table(data=pr_data, columns=["recall", "precision"])
        pr_plot = wandb.plot.line(
            pr_table, "recall", "precision", title="Precision-Recall Curve"
        )
        log_dict["pr_curve"] = pr_plot
    except Exception as e:
        print(f"Error creating PR curve: {e}")
        pass

    # Log everything to W&B
    wandb.log(log_dict)


def evaluate_final_model(model, test_loader, criterion, device, num_epochs):
    """Evaluate the best model on the test set."""
    # Load best model
    checkpoint = torch.load("best_model_acc.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded best model from epoch {checkpoint['epoch']+1} with accuracy {checkpoint['val_acc']:.2f}%"
    )

    # Evaluate on test set
    final_test_loss, final_test_acc = validate(
        model, test_loader, criterion, device, num_epochs
    )
    print(f"Final test accuracy: {final_test_acc:.2f}%")

    return final_test_loss, final_test_acc


def setup_wandb(config):
    """Initialize and configure Weights & Biases."""
    wandb.init(
        project="mri-alzheimers-classification",
        config=config,
        id="mczwjb4p",
        resume="must",
    )
    return wandb.config


def setup_training(model, train_dataset, learning_rate, device):
    """Set up training components: loss function, optimizer, scheduler."""
    # Calculate class weights for imbalanced data
    num_ad = train_dataset.labels.count(1)
    num_cn = train_dataset.labels.count(0)
    total = num_ad + num_cn

    # Inverse frequency weighting
    weight_cn = total / (2 * num_cn) if num_cn > 0 else 1.0
    weight_ad = total / (2 * num_ad) if num_ad > 0 else 1.0

    class_weights = torch.tensor([weight_cn, weight_ad], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Set up parameter groups with different learning rates
    fc_params = list(model.resnet.fc.parameters())
    other_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not any(p is fc_param for fc_param in fc_params)
    ]

    param_groups = [
        {"params": other_params, "lr": learning_rate},
        {"params": fc_params, "lr": learning_rate * 10},  # Higher LR for final layer
    ]

    optimizer = optim.AdamW(param_groups, lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=learning_rate / 100
    )

    return criterion, optimizer, scheduler


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint if available."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_val_acc = 0.0
        best_val_loss = float("inf")

    return start_epoch, best_val_acc, best_val_loss


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    val_acc,
    val_loss,
    best_val_acc,
    best_val_loss,
    filepath,
    is_best=False,
    metric_type=None,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, filepath)

    if is_best and metric_type:
        try:
            artifact = wandb.Artifact(f"best_model_{metric_type}", type="model")
            artifact.add_file(filepath)
            wandb.log_artifact(artifact)
            print(f"Model saved (best {metric_type})!")
        except OSError as e:
            print(f"Failed to log artifact to W&B: {e}")
            print("Continuing training without W&B artifact logging...")


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    start_epoch,
    patience,
    checkpoint_path,
):
    """Execute the training loop with validation and early stopping."""
    best_val_acc = 0.0
    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model by accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_acc,
                val_loss,
                best_val_acc,
                best_val_loss,
                "best_model_acc.pth",
                is_best=True,
                metric_type="acc",
            )
            early_stopping_counter = 0

        # Save best model by loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_acc,
                val_loss,
                best_val_acc,
                best_val_loss,
                "best_model_loss.pth",
                is_best=True,
                metric_type="loss",
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Check for early stopping
        if early_stopping_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs without improvement.")
            break

        # Save regular checkpoint
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            val_acc,
            val_loss,
            best_val_acc,
            best_val_loss,
            checkpoint_path,
        )

    return epoch + 1, best_val_acc, best_val_loss


def validate_checkpoint(checkpoint_path, data_path, architecture, batch_size=2):
    """
    Run validation on a specific checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint
        data_path (str): Path to the dataset
        batch_size (int): Batch size for validation

    Returns:
        dict: Validation metrics
    """
    print(f"Validating checkpoint: {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None

    # Create validation dataset and loader
    val_dataset = MRIDataset(data_path, split="val", apply_augmentation=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create model with same architecture as used during training
    model = MRIModel(num_classes=2, freeze_layers=True, architecture=architecture)
    model = model.to(device)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        previous_val_acc = checkpoint.get("val_acc", "N/A")
        print(
            f"Loaded checkpoint from epoch {epoch+1} with previous validation accuracy: {previous_val_acc:.2f}%"
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

    # Set up criterion (without class weights since we're just evaluating)
    criterion = nn.CrossEntropyLoss()

    # Run validation
    model.eval()
    val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

    print(f"\nValidation Results:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # If MPS device, clear cache
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except:
            pass

    return {"epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc}


def main(data_path):
    # Configuration
    config = {
        "architecture": "r3d_18",
        "dataset": "MRI-AD-CN",
        "epochs": 20,
        "batch_size": 2,
        "learning_rate": 0.0001,
        "optimizer": "AdamW",
        "device": str(device),
        "input_dimensions": "128x128x128",
        "freeze_layers": True,
        "data_augmentation": True,
        "patience": 5,
    }

    # Initialize wandb
    config = setup_wandb(config)

    # Create datasets and loaders
    (
        train_dataset,
        train_loader,
        val_loader,
        test_loader,
        dataset_stats,
    ) = create_datasets_and_loaders(
        data_path, config.batch_size, config.data_augmentation
    )

    # Update wandb config with dataset stats
    wandb.config.update(dataset_stats)

    # Setup model
    model, model_stats = setup_model(config.freeze_layers, device, config.architecture)

    # Display model stats
    print(f"Total parameters: {model_stats['total_params']:,}")
    print(
        f"Trainable parameters: {model_stats['trainable_params']:,} ({model_stats['trainable_params']/model_stats['total_params']:.2%})"
    )
    print(
        f"Frozen parameters: {model_stats['frozen_params']:,} ({model_stats['frozen_params']/model_stats['total_params']:.2%})"
    )

    # Update wandb config with model stats
    wandb.config.update(model_stats)

    # Watch model in wandb
    wandb.watch(model, log="all", log_freq=10)

    # Setup training components
    criterion, optimizer, scheduler = setup_training(
        model, train_dataset, config.learning_rate, device
    )

    # Load checkpoint if exists
    checkpoint_path = "checkpoints/checkpoint.pth"
    start_epoch, best_val_acc, best_val_loss = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path
    )

    # Run to check validation pipeline changes
    # validate_checkpoint(checkpoint_path, data_path, config.architecture, batch_size=config.batch_size)

    # Train model
    # epochs_trained, best_val_acc, best_val_loss = train_model(
    #     model,
    #     train_loader,
    #     val_loader,
    #     criterion,
    #     optimizer,
    #     scheduler,
    #     device,
    #     config.epochs,
    #     start_epoch,
    #     config.patience,
    #     checkpoint_path,
    # )

    # Final evaluation on test set
    evaluate_final_model(model, test_loader, criterion, device, config.epochs)

    # Log final metrics
    wandb.run.summary["best_val_acc"] = best_val_acc
    wandb.run.summary["best_val_loss"] = best_val_loss
    # wandb.run.summary["total_epochs"] = epochs_trained

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Ensure checkpoint directory exists
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    data_path = "./data/adni-split"
    main(data_path)
