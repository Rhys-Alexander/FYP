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


class DeviceManager:
    """Centralized device management"""

    @staticmethod
    def get_device():
        """Get the best available device."""
        return (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )

    @staticmethod
    def empty_cache():
        """Empty the device cache based on device type."""
        device = DeviceManager.get_device()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            try:
                torch.mps.empty_cache()
            except:
                print("MPS cache management not available")
                pass

    @staticmethod
    def to_device(data, device=None, non_blocking=True):
        """Move data to device with standardized approach"""
        if device is None:
            device = DeviceManager.get_device()

        if isinstance(data, (list, tuple)):
            return [DeviceManager.to_device(x, device, non_blocking) for x in data]

        return data.to(device, non_blocking=non_blocking)


class Config:
    """Centralized configuration management"""

    def __init__(self, **kwargs):
        # Default configuration
        self.architecture = "r3d_18"
        self.dataset = "MRI-AD-CN"
        self.epochs = 2
        self.batch_size = 2
        self.learning_rate = 0.0001
        self.optimizer = "AdamW"
        self.device = DeviceManager.get_device()
        self.input_dimensions = "128x128x128"
        self.freeze_layers = True
        self.data_augmentation = True
        self.patience = 5
        self.target_size = (128, 128, 128)
        self.checkpoint_dir = "checkpoints"

        # Override defaults with provided arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def to_dict(self):
        """Convert config to dictionary for wandb"""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


class MetricsManager:
    """Handles metric computation and logging across training/validation/test"""

    @staticmethod
    def compute_metrics(labels, preds, probs):
        """Calculate and return all classification metrics"""
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

    @staticmethod
    def log_metrics(labels, preds, probs, loss, epoch=None, prefix="val"):
        """Compute metrics and log to wandb with visualizations"""
        metrics = MetricsManager.compute_metrics(labels, preds, probs)

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

        # Log everything to W&B
        wandb.log(log_dict)

        return metrics


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

        # Generic function to load files from a directory with a specific label
        def load_from_dir(directory, label):
            files = [f for f in os.listdir(directory) if f.endswith(".nii.gz")]
            for file in files:
                self.samples.append(os.path.join(directory, file))
                self.labels.append(label)
            return len(files)

        # Load samples from each class directory
        ad_count = load_from_dir(ad_dir, 1)  # AD class
        cn_count = load_from_dir(cn_dir, 0)  # CN class

        if len(self.samples) == 0:
            raise ValueError(f"No .nii.gz files found in {ad_dir} or {cn_dir}")

        print(f"Loaded {ad_count} AD samples and {cn_count} CN samples")

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


def create_datasets_and_loaders(
    data_root, batch_size, use_augmentation, target_size=(128, 128, 128)
):
    """Create datasets and data loaders."""
    # Create datasets
    train_dataset = MRIDataset(
        data_root,
        split="train",
        apply_augmentation=use_augmentation,
        target_size=target_size,
    )
    val_dataset = MRIDataset(
        data_root, split="val", apply_augmentation=False, target_size=target_size
    )
    test_dataset = MRIDataset(
        data_root, split="test", apply_augmentation=False, target_size=target_size
    )

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


class ModelManager:
    """Handles model initialization, architecture selection, and summary"""

    @staticmethod
    def create_model(config):
        """Create and configure the model based on config"""
        model = MRIModel(
            num_classes=2,
            freeze_layers=config.freeze_layers,
            architecture=config.architecture,
        )
        model = DeviceManager.to_device(model, config.device)

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

    @staticmethod
    def setup_training_components(model, train_dataset, config):
        """Set up criterion, optimizer, and scheduler"""
        # Calculate class weights for imbalanced data
        num_ad = train_dataset.labels.count(1)
        num_cn = train_dataset.labels.count(0)
        total = num_ad + num_cn

        # Inverse frequency weighting
        weight_cn = total / (2 * num_cn) if num_cn > 0 else 1.0
        weight_ad = total / (2 * num_ad) if num_ad > 0 else 1.0

        class_weights = torch.tensor([weight_cn, weight_ad], device=config.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Set up parameter groups with different learning rates
        fc_params = list(model.resnet.fc.parameters())
        other_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and not any(p is fc_param for fc_param in fc_params)
        ]

        param_groups = [
            {"params": other_params, "lr": config.learning_rate},
            {
                "params": fc_params,
                "lr": config.learning_rate * 10,
            },  # Higher LR for final layer
        ]

        optimizer = optim.AdamW(
            param_groups, lr=config.learning_rate, weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=config.learning_rate / 100
        )

        return criterion, optimizer, scheduler


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


class CheckpointManager:
    """Handles saving and loading model checkpoints."""

    def __init__(self, checkpoint_dir="checkpoints"):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir (str): Directory to save checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
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
        """
        Save model checkpoint to file.

        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch number
            val_acc: Validation accuracy
            val_loss: Validation loss
            best_val_acc: Best validation accuracy so far
            best_val_loss: Best validation loss so far
            filepath: Path to save the checkpoint
            is_best: Whether this is a best model checkpoint
            metric_type: Type of metric for best model (e.g., "acc" or "loss")
        """
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

        save_path = os.path.join(self.checkpoint_dir, filepath)
        torch.save(checkpoint, save_path)

        if is_best and metric_type:
            try:
                artifact = wandb.Artifact(f"best_model_{metric_type}", type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)
                print(f"Model saved (best {metric_type})!")
            except OSError as e:
                print(f"Failed to log artifact to W&B: {e}")
                print("Continuing training without W&B artifact logging...")

    def save_best_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        val_acc,
        val_loss,
        best_val_acc,
        best_val_loss,
    ):
        """Centralized best checkpoint saving with appropriate naming and logging"""
        updated = False

        # Check if best accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            self.save(
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
            updated = True

        # Check if best loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            self.save(
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
            updated = True

        # Save regular checkpoint
        self.save(
            model,
            optimizer,
            scheduler,
            epoch,
            val_acc,
            val_loss,
            best_val_acc,
            best_val_loss,
            "checkpoint.pth",
        )

        return updated, best_val_acc, best_val_loss

    def load(self, model, optimizer=None, scheduler=None, filepath="checkpoint.pth"):
        """
        Load model checkpoint if available.

        Args:
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            filepath: Checkpoint file to load

        Returns:
            tuple: (start_epoch, best_val_acc, best_val_loss)
        """
        checkpoint = self.get(filepath)

        if checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if scheduler is not None:
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

    def get(self, filepath):
        """
        Get checkpoint data without loading it into a model.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            dict: The checkpoint dictionary or None if file doesn't exist
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filepath)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            return checkpoint
        else:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}")
            return None


class TrainerEngine:
    """Handles training, validation, and testing workflows."""

    def __init__(
        self, model, criterion, optimizer, scheduler, device, checkpoint_manager=None
    ):
        """
        Initialize the trainer with model and training components.

        Args:
            model: The PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Computation device (CPU/GPU/MPS)
            checkpoint_manager: Optional CheckpointManager instance
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")

    def train_one_epoch(self, dataloader, epoch):
        """
        Train model for one epoch with optimized PyTorch practices.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            # Move data to device with non_blocking for potential performance gain
            inputs = DeviceManager.to_device(inputs, self.device)
            labels = DeviceManager.to_device(labels, self.device)

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

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

        if self.device == "mps":
            DeviceManager.empty_cache()

        return epoch_loss, epoch_acc

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs,
        start_epoch=0,
        patience=5,
    ):
        """
        Execute the training loop with validation and early stopping.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Total number of epochs to train
            start_epoch: Starting epoch (for resuming training)
            patience: Number of epochs to wait for improvement before early stopping

        Returns:
            tuple: (epochs_trained, best_validation_accuracy, best_validation_loss)
        """
        early_stopping_counter = 0

        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self.evaluate(val_loader, epoch)

            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr:.6f}")

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save checkpoints and update best metrics
            updated, self.best_val_acc, self.best_val_loss = (
                self.checkpoint_manager.save_best_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_acc,
                    val_loss,
                    self.best_val_acc,
                    self.best_val_loss,
                )
            )

            # Update early stopping counter
            if not updated:
                early_stopping_counter += 1
                print(
                    f"No improvement detected. Early stopping counter: {early_stopping_counter}/{patience}"
                )
            else:
                early_stopping_counter = 0
                print("Model improved! Early stopping counter reset.")

            # Check for early stopping
            if early_stopping_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs without improvement.")
                break

        return epoch + 1, self.best_val_acc, self.best_val_loss

    def evaluate(self, dataloader, epoch=None, prefix=None, checkpoint_path=None):
        """
        Generic model evaluation function that works for both validation and test sets.

        Args:
            dataloader: DataLoader containing evaluation data
            epoch: Current training epoch (None for standalone evaluation)
            prefix: Metric prefix for logging ('val' or 'test')
            checkpoint_path: Optional path to load model checkpoint before evaluation

        Returns:
            Tuple of (evaluation loss, evaluation accuracy)
        """
        # Determine prefix from dataset if not explicitly provided
        if prefix is None:
            prefix = dataloader.dataset.split

        # Load checkpoint if specified
        if checkpoint_path:
            checkpoint = self.checkpoint_manager.get(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            val_acc = checkpoint.get("val_acc", 0.0)
            epoch = checkpoint.get("epoch", 0)
            print(
                f"Loaded checkpoint from epoch {epoch+1} with accuracy {val_acc:.2f}%"
            )

        # Set model to evaluation mode
        self.model.eval()

        # Collect predictions and calculate loss
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        # Determine description for progress bar
        desc = (
            f"{prefix.capitalize()} Epoch {epoch+1}"
            if epoch is not None
            else f"{prefix.capitalize()}"
        )

        # Collect predictions
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=desc):
                inputs, labels = DeviceManager.to_device(
                    inputs, self.device
                ), DeviceManager.to_device(labels, self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
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

        # Calculate metrics and log to W&B
        loss = running_loss / len(dataloader.dataset)
        metrics = MetricsManager.compute_metrics(all_labels, all_preds, all_probs)
        MetricsManager.log_metrics(
            all_labels, all_preds, all_probs, loss, epoch, prefix
        )

        acc = metrics["accuracy"] * 100  # Convert to percentage

        return loss, acc


def main(data_path):
    # Initialize configuration
    config = Config(data_path=data_path)

    # Initialize wandb with config
    wandb.init(
        project="mri-alzheimers-classification",
        config=config.to_dict(),
    )
    config.update(**wandb.config)

    # Create datasets and loaders
    train_dataset, train_loader, val_loader, test_loader, dataset_stats = (
        create_datasets_and_loaders(
            config.data_path,
            config.batch_size,
            config.data_augmentation,
            target_size=config.target_size,
        )
    )

    # Update config with dataset stats
    config.update(**dataset_stats)

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)

    # Setup model
    model, model_stats = ModelManager.create_model(config)

    # Display model stats
    print(f"Total parameters: {model_stats['total_params']:,}")
    print(
        f"Trainable parameters: {model_stats['trainable_params']:,} "
        f"({model_stats['trainable_params']/model_stats['total_params']:.2%})"
    )
    print(
        f"Frozen parameters: {model_stats['frozen_params']:,} "
        f"({model_stats['frozen_params']/model_stats['total_params']:.2%})"
    )

    # Update config with model stats
    config.update(**model_stats)

    # Watch model in wandb
    wandb.watch(model, log="all", log_freq=10)

    # Setup training components
    criterion, optimizer, scheduler = ModelManager.setup_training_components(
        model, train_dataset, config
    )

    # Load checkpoint if exists
    start_epoch, best_val_acc, best_val_loss = checkpoint_manager.load(
        model, optimizer, scheduler
    )

    # Create trainer engine
    trainer = TrainerEngine(
        model, criterion, optimizer, scheduler, config.device, checkpoint_manager
    )
    trainer.best_val_acc = best_val_acc
    trainer.best_val_loss = best_val_loss

    # Train model
    epochs_trained, best_val_acc, best_val_loss = trainer.train(
        train_loader,
        val_loader,
        config.epochs,
        start_epoch,
        config.patience,
    )

    # Final evaluation on test set
    trainer.evaluate(
        test_loader,
        epoch=config.epochs,
        prefix="test",
        checkpoint_path="best_model_acc.pth",
    )

    # Log final metrics
    wandb.run.summary["best_val_acc"] = best_val_acc
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["total_epochs"] = epochs_trained

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

    data_path = "./data/adni-split-test"
    main(data_path)
