import os
import random

import wandb
import torch
import cv2

import numpy as np
import nibabel as nib
import torchio as tio
import torch.nn as nn
import torch.optim as optim
import torchvision.models.video as models

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    precision_recall_curve,
)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

if device == "mps":
    # Empty cache periodically during training to avoid memory fragmentation
    def empty_cache():
        try:
            torch.mps.empty_cache()
        except:
            print("MPS cache management not available")
            pass


class Config:
    """Centralized configuration management"""

    def __init__(self, **kwargs):
        # Default config
        self.architecture = "r3d_18"
        self.dataset = "MRI-AD-CN"
        self.epochs = 20
        self.batch_size = 2
        self.learning_rate = 0.0001
        self.optimizer = "AdamW"
        self.device = device
        self.input_dimensions = "128x128x128"
        self.freeze_layers = True
        self.data_augmentation = True
        self.patience = 5
        self.target_size = (128, 128, 128)
        self.checkpoint_dir = "checkpoints"
        self.cam_output_dir = "cam_visualizations"

        # Override defaults with provided arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Sync with wandb if a run is active
        if wandb.run is not None:
            # Update wandb config with the changed values
            for key, value in kwargs.items():
                wandb.config[key] = value

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

        for cls in np.unique(labels):
            mask = labels == cls
            if np.any(mask):
                metrics[f"class_{cls}_acc"] = accuracy_score(labels[mask], preds[mask])
            else:
                metrics[f"class_{cls}_acc"] = 0

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

        cm = confusion_matrix(labels, preds, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, _, _ = cm.ravel()
            metrics["specificity"] = tn / (tn + fp + 1e-10)
        else:
            metrics["specificity"] = 0.0

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

        conf_matrix = wandb.plot.confusion_matrix(
            preds=preds, y_true=labels, class_names=["CN", "AD"]
        )
        log_dict["confusion_matrix"] = conf_matrix

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

        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Split must be one of 'train', 'val', 'test', got {split}"
            )

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

        def load_from_dir(directory, label):
            files = [f for f in os.listdir(directory) if f.endswith(".nii.gz")]
            for file in files:
                self.samples.append(os.path.join(directory, file))
                self.labels.append(label)
            return len(files)

        ad_count = load_from_dir(ad_dir, 1)  # AD class
        cn_count = load_from_dir(cn_dir, 0)  # CN class

        if len(self.samples) == 0:
            raise ValueError(f"No .nii.gz files found in {ad_dir} or {cn_dir}")

        print(f"Loaded {ad_count} AD samples and {cn_count} CN samples")

    def _setup_transforms(self):
        if self.apply_augmentation:
            self.transforms = tio.Compose(
                [
                    tio.RandomNoise(mean=0.0, std=0.1, p=0.3),
                    tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.3),
                    tio.ZNormalization(),
                ]
            )
        else:
            self.transforms = tio.Compose([tio.ZNormalization()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # load the .nii.gz file
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
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

            img_data = self.transforms(img_data)

            # Convert to tensor if not already a tensor
            if not isinstance(img_data, torch.Tensor):
                img_data = torch.tensor(img_data, dtype=torch.float32)

            # Ensure the label is also a tensor
            label = torch.tensor(label, dtype=torch.long)

            # Return the original image path along with data and label for visualization later
            return img_data, label, img_path

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            raise


def create_datasets_and_loaders(
    data_root, batch_size, use_augmentation, target_size=(128, 128, 128)
):
    """Create datasets and data loaders."""
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
        "test_AD_samples": test_dataset.labels.count(1),
        "test_CN_samples": test_dataset.labels.count(0),
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
        model = model.to(device)

        trainable_params = model.count_trainable_params()
        total_params = model.count_total_params()
        frozen_params = total_params - trainable_params

        model_stats = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "frozen_percentage": (
                (frozen_params / total_params) if total_params > 0 else 0
            ),
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


class MRIModel(nn.Module):
    def __init__(self, num_classes=2, freeze_layers=True, architecture="r3d_18"):
        super(MRIModel, self).__init__()
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

        if freeze_layers:
            self._freeze_layers()

    def _freeze_layers(self):
        """Freeze most layers of the ResNet model, leaving only layer4 and fc unfrozen"""
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
            "wandb_run_id": wandb.run.id if wandb.run else None,
        }

        save_path = os.path.join(self.checkpoint_dir, filepath)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

        if is_best and metric_type and wandb.run:
            try:
                artifact_name = f"{wandb.run.name}-best_model_{metric_type}"  # Use run name for uniqueness
                artifact = wandb.Artifact(artifact_name, type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(
                    artifact, aliases=[f"best_{metric_type}", f"epoch_{epoch}"]
                )
                print(f"Model artifact saved to W&B (best {metric_type})!")
            except Exception as e:
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
        updated_acc = False
        updated_loss = False

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
            updated_acc = True

        # Use <= for loss to save even if accuracy didn't improve but loss did
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            # Only save if it's truly the best loss or if accuracy also improved
            # This prevents overwriting best_loss model if only acc improved
            if val_loss < best_val_loss or updated_acc:
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
                updated_loss = True

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

        return (
            updated_acc or updated_loss,
            best_val_acc,
            best_val_loss,
        )

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
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded model state dict from {filepath}")
            except RuntimeError as e:
                print(f"Error loading model state_dict: {e}")
                print("Attempting to load with strict=False")
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)

            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("Loaded optimizer state dict.")
                except Exception as e:
                    print(
                        f"Could not load optimizer state: {e}. Initializing optimizer from scratch."
                    )
            else:
                print(
                    "Optimizer state not found in checkpoint or optimizer not provided."
                )

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    print("Loaded scheduler state dict.")
                except Exception as e:
                    print(
                        f"Could not load scheduler state: {e}. Initializing scheduler from scratch."
                    )
            else:
                print(
                    "Scheduler state not found in checkpoint or scheduler not provided."
                )

            start_epoch = checkpoint.get("epoch", -1) + 1  # Use get with default
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"Resuming training from epoch {start_epoch}")
            print(
                f"Previous best val_acc: {best_val_acc:.4f}, best val_loss: {best_val_loss:.4f}"
            )
        else:
            print(
                f"No checkpoint found at {os.path.join(self.checkpoint_dir, filepath)}. Starting training from scratch."
            )
            start_epoch = 0
            best_val_acc = 0.0
            best_val_loss = float("inf")

        return start_epoch, best_val_acc, best_val_loss

    def get(self, filepath):
        """
        Get checkpoint data without loading it into a model.

        Args:
            filepath: Path to the checkpoint file relative to checkpoint_dir

        Returns:
            dict: The checkpoint dictionary or None if file doesn't exist
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filepath)

        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path)
                return checkpoint
            except Exception as e:
                print(f"Error loading checkpoint file {checkpoint_path}: {e}")
                return None
        else:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}")
            return None

    def get_wandb_run_id(self, filepath="checkpoint.pth"):
        """
        Get wandb run ID from checkpoint if available.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            str: wandb run ID or None if not available
        """
        checkpoint = self.get(filepath)
        if checkpoint and "wandb_run_id" in checkpoint:
            return checkpoint["wandb_run_id"]
        return None


class TrainerEngine:
    """Handles training, validation, and testing workflows."""

    def __init__(self, model, criterion, optimizer, scheduler, checkpoint_manager=None):
        """
        Initialize the trainer with model and training components.

        Args:
            model: The PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            checkpoint_manager: Optional CheckpointManager instance
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
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

        running_loss = torch.tensor(0.0, device=device)
        running_corrects = torch.tensor(0.0, device=device)

        for inputs, labels, _ in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad(set_to_none=True)  # More memory efficient

            # Forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = (running_loss / len(dataloader.dataset)).item()
        epoch_acc = (100 * running_corrects / len(dataloader.dataset)).item()

        if wandb.run:
            wandb.log(
                {
                    "train_loss": epoch_loss,
                    "train_acc": epoch_acc,
                    "epoch": epoch,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

        if device == "mps":
            empty_cache()

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
            print("-" * 10)

            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)

            val_loss, val_acc = self.evaluate(val_loader, epoch, prefix="val")

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr:.6f}")

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

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

            if not updated:
                early_stopping_counter += 1
                print(
                    f"No improvement detected. Early stopping counter: {early_stopping_counter}/{patience}"
                )
            else:
                early_stopping_counter = 0
                print("Model improved! Early stopping counter reset.")

            if early_stopping_counter >= patience:
                print(
                    f"Early stopping after {epoch+1} epochs ({early_stopping_counter} epochs without improvement)."
                )
                break

        print("\nTraining finished.")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")

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
            Tuple of (evaluation loss, evaluation accuracy in percentage)
        """
        if prefix is None:
            # Infer prefix from dataset split if possible
            try:
                prefix = dataloader.dataset.split
            except AttributeError:
                prefix = "eval"  # Default if split attribute doesn't exist

        if checkpoint_path:
            checkpoint = self.checkpoint_manager.get(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            val_acc = checkpoint.get("val_acc", 0.0)
            epoch = checkpoint.get("epoch", 0)
            print(
                f"Loaded checkpoint from epoch {epoch+1} with accuracy {val_acc:.2f}%"
            )

        self.model.eval()

        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        # Determine description for progress bar
        desc = (
            f"{prefix.capitalize()} Epoch {epoch+1}"
            if epoch is not None
            else f"{prefix.capitalize()} Evaluation"
        )

        # Collect predictions without computing gradients
        with torch.no_grad():
            for inputs, labels, _ in tqdm(dataloader, desc=desc):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                # Get probabilities for the positive class (AD)
                probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs.data, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        if len(all_labels) > 0:
            loss = running_loss / len(dataloader.dataset)
            metrics = MetricsManager.compute_metrics(all_labels, all_preds, all_probs)
            if wandb.run:
                MetricsManager.log_metrics(
                    all_labels, all_preds, all_probs, loss, epoch, prefix
                )
            acc = metrics["accuracy"] * 100  # Convert to percentage
        else:
            print(f"Warning: No samples processed during {prefix} evaluation.")
            loss = 0.0
            acc = 0.0

        return loss, acc


def visualize_grad_cam(model, dataloader, checkpoint_manager, config, num_images=5):
    """
    Generates Grad-CAM visualizations for a few images using the best model.

    Args:
        model: The MRIModel instance.
        dataloader: DataLoader (e.g., test_loader) to get images from.
        checkpoint_manager: CheckpointManager instance to load the best model.
        config: Configuration object containing output dir, device, etc.
        num_images: Number of images to generate visualizations for.
    """
    print("\nGenerating Grad-CAM visualizations...")

    best_acc_checkpoint = checkpoint_manager.get("best_model_acc.pth")
    if not best_acc_checkpoint:
        print("Could not find best_model_acc.pth. Skipping CAM visualization.")
        return
    try:
        model.load_state_dict(best_acc_checkpoint["model_state_dict"])
        print("Loaded best accuracy model for CAM visualization.")
    except Exception as e:
        print(f"Error loading best model state_dict for CAM: {e}")
        print("Attempting to load with strict=False")
        try:
            model.load_state_dict(best_acc_checkpoint["model_state_dict"], strict=False)
        except Exception as e_strict:
            print(
                f"Failed to load model even with strict=False: {e_strict}. Skipping CAM."
            )
            return

    model.eval()

    target_layers = [model.resnet.layer4[-1]]

    print("Initializing GradCAM...")
    try:
        cam = GradCAM(model=model, target_layers=target_layers)
    except Exception as e:
        print(f"Error initializing GradCAM: {e}")
        return

    cam_output_dir = config.cam_output_dir
    os.makedirs(cam_output_dir, exist_ok=True)

    images_processed = 0
    subjects_visualized = {
        "AD_correct": 0,
        "CN_correct": 0,
        "AD_incorrect": 0,
        "CN_incorrect": 0,
    }
    max_per_category = 3

    for inputs, labels, img_paths in dataloader:
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        for i in range(inputs.size(0)):
            input_tensor = inputs[i : i + 1].to(config.device)
            label = labels[i].item()
            img_path = img_paths[i]
            base_filename = os.path.basename(img_path).replace(".nii.gz", "")

            # Get model prediction
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output.data, 1)
                prediction = pred.item()

            category = f"{'AD' if label == 1 else 'CN'}_{'correct' if label == prediction else 'incorrect'}"

            # Limit the number of visualizations per category
            if subjects_visualized[category] >= max_per_category:
                continue

            # Define targets for CAM
            targets = [ClassifierOutputTarget(label)]

            try:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            except Exception as e:
                print(f"Error generating CAM for {img_path}: {e}")
                continue
            if grayscale_cam is None or grayscale_cam.ndim == 0:
                continue
            grayscale_cam = grayscale_cam[0, :]  # Shape [D, H, W]

            # --- Visualize Multiple Slices ---
            # Define key slices (e.g., around hippocampus) or a range
            # slice_indices_to_visualize = [
            #     grayscale_cam.shape[0] // 3,
            #     grayscale_cam.shape[0] // 2,
            #     2 * grayscale_cam.shape[0] // 3,
            # ]
            slice_indices_to_visualize = [55, 65, 75]

            for slice_idx in slice_indices_to_visualize:
                cam_slice = grayscale_cam[slice_idx, :, :]
                img_slice_tensor = input_tensor[0, 0, slice_idx, :, :].cpu().numpy()

                img_slice_normalized = cv2.normalize(
                    img_slice_tensor, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                img_slice_bgr = cv2.cvtColor(img_slice_normalized, cv2.COLOR_GRAY2BGR)

                visualization = show_cam_on_image(
                    img_slice_bgr / 255.0, cam_slice, use_rgb=False
                )

                output_filename = f"{base_filename}_true{label}_pred{prediction}_slice{slice_idx}_gradcam.png"
                output_path = os.path.join(cam_output_dir, output_filename)
                cv2.imwrite(output_path, visualization)
                # Log to wandb if desired
                # wandb.log({f"CAM_{category}_{base_filename}_slice{slice_idx}": wandb.Image(output_path)})

            subjects_visualized[category] += 1
            images_processed += 1  # Count subjects, not slices

    print(
        f"Finished generating Grad-CAM visualizations for {subjects_visualized} subjects."
    )


def main(data_path):
    config = Config(data_path=data_path)

    checkpoint_manager = CheckpointManager(config.checkpoint_dir)

    wandb_run_id = checkpoint_manager.get_wandb_run_id()

    wandb.init(
        project="mri-alzheimers-classification",
        config=config.to_dict(),
        id=wandb_run_id,  # Use existing run ID if available
        resume="must" if wandb_run_id else None,  # Resume if ID exists
    )
    config.update(**wandb.config)
    wandb.config.update(config.to_dict())

    train_dataset, train_loader, val_loader, test_loader, dataset_stats = (
        create_datasets_and_loaders(
            config.data_path,
            config.batch_size,
            config.data_augmentation,
            target_size=config.target_size,
        )
    )

    config.update(**dataset_stats)
    wandb.config.update(dataset_stats)

    model, model_stats = ModelManager.create_model(config)

    print(f"Model Architecture: {config.architecture}")
    print(f"Total parameters: {model_stats['total_params']:,}")
    print(
        f"Trainable parameters: {model_stats['trainable_params']:,} "
        f"({model_stats['trainable_params']/model_stats['total_params']:.2%})"
    )
    print(
        f"Frozen parameters: {model_stats['frozen_params']:,} "
        f"({model_stats['frozen_params']/model_stats['total_params']:.2%})"
    )

    config.update(**model_stats)
    wandb.config.update(model_stats)

    wandb.watch(model, log="all", log_freq=10)

    criterion, optimizer, scheduler = ModelManager.setup_training_components(
        model, train_dataset, config
    )

    start_epoch, best_val_acc, best_val_loss = checkpoint_manager.load(
        model, optimizer, scheduler
    )

    trainer = TrainerEngine(model, criterion, optimizer, scheduler, checkpoint_manager)
    trainer.best_val_acc = best_val_acc
    trainer.best_val_loss = best_val_loss

    # --- Training ---
    print("\nStarting Training...")
    epochs_trained, final_best_val_acc, final_best_val_loss = trainer.train(
        train_loader,
        val_loader,
        config.epochs,
        start_epoch,
        config.patience,
    )

    # --- Testing ---
    print("\nStarting Testing using best accuracy model...")
    trainer.evaluate(
        test_loader,
        epoch=None,  # No specific epoch needed for final test
        prefix="test",
        checkpoint_path="best_model_acc.pth",  # Evaluate the best model based on validation accuracy
    )

    wandb.run.summary["best_val_acc"] = final_best_val_acc
    wandb.run.summary["best_val_loss"] = final_best_val_loss
    wandb.run.summary["total_epochs"] = epochs_trained

    # --- Grad-CAM Visualization ---
    visualize_grad_cam(model, test_loader, checkpoint_manager, config, num_images=10)

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

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("cam_visualizations"):
        os.makedirs("cam_visualizations")

    data_path = "./data/adni-cv-splits/fold_2"
    if not os.path.isdir(data_path):
        print(f"Error: Data path not found at {data_path}")
        print("Please ensure the data path is correct relative to the script location.")
        exit()

    main(data_path)
