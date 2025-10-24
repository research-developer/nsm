"""
Generic training loop for NSM models with cycle consistency and task losses.

Implements training infrastructure with:
- Cycle consistency loss (WHY/WHAT reconstruction)
- Task-specific loss (classification/regression/link_prediction)
- Temperature annealing for confidence propagation
- Gradient clipping and monitoring
- Checkpointing and early stopping
- Tensorboard/WandB logging support

Usage:
    trainer = NSMTrainer(model, optimizer, device)
    trainer.train(train_loader, val_loader, epochs=100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path
import time
import json
from tqdm import tqdm

from ..models.confidence.temperature import TemperatureScheduler


class NSMTrainer:
    """Generic trainer for NSM models with cycle consistency.

    Combines task loss with cycle consistency loss:
        L_total = L_task + λ_cycle * L_cycle

    Args:
        model (nn.Module): NSMModel instance
        optimizer (Optimizer): PyTorch optimizer
        device (torch.device): Training device
        cycle_loss_weight (float): Weight for cycle consistency loss
        gradient_clip (float): Max gradient norm for clipping
        temp_scheduler (TemperatureScheduler, optional): Temperature annealing
        checkpoint_dir (str): Directory for saving checkpoints
        log_interval (int): Steps between logging
        use_wandb (bool): Enable Weights & Biases logging
        use_tensorboard (bool): Enable Tensorboard logging

    Example:
        >>> model = NSMModel(node_features=64, num_relations=16, num_classes=2)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> trainer = NSMTrainer(model, optimizer, device='cuda')
        >>>
        >>> train_metrics = trainer.train_epoch(train_loader, epoch=0)
        >>> val_metrics = trainer.validate(val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        cycle_loss_weight: float = 0.1,
        gradient_clip: float = 1.0,
        temp_scheduler: Optional[TemperatureScheduler] = None,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.cycle_loss_weight = cycle_loss_weight
        self.gradient_clip = gradient_clip
        self.temp_scheduler = temp_scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        # Class weights for balanced loss (anti-collapse)
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = class_weights.to(device)

        # Logging
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'tensorboard'))

        if use_wandb:
            import wandb
            self.wandb = wandb

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0

    def compute_task_loss(
        self,
        output: Dict[str, Any],
        labels: torch.Tensor,
        task_type: str
    ) -> torch.Tensor:
        """Compute task-specific loss.

        Args:
            output (dict): Model output containing 'logits'
            labels (Tensor): Ground truth labels
            task_type (str): 'classification', 'regression', or 'link_prediction'

        Returns:
            Tensor: Task loss (scalar)
        """
        logits = output['logits']

        if task_type == 'classification':
            return F.cross_entropy(logits, labels, weight=self.class_weights)
        elif task_type == 'regression':
            return F.mse_loss(logits.squeeze(), labels.float())
        elif task_type == 'link_prediction':
            # Binary/multi-class classification for edge existence
            return F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def compute_total_loss(
        self,
        output: Dict[str, Any],
        labels: torch.Tensor,
        task_type: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss: task + cycle consistency.

        Args:
            output (dict): Model output with 'logits' and 'cycle_loss'
            labels (Tensor): Ground truth labels
            task_type (str): Task type

        Returns:
            Tuple containing:
            - total_loss (Tensor): Combined loss for backprop
            - loss_dict (dict): Individual loss components for logging
        """
        # Check if model is using dual-pass mode
        use_dual_pass = hasattr(self.model, 'use_dual_pass') and self.model.use_dual_pass

        if use_dual_pass and 'logits_abstract' in output and 'logits_concrete' in output:
            # DUAL-PASS MODE: Compute losses for all three predictions
            task_loss_abstract = self.compute_task_loss({'logits': output['logits_abstract']}, labels, task_type)
            task_loss_concrete = self.compute_task_loss({'logits': output['logits_concrete']}, labels, task_type)
            task_loss_fused = self.compute_task_loss(output, labels, task_type)  # Uses output['logits'] which is fused

            # Combined task loss (fused is primary, abstract/concrete are auxiliary)
            task_loss = (
                0.5 * task_loss_fused +      # Primary: fused prediction
                0.25 * task_loss_abstract +  # Auxiliary: abstract prediction
                0.25 * task_loss_concrete    # Auxiliary: concrete prediction
            )

            cycle_loss = output.get('cycle_loss', torch.tensor(0.0, device=self.device))
            total_loss = task_loss + self.cycle_loss_weight * cycle_loss

            loss_dict = {
                'task_loss': task_loss.item(),
                'task_loss_abstract': task_loss_abstract.item(),
                'task_loss_concrete': task_loss_concrete.item(),
                'task_loss_fused': task_loss_fused.item(),
                'cycle_loss': cycle_loss.item(),
                'total_loss': total_loss.item()
            }
        else:
            # SINGLE-PASS MODE (original behavior)
            task_loss = self.compute_task_loss(output, labels, task_type)
            cycle_loss = output.get('cycle_loss', torch.tensor(0.0, device=self.device))

            total_loss = task_loss + self.cycle_loss_weight * cycle_loss

            loss_dict = {
                'task_loss': task_loss.item(),
                'cycle_loss': cycle_loss.item(),
                'total_loss': total_loss.item()
            }

        return total_loss, loss_dict

    def train_step(
        self,
        batch: Dict[str, Any],
        task_type: str
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch (dict): Batch data with 'x', 'edge_index', 'edge_type', 'edge_attr', 'y'
            task_type (str): Task type

        Returns:
            dict: Loss components and metrics
        """
        self.model.train()

        # Move batch to device
        x = batch['x'].to(self.device)
        edge_index = batch['edge_index'].to(self.device)
        edge_type = batch['edge_type'].to(self.device)
        edge_attr = batch.get('edge_attr', None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        batch_idx = batch.get('batch', None)
        if batch_idx is not None:
            batch_idx = batch_idx.to(self.device)
        labels = batch['y'].to(self.device)

        # Update temperature if scheduled
        if self.temp_scheduler is not None:
            temp = self.temp_scheduler.get_temperature()
            if hasattr(self.model, 'hierarchical'):
                self.model.hierarchical.semiring.temperature = temp

        # Forward pass
        output = self.model(x, edge_index, edge_type, edge_attr, batch_idx)

        # Compute loss
        total_loss, loss_dict = self.compute_total_loss(output, labels, task_type)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip
        )
        loss_dict['grad_norm'] = grad_norm.item()

        self.optimizer.step()
        self.global_step += 1

        return loss_dict

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        task_type: str = 'classification'
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader (DataLoader): Training data
            epoch (int): Current epoch number
            task_type (str): Task type

        Returns:
            dict: Averaged metrics over epoch
        """
        self.model.train()

        epoch_metrics = {
            'task_loss': 0.0,
            'cycle_loss': 0.0,
            'total_loss': 0.0,
            'grad_norm': 0.0
        }

        num_batches = len(train_loader)

        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Training step
                loss_dict = self.train_step(batch, task_type)

                # Accumulate metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += loss_dict[key]

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'cycle': f"{loss_dict['cycle_loss']:.4f}",
                    'grad': f"{loss_dict['grad_norm']:.2f}"
                })

                # Log to tensorboard/wandb
                if self.global_step % self.log_interval == 0:
                    self._log_metrics(loss_dict, prefix='train', step=self.global_step)

        # Average over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        # Temperature annealing
        if self.temp_scheduler is not None:
            new_temp = self.temp_scheduler.step()
            epoch_metrics['temperature'] = new_temp

        return epoch_metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        task_type: str = 'classification',
        compute_metrics: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Validate model on validation set.

        Args:
            val_loader (DataLoader): Validation data
            task_type (str): Task type
            compute_metrics (callable, optional): Function to compute additional metrics

        Returns:
            dict: Validation metrics
        """
        self.model.eval()

        val_metrics = {
            'task_loss': 0.0,
            'cycle_loss': 0.0,
            'total_loss': 0.0
        }

        all_preds = []
        all_labels = []

        num_batches = len(val_loader)

        for batch in tqdm(val_loader, desc="Validation"):
            # Move batch to device
            x = batch['x'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_type = batch['edge_type'].to(self.device)
            edge_attr = batch.get('edge_attr', None)
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)
            batch_idx = batch.get('batch', None)
            if batch_idx is not None:
                batch_idx = batch_idx.to(self.device)
            labels = batch['y'].to(self.device)

            # Forward pass
            output = self.model(x, edge_index, edge_type, edge_attr, batch_idx)

            # Compute loss
            _, loss_dict = self.compute_total_loss(output, labels, task_type)

            # Accumulate metrics
            for key in val_metrics:
                val_metrics[key] += loss_dict[key]

            # Store predictions for metric computation
            all_preds.append(output['logits'].cpu())
            all_labels.append(labels.cpu())

        # Average over validation set
        for key in val_metrics:
            val_metrics[key] /= num_batches

        # Compute additional metrics (accuracy, etc.)
        if compute_metrics is not None:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            additional_metrics = compute_metrics(all_preds, all_labels, task_type)
            val_metrics.update(additional_metrics)

        return val_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        task_type: str = 'classification',
        compute_metrics: Optional[Callable] = None,
        early_stopping_patience: int = 20,
        save_best_only: bool = True
    ) -> Dict[str, Any]:
        """Full training loop with validation and checkpointing.

        Args:
            train_loader (DataLoader): Training data
            val_loader (DataLoader): Validation data
            epochs (int): Number of training epochs
            task_type (str): Task type
            compute_metrics (callable, optional): Additional metrics function
            early_stopping_patience (int): Epochs without improvement before stopping
            save_best_only (bool): Only save checkpoint when validation improves

        Returns:
            dict: Training history
        """
        history = {
            'train': [],
            'val': []
        }

        best_checkpoint_path = None

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch, task_type)

            # Validate
            val_metrics = self.validate(val_loader, task_type, compute_metrics)

            # Timing
            epoch_time = time.time() - start_time
            train_metrics['epoch_time'] = epoch_time
            val_metrics['epoch_time'] = epoch_time

            # Store history
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)

            # Log epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(task: {train_metrics['task_loss']:.4f}, "
                  f"cycle: {train_metrics['cycle_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['total_loss']:.4f} "
                  f"(task: {val_metrics['task_loss']:.4f}, "
                  f"cycle: {val_metrics['cycle_loss']:.4f})")

            if 'accuracy' in val_metrics:
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")

            # Log to wandb/tensorboard
            self._log_metrics(train_metrics, prefix='train_epoch', step=epoch)
            self._log_metrics(val_metrics, prefix='val_epoch', step=epoch)

            # Checkpointing
            val_loss = val_metrics['total_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                if save_best_only or not save_best_only:
                    checkpoint_path = self.save_checkpoint(epoch, val_metrics)
                    best_checkpoint_path = checkpoint_path
                    print(f"  ✓ Saved best checkpoint: {checkpoint_path}")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

        # Save final history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best checkpoint: {best_checkpoint_path}")

        return history

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> Path:
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch
            metrics (dict): Validation metrics

        Returns:
            Path: Checkpoint file path
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }

        if self.temp_scheduler is not None:
            checkpoint['temp_scheduler'] = self.temp_scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Also save as 'best_model.pt'
        best_path = self.checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            checkpoint_path (str): Path to checkpoint file

        Returns:
            dict: Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if 'temp_scheduler' in checkpoint and self.temp_scheduler is not None:
            self.temp_scheduler.load_state_dict(checkpoint['temp_scheduler'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")

        return checkpoint

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str,
        step: int
    ) -> None:
        """Log metrics to tensorboard/wandb.

        Args:
            metrics (dict): Metrics to log
            prefix (str): Metric name prefix
            step (int): Global step
        """
        if self.use_tensorboard and hasattr(self, 'writer'):
            for key, value in metrics.items():
                self.writer.add_scalar(f'{prefix}/{key}', value, step)

        if self.use_wandb:
            wandb_metrics = {f'{prefix}/{key}': value for key, value in metrics.items()}
            self.wandb.log(wandb_metrics, step=step)


def compute_classification_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    task_type: str
) -> Dict[str, float]:
    """Compute classification metrics (accuracy, F1, etc.).

    Args:
        preds (Tensor): Predicted logits [batch_size, num_classes]
        labels (Tensor): Ground truth labels [batch_size]
        task_type (str): Task type

    Returns:
        dict: Computed metrics
    """
    metrics = {}

    if task_type == 'classification':
        # Multi-class classification
        pred_labels = torch.argmax(preds, dim=1)
        correct = (pred_labels == labels).sum().item()
        total = labels.size(0)
        metrics['accuracy'] = correct / total

        # Per-class accuracy (if useful)
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = labels == label
            class_correct = (pred_labels[mask] == labels[mask]).sum().item()
            class_total = mask.sum().item()
            if class_total > 0:
                metrics[f'accuracy_class_{label.item()}'] = class_correct / class_total

    elif task_type == 'link_prediction':
        # Binary classification: Handle [batch_size, 2] logits OR [batch_size, 1] probabilities
        if preds.dim() == 2 and preds.size(1) == 2:
            # Two-class logits: apply argmax (like standard classification)
            pred_labels = torch.argmax(preds, dim=1)
        else:
            # Single probability: apply sigmoid threshold
            pred_labels = (torch.sigmoid(preds.squeeze()) > 0.5).long()

        # Labels should be [batch_size] with values 0 or 1
        correct = (pred_labels == labels).sum().item()
        total = labels.size(0)
        metrics['accuracy'] = correct / total

        # Per-class accuracy (class 0 = false link, class 1 = true link)
        for label_val in [0, 1]:
            mask = labels == label_val
            if mask.sum() > 0:
                class_correct = (pred_labels[mask] == labels[mask]).sum().item()
                class_total = mask.sum().item()
                metrics[f'accuracy_class_{label_val}'] = class_correct / class_total

    elif task_type == 'regression':
        # MSE and MAE
        mse = F.mse_loss(preds.squeeze(), labels.float())
        mae = F.l1_loss(preds.squeeze(), labels.float())
        metrics['mse'] = mse.item()
        metrics['mae'] = mae.item()

    return metrics
