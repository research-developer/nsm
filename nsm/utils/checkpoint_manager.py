"""
Checkpoint management utilities for NSM experiments.

Provides consistent checkpoint saving/loading across local and Modal environments.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoint saving and loading.

    Features:
    - Consistent format across experiments
    - Metadata tracking (config, metrics, timestamp)
    - Best model tracking
    - Modal volume integration
    """

    def __init__(self, checkpoint_dir: str = "/checkpoints", experiment_name: str = "nsm"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints (Modal volume path or local)
            experiment_name: Experiment identifier
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        optimizer: Optional[torch.optim.Optimizer] = None,
        is_best: bool = False,
        prefix: str = ""
    ) -> Path:
        """
        Save model checkpoint with metadata.

        Args:
            model: PyTorch model
            epoch: Current epoch number
            metrics: Dictionary of validation metrics
            config: Training configuration
            optimizer: Optional optimizer state
            is_best: Whether this is the best model so far
            prefix: Optional prefix for checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if prefix:
            filename = f"{prefix}_{self.experiment_name}_epoch{epoch}_{timestamp}.pt"
        else:
            filename = f"{self.experiment_name}_epoch{epoch}_{timestamp}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': config,
            'timestamp': timestamp,
            'experiment_name': self.experiment_name
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")

        # Also save best model separately
        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"🌟 Saved best model: {best_path}")

        # Save metadata JSON for easy inspection
        metadata_path = checkpoint_path.with_suffix('.json')
        metadata = {
            'epoch': epoch,
            'metrics': metrics,
            'config': config,
            'timestamp': timestamp,
            'checkpoint_file': filename,
            'is_best': is_best
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load checkpoint into model.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to restore state
            device: Device to map tensors to

        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from epoch {checkpoint['epoch']}")

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Restored optimizer state")

        return checkpoint

    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Optional[Dict[str, Any]]:
        """
        Load best checkpoint for this experiment.

        Args:
            model: Model to load weights into
            optimizer: Optional optimizer
            device: Device to map to

        Returns:
            Checkpoint dict if found, None otherwise
        """
        best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"

        if not best_path.exists():
            print(f"⚠️  No best checkpoint found at {best_path}")
            return None

        return self.load_checkpoint(best_path, model, optimizer, device)

    def list_checkpoints(self) -> list:
        """List all checkpoints for this experiment."""
        pattern = f"{self.experiment_name}*.pt"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        return checkpoints

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get most recent checkpoint path."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]


def save_nsm_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    val_accuracy: float,
    config: Dict[str, Any],
    checkpoint_dir: str = "/checkpoints",
    experiment_name: str = "nsm",
    is_best: bool = False
) -> Path:
    """
    Convenience function for NSM checkpoint saving.

    Args:
        model: NSM model
        epoch: Training epoch
        val_accuracy: Validation accuracy
        config: Training config
        checkpoint_dir: Checkpoint directory
        experiment_name: Experiment name
        is_best: Is this the best model?

    Returns:
        Path to saved checkpoint
    """
    manager = CheckpointManager(checkpoint_dir, experiment_name)

    metrics = {'val_accuracy': val_accuracy}

    return manager.save_checkpoint(
        model=model,
        epoch=epoch,
        metrics=metrics,
        config=config,
        is_best=is_best
    )


def load_nsm_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Convenience function for NSM checkpoint loading.

    Args:
        model: NSM model to load into
        checkpoint_path: Path to checkpoint
        device: Device to map to

    Returns:
        Checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Infer experiment name from filename
    experiment_name = checkpoint_path.stem.split('_')[0]
    manager = CheckpointManager(checkpoint_path.parent, experiment_name)

    return manager.load_checkpoint(checkpoint_path, model, device=device)


# Export public API
__all__ = [
    'CheckpointManager',
    'save_nsm_checkpoint',
    'load_nsm_checkpoint'
]
