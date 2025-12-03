"""
Training and evaluation functions
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import os
from pathlib import Path


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str = "cuda",
    epoch: int = 0
) -> Dict[str, float]:
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        epoch: Current epoch number
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos = videos.to(device)
        # Ensure labels are float32 (in case they're float64)
        labels = labels.to(device).float()
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(videos)
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return {"Loss": avg_loss}


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate model on validation set
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Dictionary with evaluation metrics (MAE, Loss)
    """
    from metrics import evaluate_model
    
    return evaluate_model(model, val_loader, device, criterion)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Path,
    is_best: bool = False
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save regular checkpoint
    torch.save(checkpoint, filepath)
    
    # Save best model separately
    if is_best:
        best_path = filepath.parent / f"best_{filepath.name}"
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    filepath: Path,
    device: str = "cuda"
) -> Tuple[int, Dict[str, float]]:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint
        device: Device to load on
    
    Returns:
        Tuple of (epoch, metrics)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    
    return epoch, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str = "cuda",
    checkpoint_dir: Path = Path("checkpoints"),
    resume_from: Path = None
) -> nn.Module:
    """
    Full training loop
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to run on
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from (optional)
    
    Returns:
        Trained model
    """
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_mae = float('inf')
    
    if resume_from is not None and resume_from.exists():
        print(f"Resuming from checkpoint: {resume_from}")
        start_epoch, metrics = load_checkpoint(model, optimizer, resume_from, device)
        best_mae = metrics.get('MAE', float('inf'))
    
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Train Loss: {train_metrics['Loss']:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val MAE: {val_metrics['MAE']:.4f}, Val Loss: {val_metrics['Loss']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['Loss'])
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != learning_rate:  # Only print if LR changed
            print(f"Learning rate updated to: {current_lr:.2e}")
        
        # Save checkpoint
        is_best = val_metrics['MAE'] < best_mae
        if is_best:
            best_mae = val_metrics['MAE']
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        save_checkpoint(
            model, optimizer, epoch, val_metrics, checkpoint_path, is_best=is_best
        )
    
    print(f"\nTraining completed! Best Val MAE: {best_mae:.4f}")
    return model

