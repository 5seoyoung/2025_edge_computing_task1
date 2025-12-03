"""
Metrics and utility functions for model evaluation
"""
import torch
import torch.nn as nn
import time
from typing import Dict, Tuple


def count_params(model: nn.Module) -> int:
    """
    Count total number of trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_sparsity(model: nn.Module) -> float:
    """
    Calculate sparsity (ratio of zero weights) in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Sparsity ratio (0.0 to 1.0)
    """
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        if param is not None:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    if total_params == 0:
        return 0.0
    
    return zero_params / total_params


def calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        MAE value
    """
    return torch.mean(torch.abs(predictions - targets)).item()


def measure_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: str = "cuda",
    warmup_iterations: int = 10,
    measure_iterations: int = 100
) -> float:
    """
    Measure inference latency in milliseconds per video
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor (B, N, C, H, W)
        device: Device to run on ('cuda' or 'cpu')
        warmup_iterations: Number of warmup iterations
        measure_iterations: Number of iterations to measure
    
    Returns:
        Average latency in milliseconds per video
    """
    model.eval()
    model = model.to(device)
    sample_input = sample_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(sample_input)
    
    # Synchronize before measurement
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(measure_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(sample_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
    
    # Calculate average latency per video
    # If batch_size > 1, divide by batch size
    batch_size = sample_input.shape[0]
    avg_latency_per_video = sum(latencies) / len(latencies) / batch_size
    
    return avg_latency_per_video


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    criterion: nn.Module = None
) -> Dict[str, float]:
    """
    Evaluate model on dataset
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run on
        criterion: Loss function (optional)
    
    Returns:
        Dictionary with metrics (MAE, and optionally loss)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for videos, labels in data_loader:
            videos = videos.to(device)
            # Ensure labels are float32
            labels = labels.to(device).float()
            
            predictions = model(videos)
            all_predictions.append(predictions.cpu())
            all_targets.append(labels.cpu())
            
            if criterion is not None:
                loss = criterion(predictions, labels)
                total_loss += loss.item()
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    mae = calculate_mae(all_predictions, all_targets)
    
    results = {"MAE": mae}
    
    if criterion is not None:
        results["Loss"] = total_loss / len(data_loader)
    
    return results

