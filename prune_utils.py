"""
Pruning utilities for Unstructured and Structured Pruning
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Dict, Tuple
from pathlib import Path


def unstructured_pruning(
    model: nn.Module,
    pruning_ratio: float,
    method: str = "l1_unstructured"
) -> nn.Module:
    """
    Apply unstructured pruning (L1 magnitude-based) to model
    
    Args:
        model: PyTorch model
        pruning_ratio: Ratio of weights to prune (0.0 to 1.0)
        method: Pruning method ('l1_unstructured' or 'random_unstructured')
    
    Returns:
        Pruned model
    """
    # Parameters to prune: Conv2d and Linear layers
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Skip if it's the regression head (we might want to keep it)
            # For now, we'll prune all layers
            parameters_to_prune.append((module, 'weight'))
    
    if len(parameters_to_prune) == 0:
        print("Warning: No parameters found to prune")
        return model
    
    # Global unstructured pruning
    if method == "l1_unstructured":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio
        )
    elif method == "random_unstructured":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=pruning_ratio
        )
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    # Make pruning permanent (remove masks and zero out weights)
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    print(f"Applied unstructured pruning with ratio {pruning_ratio:.2f}")
    return model


def structured_pruning(
    model: nn.Module,
    pruning_ratio: float,
    dim: int = 0
) -> nn.Module:
    """
    Apply structured pruning (channel-wise) to Conv2d layers
    
    Args:
        model: PyTorch model
        pruning_ratio: Ratio of channels to prune (0.0 to 1.0)
        dim: Dimension to prune (0 for output channels, 1 for input channels)
            For Conv2d, dim=0 prunes output channels
    
    Returns:
        Pruned model
    """
    # Only prune Conv2d layers (not Linear layers for structured pruning)
    # Structured pruning must be applied layer-wise (not globally)
    pruned_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Apply LnStructured pruning to each Conv2d layer
            prune.ln_structured(
                module,
                name='weight',
                amount=pruning_ratio,
                n=2,  # L2 norm
                dim=dim
            )
            pruned_layers += 1
    
    if pruned_layers == 0:
        print("Warning: No Conv2d layers found to prune")
        return model
    
    # Make pruning permanent (remove masks and zero out weights)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')
    
    print(f"Applied structured pruning with ratio {pruning_ratio:.2f} (dim={dim}) to {pruned_layers} Conv2d layers")
    return model


def fine_tune_pruned_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str = "cuda"
) -> nn.Module:
    """
    Fine-tune pruned model for a few epochs
    
    Args:
        model: Pruned PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        device: Device to run on
    
    Returns:
        Fine-tuned model
    """
    from train import train_one_epoch, evaluate
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"\nFine-tuning pruned model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Fine-tune Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_metrics['Loss']:.4f}, "
              f"Val MAE: {val_metrics['MAE']:.4f}")
    
    return model


def apply_pruning_experiment(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    pruning_type: str,
    pruning_ratios: List[float],
    fine_tune_epochs: int,
    learning_rate: float,
    device: str = "cuda",
    sample_input: torch.Tensor = None
) -> List[Dict]:
    """
    Apply pruning experiments and collect results
    
    Args:
        model: Base model (will be copied for each experiment)
        train_loader: Training data loader
        val_loader: Validation data loader
        pruning_type: 'unstructured' or 'structured'
        pruning_ratios: List of pruning ratios to test
        fine_tune_epochs: Number of epochs for fine-tuning
        learning_rate: Learning rate for fine-tuning
        device: Device to run on
        sample_input: Sample input for latency measurement
    
    Returns:
        List of dictionaries with results for each pruning ratio
    """
    from metrics import count_params, calculate_sparsity, evaluate_model, measure_latency
    import copy
    
    results = []
    
    for ratio in pruning_ratios:
        print(f"\n{'='*60}")
        print(f"{pruning_type.upper()} Pruning - Ratio: {ratio:.2f}")
        print(f"{'='*60}")
        
        # Copy model for this experiment
        pruned_model = copy.deepcopy(model)
        
        # Apply pruning
        if pruning_type == "unstructured":
            pruned_model = unstructured_pruning(pruned_model, ratio)
        elif pruning_type == "structured":
            pruned_model = structured_pruning(pruned_model, ratio)
        else:
            raise ValueError(f"Unknown pruning type: {pruning_type}")
        
        # Calculate metrics before fine-tuning
        params_before = count_params(pruned_model)
        sparsity_before = calculate_sparsity(pruned_model)
        
        print(f"Before fine-tuning:")
        print(f"  Parameters: {params_before:,}")
        print(f"  Sparsity: {sparsity_before:.4f}")
        
        # Fine-tune
        if fine_tune_epochs > 0:
            pruned_model = fine_tune_pruned_model(
                pruned_model, train_loader, val_loader,
                fine_tune_epochs, learning_rate, device
            )
        
        # Evaluate
        eval_results = evaluate_model(pruned_model, val_loader, device)
        mae = eval_results['MAE']
        
        # Calculate final metrics
        params_after = count_params(pruned_model)
        sparsity_after = calculate_sparsity(pruned_model)
        
        # Measure latency
        if sample_input is not None:
            latency = measure_latency(pruned_model, sample_input, device)
        else:
            latency = None
        
        result = {
            'pruning_type': pruning_type,
            'pruning_ratio': ratio,
            'num_params': params_after,
            'sparsity': sparsity_after,
            'MAE': mae,
            'latency_ms_per_video': latency
        }
        
        results.append(result)
        
        print(f"\nResults:")
        print(f"  Parameters: {params_after:,}")
        print(f"  Sparsity: {sparsity_after:.4f}")
        print(f"  MAE: {mae:.4f}")
        if latency is not None:
            print(f"  Latency: {latency:.4f} ms/video")
    
    return results

