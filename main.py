"""
Main script for EF Regression Model with Pruning Experiments
"""
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import random
import numpy as np

from config import Config
from dataset import create_data_loaders
from model import EFRegressionModel
from train import train_model, load_checkpoint
from prune_utils import apply_pruning_experiment
from metrics import count_params, calculate_sparsity, evaluate_model, measure_latency


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main execution function"""
    print("="*70)
    print("Structured / Unstructured Pruning Comparison Experiment")
    print("="*70)
    
    # Set random seed for reproducibility
    set_seed(42)
    print("Random seed set to 42 for reproducibility")
    
    # Setup directories
    Config.setup_directories()
    
    # Check device and GPU info
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU Available!")
        print(f"   Device: {device}")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Batch Size: {Config.BATCH_SIZE}")
    else:
        device = torch.device("cpu")
        print(f"⚠️  GPU not available, using CPU")
        print(f"   Device: {device}")
        print(f"   Batch Size: {Config.BATCH_SIZE}")
    
    # Auto-adjust NUM_WORKERS based on system recommendation
    import os
    import multiprocessing
    max_workers = min(Config.NUM_WORKERS, multiprocessing.cpu_count(), 4)  # PyTorch recommends max 4
    if max_workers < Config.NUM_WORKERS:
        print(f"   ⚠️  Adjusted NUM_WORKERS: {Config.NUM_WORKERS} -> {max_workers} (system recommendation)")
    else:
        print(f"   Num Workers: {max_workers}")
    
    # Load data
    print("\n" + "="*70)
    print("Loading Dataset")
    print("="*70)
    
    if not Config.VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {Config.VIDEO_DIR}")
    if not Config.FILELIST_PATH.exists():
        raise FileNotFoundError(f"FileList.csv not found: {Config.FILELIST_PATH}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        video_dir=Config.VIDEO_DIR,
        filelist_path=Config.FILELIST_PATH,
        num_frames=Config.NUM_FRAMES,
        image_size=Config.IMAGE_SIZE,
        batch_size=Config.BATCH_SIZE,
        num_workers=max_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Get sample input for latency measurement
    sample_videos, _ = next(iter(val_loader))
    sample_input = sample_videos[:1].to(device)  # Single video for latency
    
    # ========== Baseline Model Training ==========
    print("\n" + "="*70)
    print("Baseline Model Training")
    print("="*70)
    
    baseline_model = EFRegressionModel(
        num_frames=Config.NUM_FRAMES,
        pretrained=True
    )
    
    baseline_checkpoint_path = Config.CHECKPOINT_DIR / "baseline_best.pth"
    
    # Train baseline if checkpoint doesn't exist
    if not baseline_checkpoint_path.exists():
        print("Training baseline model...")
        baseline_model = train_model(
            model=baseline_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=Config.NUM_EPOCHS,
            learning_rate=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            device=device,
            checkpoint_dir=Config.CHECKPOINT_DIR
        )
        
        # Save final baseline model
        torch.save(baseline_model.state_dict(), baseline_checkpoint_path)
    else:
        print(f"Loading baseline model from {baseline_checkpoint_path}")
        baseline_model.load_state_dict(torch.load(baseline_checkpoint_path, map_location=device))
        baseline_model = baseline_model.to(device)
    
    # Evaluate baseline
    print("\nEvaluating baseline model...")
    baseline_model.eval()
    baseline_results = evaluate_model(baseline_model, test_loader, device)
    baseline_params = count_params(baseline_model)
    baseline_sparsity = calculate_sparsity(baseline_model)
    baseline_latency = measure_latency(
        baseline_model, sample_input, device,
        Config.LATENCY_WARMUP_ITERATIONS,
        Config.LATENCY_MEASURE_ITERATIONS
    )
    
    baseline_summary = {
        'model_type': 'baseline',
        'num_params': baseline_params,
        'sparsity': baseline_sparsity,
        'MAE': baseline_results['MAE'],
        'latency_ms_per_video': baseline_latency
    }
    
    print(f"\nBaseline Results:")
    print(f"  Parameters: {baseline_params:,}")
    print(f"  Sparsity: {baseline_sparsity:.4f}")
    print(f"  MAE: {baseline_results['MAE']:.4f}")
    print(f"  Latency: {baseline_latency:.4f} ms/video")
    
    # ========== Unstructured Pruning Experiments ==========
    print("\n" + "="*70)
    print("Unstructured Pruning Experiments")
    print("="*70)
    
    unstructured_results = apply_pruning_experiment(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        pruning_type="unstructured",
        pruning_ratios=Config.UNSTRUCTURED_PRUNING_RATIOS,
        fine_tune_epochs=Config.FINE_TUNE_EPOCHS,
        learning_rate=Config.LEARNING_RATE * 0.1,  # Lower LR for fine-tuning
        device=device,
        sample_input=sample_input
    )
    
    # ========== Structured Pruning Experiments ==========
    print("\n" + "="*70)
    print("Structured Pruning Experiments")
    print("="*70)
    
    # Reload baseline for structured pruning (since unstructured modifies it)
    baseline_model.load_state_dict(torch.load(baseline_checkpoint_path, map_location=device))
    baseline_model = baseline_model.to(device)
    
    structured_results = apply_pruning_experiment(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        pruning_type="structured",
        pruning_ratios=Config.STRUCTURED_PRUNING_RATIOS,
        fine_tune_epochs=Config.FINE_TUNE_EPOCHS,
        learning_rate=Config.LEARNING_RATE * 0.1,
        device=device,
        sample_input=sample_input
    )
    
    # ========== Save Results ==========
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)
    
    all_results = {
        'baseline': baseline_summary,
        'unstructured_pruning': unstructured_results,
        'structured_pruning': structured_results,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_frames': Config.NUM_FRAMES,
            'image_size': Config.IMAGE_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'num_epochs': Config.NUM_EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
            'fine_tune_epochs': Config.FINE_TUNE_EPOCHS
        }
    }
    
    # Save as JSON
    results_path = Config.RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Create summary table
    print("\n" + "="*70)
    print("Summary Table")
    print("="*70)
    
    summary_data = []
    
    # Baseline
    summary_data.append({
        'Model': 'Baseline',
        'Pruning Ratio': '-',
        'Parameters': baseline_summary['num_params'],
        'Sparsity': f"{baseline_summary['sparsity']:.4f}",
        'MAE': f"{baseline_summary['MAE']:.4f}",
        'Latency (ms/video)': f"{baseline_summary['latency_ms_per_video']:.4f}"
    })
    
    # Unstructured pruning
    for result in unstructured_results:
        summary_data.append({
            'Model': 'Unstructured',
            'Pruning Ratio': f"{result['pruning_ratio']:.2f}",
            'Parameters': result['num_params'],
            'Sparsity': f"{result['sparsity']:.4f}",
            'MAE': f"{result['MAE']:.4f}",
            'Latency (ms/video)': f"{result['latency_ms_per_video']:.4f}" if result['latency_ms_per_video'] else 'N/A'
        })
    
    # Structured pruning
    for result in structured_results:
        summary_data.append({
            'Model': 'Structured',
            'Pruning Ratio': f"{result['pruning_ratio']:.2f}",
            'Parameters': result['num_params'],
            'Sparsity': f"{result['sparsity']:.4f}",
            'MAE': f"{result['MAE']:.4f}",
            'Latency (ms/video)': f"{result['latency_ms_per_video']:.4f}" if result['latency_ms_per_video'] else 'N/A'
        })
    
    # Print table
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    
    # Save CSV
    csv_path = Config.RESULTS_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to: {csv_path}")
    
    print("\n" + "="*70)
    print("Experiment Completed!")
    print("="*70)


if __name__ == "__main__":
    main()

