"""
Configuration file for EF Regression Model with Pruning Experiments
"""
import os
from pathlib import Path

class Config:
    # Data paths
    # EchoNet-Dynamic 데이터셋 경로 설정 (사용자 환경에 맞게 수정 필요)
    DATA_ROOT = Path("data/echonet_dynamic")  # 기본 경로, config.py에서 수정 가능
    
    VIDEO_DIR = DATA_ROOT / "Videos"
    FILELIST_PATH = DATA_ROOT / "FileList.csv"
    
    # Model parameters
    NUM_FRAMES = 32  # Number of frames to sample from video
    IMAGE_SIZE = 112  # Resize to 112x112
    BATCH_SIZE = 16  # Batch size (GPU: 16-32, CPU: 8)
    NUM_WORKERS = 4  # DataLoader workers (auto-adjusted in main.py)
    
    # Training parameters
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DEVICE = "cuda"  # Auto-detected in main.py (cuda if available, else cpu)
    
    # Pruning parameters
    UNSTRUCTURED_PRUNING_RATIOS = [0.5, 0.8, 0.9]  # Unstructured pruning ratios
    STRUCTURED_PRUNING_RATIOS = [0.3, 0.5, 0.7]  # Structured pruning ratios
    FINE_TUNE_EPOCHS = 2  # Epochs for fine-tuning after pruning
    
    # Paths
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR = Path("results")
    
    # Latency measurement
    LATENCY_WARMUP_ITERATIONS = 10
    LATENCY_MEASURE_ITERATIONS = 100
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        cls.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True, parents=True)

