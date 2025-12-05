#!/usr/bin/env python3
"""
GPU 서버 환경 확인 스크립트
"""
import torch
import sys
from pathlib import Path

def check_gpu():
    """GPU 및 환경 확인"""
    print("="*70)
    print("GPU Server Environment Check")
    print("="*70)
    
    # PyTorch 버전
    print(f"\n📦 PyTorch Version: {torch.__version__}")
    
    # CUDA 확인
    print(f"\n🔍 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"      Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"      Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")
    else:
        print("     CUDA not available. Please check:")
        print("      - CUDA drivers installed")
        print("      - PyTorch built with CUDA support")
        print("      - GPU accessible")
    
    # Python 버전
    print(f"\n Python Version: {sys.version}")
    
    # 의존성 확인
    print(f"\n Dependencies:")
    try:
        import torchvision
        print(f"   torchvision: {torchvision.__version__}")
    except ImportError:
        print(f"    torchvision: Not installed")
    
    try:
        import cv2
        print(f"    opencv-python: {cv2.__version__}")
    except ImportError:
        print(f"    opencv-python: Not installed")
    
    try:
        import pandas
        print(f"    pandas: {pandas.__version__}")
    except ImportError:
        print(f"    pandas: Not installed")
    
    try:
        import numpy
        print(f"    numpy: {numpy.__version__}")
    except ImportError:
        print(f"    numpy: Not installed")
    
    # Config 확인
    print(f"\n Configuration:")
    try:
        from config import Config
        print(f"   Data Root: {Config.DATA_ROOT}")
        print(f"   Video Dir exists: {Config.VIDEO_DIR.exists()}")
        print(f"   FileList exists: {Config.FILELIST_PATH.exists()}")
        print(f"   Batch Size: {Config.BATCH_SIZE}")
        print(f"   Num Workers: {Config.NUM_WORKERS}")
        print(f"   Device: {Config.DEVICE}")
    except Exception as e:
        print(f"     Config check failed: {e}")
    
    # 권장 사항
    print(f"\n Recommendations:")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 8:
            print(f"   - GPU Memory: {gpu_memory:.2f} GB (Small)")
            print(f"     Recommended BATCH_SIZE: 8-16")
        elif gpu_memory < 16:
            print(f"   - GPU Memory: {gpu_memory:.2f} GB (Medium)")
            print(f"     Recommended BATCH_SIZE: 16-32")
        else:
            print(f"   - GPU Memory: {gpu_memory:.2f} GB (Large)")
            print(f"     Recommended BATCH_SIZE: 32-64")
    else:
        print(f"   - Using CPU mode")
        print(f"     Recommended BATCH_SIZE: 4-8")
    
    print("\n" + "="*70)
    
    # 종료 코드
    if torch.cuda.is_available():
        print(" GPU environment ready!")
        return 0
    else:
        print("⚠️  GPU not available, but CPU mode is supported")
        return 1

if __name__ == "__main__":
    exit_code = check_gpu()
    sys.exit(exit_code)

