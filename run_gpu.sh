#!/bin/bash
# GPU 서버 실행 스크립트

echo "=========================================="
echo "EF Regression Model - GPU Server Run"
echo "=========================================="

# GPU 확인
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# 의존성 확인
echo ""
echo "Checking dependencies..."
python3 -c "import torch; import torchvision; import cv2; import pandas; print('✅ All dependencies available')" || {
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
}

# 데이터 경로 확인
echo ""
echo "Checking data paths..."
python3 -c "from config import Config; print(f'Data Root: {Config.DATA_ROOT}'); print(f'Video Dir exists: {Config.VIDEO_DIR.exists()}'); print(f'FileList exists: {Config.FILELIST_PATH.exists()}')"

# 실행
echo ""
echo "Starting training..."
echo "=========================================="
python3 main.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="

