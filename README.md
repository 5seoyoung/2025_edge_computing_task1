# Structured / Unstructured Pruning 비교 실험

**2025년 2학기 국민대학교 엣지컴퓨팅 강의 과제 #1**

EchoNet-Dynamic 데이터셋을 활용한 좌심실 박출률(EF) 회귀 모델에 Structured와 Unstructured Pruning을 적용하여 비교 및 분석한 프로젝트입니다.

## 과제 개요

- **과제명**: Structured / Unstructured pruning 비교하기
- **목표**: Structured / Unstructured pruning을 자유로운 task에 적용해보고 비교 및 분석하기
- **데이터셋**: EchoNet-Dynamic (심초음파 영상 데이터)
- **Task**: 좌심실 박출률(Ejection Fraction, EF) 회귀 예측

## 프로젝트 구조

```
project/
├── config.py                    # 하이퍼파라미터 설정
├── dataset.py                   # EchoNetVideoDataset 클래스
├── model.py                     # EFRegressionModel 구현
├── train.py                     # 학습 및 평가 함수
├── prune_utils.py               # Pruning 함수들 (Structured/Unstructured)
├── metrics.py                   # Sparsity, Latency, MAE 계산
├── main.py                      # 전체 실험 실행 스크립트
├── check_gpu.py                 # GPU 환경 확인 스크립트
├── run_gpu.sh                   # GPU 서버 실행 스크립트
├── create_sample_data.py        # 샘플 데이터 생성 스크립트
├── requirements.txt             # 패키지 의존성
└── README.md                    # 프로젝트 설명
```

## 설치 방법

1. 가상환경 생성 (권장):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

2. 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 경로 설정

`config.py`에서 데이터 경로를 설정하세요:

```python
DATA_ROOT = Path("/path/to/echonet_dynamic")
```

### 2. 실험 실행

```bash
# GPU 환경 확인 (선택사항)
python check_gpu.py

# 실험 실행
python main.py
```

실험은 다음 순서로 진행됩니다:
1. Baseline 모델 학습 (ResNet-18 기반)
2. Unstructured Pruning 실험 (비율: 0.5, 0.8, 0.9)
3. Structured Pruning 실험 (비율: 0.3, 0.5, 0.7)
4. 결과 저장 (JSON, CSV)

### 3. 결과 시각화

```bash
# 실험 결과 그래프 생성
python visualize_results.py

# 모델 아키텍처 다이어그램 생성
python visualize_model_architecture.py
```

생성된 그래프는 `figures/` 디렉토리에 저장됩니다.

## 주요 기능

### 1. 모델 아키텍처 (`model.py`)

- **Backbone**: ResNet-18 (ImageNet pretrained)
- **Temporal Aggregation**: Mean pooling over frames
- **Output**: EF value (0-100)

### 2. Pruning 방법 (`prune_utils.py`)

#### Unstructured Pruning
- **방법**: L1 magnitude-based global pruning
- **대상**: 모든 Conv2d 및 Linear 레이어
- **비율**: 0.5, 0.8, 0.9

#### Structured Pruning
- **방법**: L2 norm-based channel-wise pruning
- **대상**: Conv2d 레이어만
- **비율**: 0.3, 0.5, 0.7

### 3. 평가 지표 (`metrics.py`)

- **MAE**: Mean Absolute Error (예측 정확도)
- **Sparsity**: 0 weight 비율 (압축률)
- **Latency**: 영상 단위 추론 시간 (ms/video)
- **Parameters**: 모델 파라미터 수

## Methodology

### 4.1 Dataset

- **EchoNet-Dynamic**
  - 데이터셋 설명: 심초음파 영상 데이터셋
  - 전처리 과정: 프레임 샘플링 (32 frames), 리사이즈 (112×112)
  - Train/Val/Test 분할: 100/20/20 (샘플 데이터셋 기준)

### 4.2 Model Architecture

- **Backbone**: ResNet-18 (pretrained on ImageNet)
- **Temporal Aggregation**: Mean pooling over frames
- **Regression Head**: Linear layer (512 → 1)
- **Input**: (B, 32, 3, 112, 112)
- **Output**: EF value (0-100)

### 4.3 Pruning Methods

#### 4.3.1 Unstructured Pruning

- **Method**: L1 magnitude-based global pruning
- **Target**: All Conv2d and Linear layers
- **Ratios**: 0.5, 0.8, 0.9
- **Process**:
  1. Global L1 pruning
  2. Remove pruning mask (make permanent)
  3. Fine-tuning (2 epochs, LR=1e-5)

#### 4.3.2 Structured Pruning

- **Method**: L2 norm-based channel-wise pruning
- **Target**: Conv2d layers only
- **Ratios**: 0.3, 0.5, 0.7
- **Process**:
  1. Layer-wise L2 structured pruning
  2. Remove pruning mask
  3. Fine-tuning (2 epochs, LR=1e-5)

### 4.4 Training Details

- **Optimizer**: Adam
- **Learning Rate**: 1e-4 (baseline), 1e-5 (fine-tuning)
- **Batch Size**: 16
- **Epochs**: 50 (baseline), 2 (fine-tuning)
- **Loss**: MSE
- **Hardware**: GPU (8GB CUDA)

### 4.5 Evaluation Metrics

- **MAE**: Mean Absolute Error
- **Sparsity**: Ratio of zero weights
- **Latency**: Inference time per video (ms)
- **Parameters**: Total trainable parameters

## Experiments

### 5.1 Experimental Setup

- **Hardware**: CUDA GPU (8GB)
- **Software**: PyTorch 2.5.1, CUDA 12.1
- **Dataset**: Sample EchoNet-Dynamic
- **Random Seed**: 42 (reproducibility)

### 5.2 Baseline Results

**Table 1: Baseline Model Performance**

| Metric | Value |
|--------|-------|
| Parameters | 11,177,025 |
| Sparsity | 0.0000 |
| MAE | 31.33 |
| Latency (ms/video) | 2.61 |

### 5.3 Unstructured Pruning Results

**Table 2: Unstructured Pruning Results**

| Ratio | Parameters | Sparsity (Before) | Sparsity (After) | MAE | Latency | MAE Increase |
|-------|------------|-------------------|------------------|-----|---------|--------------|
| 0.50 | 11,177,025 | 0.4996 | 0.0085 | 36.80 | 2.65 | +17.4% |
| 0.80 | 11,177,025 | 0.7993 | 0.0039 | 41.86 | 2.58 | +33.6% |
| 0.90 | 11,177,025 | 0.8992 | 0.0010 | 49.96 | 2.72 | +59.4% |

**Key Observations:**
- Fine-tuning 후 sparsity가 크게 감소
- Pruning ratio가 높을수록 성능 저하 증가
- Latency 개선 미미

### 5.4 Structured Pruning Results

**Table 3: Structured Pruning Results**

| Ratio | Parameters | Sparsity (Before) | Sparsity (After) | MAE | Latency | MAE Increase |
|-------|------------|-------------------|------------------|-----|---------|--------------|
| 0.30 | 11,177,025 | 0.3003 | 0.1849 | 56.31 | 2.70 | +79.7% |
| 0.50 | 11,177,025 | 0.4995 | 0.3763 | 55.56 | 2.57 | +77.3% |
| 0.70 | 11,177,025 | 0.6988 | 0.6084 | 53.14 | 2.64 | +69.6% |

**Key Observations:**
- Structured pruning이 더 큰 성능 저하를 보임
- Sparsity는 어느 정도 유지되지만 MAE가 크게 증가
- Latency 개선 여전히 미미

## 실험 결과

실험 결과는 다음 위치에 저장됩니다:

- `checkpoints/`: 모델 체크포인트
- `results/`: 실험 결과 (JSON, CSV)
- `figures/`: 시각화 그래프 (PNG, PDF)

### 결과 항목

- Baseline 모델 성능
- 각 Pruning 비율별 성능 (MAE, Sparsity, Latency)
- Unstructured vs Structured Pruning 비교

## 설정 변경

`config.py`에서 다음을 수정할 수 있습니다:

```python
# 모델 파라미터
NUM_FRAMES = 32          # 비디오에서 샘플링할 프레임 수
IMAGE_SIZE = 112         # 이미지 크기
BATCH_SIZE = 16         # 배치 크기 (GPU: 16-32, CPU: 8)

# 학습 파라미터
NUM_EPOCHS = 50         # 학습 에포크 수
LEARNING_RATE = 1e-4    # 학습률

# Pruning 파라미터
UNSTRUCTURED_PRUNING_RATIOS = [0.5, 0.8, 0.9]
STRUCTURED_PRUNING_RATIOS = [0.3, 0.5, 0.7]
FINE_TUNE_EPOCHS = 2    # Pruning 후 fine-tuning 에포크 수
```

## 주의사항

1. **GPU 사용**: CUDA가 사용 가능한 경우 자동으로 GPU를 사용합니다.
2. **데이터 경로**: `config.py`의 `DATA_ROOT`를 실제 데이터 경로에 맞게 수정하세요.
3. **메모리**: 비디오 데이터는 메모리를 많이 사용할 수 있습니다. 배치 크기를 조정하거나 `NUM_FRAMES`를 줄이세요.

## 데이터셋

- **EchoNet-Dynamic**: 심초음파 영상 데이터셋
- **데이터 구조**:
  ```
  echonet_dynamic/
  ├── Videos/          # 비디오 파일들 (.avi)
  └── FileList.csv     # (FileName, EF) 컬럼 포함
  ```

## 참고사항

- 재현성을 위해 random seed가 42로 고정되어 있습니다.
- GPU 서버에서 실행 시 성능이 크게 향상됩니다 (10-50배).
- 샘플 데이터 생성 스크립트(`create_sample_data.py`)를 사용하여 테스트할 수 있습니다.

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.
