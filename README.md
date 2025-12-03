# EF Regression Model with Pruning Experiments

EchoNet-Dynamic 데이터셋을 활용한 좌심실 박출률(EF) 회귀 모델 및 Pruning 비교 실험 프로젝트입니다.

## 프로젝트 구조

```
project/
├── config.py          # 하이퍼파라미터 설정
├── dataset.py         # EchoNetVideoDataset 클래스
├── model.py           # EFRegressionModel 구현
├── train.py           # 학습 및 평가 함수
├── prune_utils.py     # Pruning 함수들
├── metrics.py         # Sparsity, Latency, MAE 계산
├── main.py            # 전체 실험 실행 스크립트
├── requirements.txt   # 패키지 의존성
└── README.md          # 프로젝트 설명
```

## 데이터 구조 및 경로 설정

### 데이터 구조

데이터는 다음과 같은 구조로 배치되어야 합니다:

```
echonet_dynamic/
├── Videos/
│   ├── video1.avi
│   ├── video2.avi
│   └── ...
├── FileList.csv       # (FileName, EF) 컬럼 포함
├── VolumeTracings.csv # (선택사항, 본 프로젝트에서는 사용하지 않음)
├── frames/            # (선택사항)
└── masks/             # (선택사항)
```

### 데이터 경로 설정 방법

**방법 1: 프로젝트 폴더 내에 데이터 배치 (권장하지 않음)**
- 프로젝트 폴더 내 `data/echonet_dynamic/` 디렉토리에 데이터를 복사
- `config.py`의 기본 설정 그대로 사용

**방법 2: 외부 경로 사용 (권장)**
- 데이터를 다른 위치에 두고 `config.py`에서 절대 경로로 설정:

```python
# config.py에서 수정
DATA_ROOT = Path("/path/to/your/echonet_dynamic")
# 예: DATA_ROOT = Path("/Users/ohseoyoung/Documents/echonet_dynamic")
```

**방법 3: 심볼릭 링크 사용 (Linux/Mac)**
```bash
# 프로젝트 폴더에서 실행
ln -s /path/to/your/echonet_dynamic data/echonet_dynamic
```

> **참고**: 데이터셋이 약 10,030개의 영상을 포함하고 있어 용량이 클 수 있습니다. 
> 프로젝트 폴더로 복사하지 않고 외부 경로를 사용하는 것을 권장합니다.

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

### 기본 실행

```bash
python main.py
```

이 스크립트는 다음을 자동으로 수행합니다:

1. **Baseline 모델 학습**: ResNet-18 기반 EF 회귀 모델 학습
2. **Unstructured Pruning 실험**: L1 기반 global pruning (비율: 0.5, 0.8, 0.9)
3. **Structured Pruning 실험**: Channel-wise pruning (비율: 0.3, 0.5, 0.7)
4. **결과 저장**: JSON 및 CSV 형식으로 결과 저장

### 설정 변경

`config.py` 파일에서 다음을 수정할 수 있습니다:

- `NUM_FRAMES`: 비디오에서 샘플링할 프레임 수 (기본값: 32)
- `IMAGE_SIZE`: 이미지 크기 (기본값: 112)
- `BATCH_SIZE`: 배치 크기 (기본값: 8)
- `NUM_EPOCHS`: 학습 에포크 수 (기본값: 50)
- `LEARNING_RATE`: 학습률 (기본값: 1e-4)
- `UNSTRUCTURED_PRUNING_RATIOS`: Unstructured pruning 비율 리스트
- `STRUCTURED_PRUNING_RATIOS`: Structured pruning 비율 리스트

## 주요 기능

### 1. Dataset Loader (`dataset.py`)

- OpenCV를 사용한 비디오 로드
- 균등 샘플링 (N개 프레임)
- 전처리 (Resize, Normalize)
- Train/Val/Test 분할

### 2. EF Regression Model (`model.py`)

- ResNet-18 백본 (pretrained)
- Temporal aggregation (mean pooling)
- Linear regression head

### 3. Pruning (`prune_utils.py`)

- **Unstructured Pruning**: L1 magnitude 기반 global pruning
- **Structured Pruning**: Channel-wise pruning (L2 norm 기반)
- Pruning 후 fine-tuning 지원

### 4. 평가 지표 (`metrics.py`)

- **MAE**: Mean Absolute Error
- **Sparsity**: 0 weight 비율
- **Parameters**: 모델 파라미터 수
- **Latency**: 영상 단위 추론 시간 (ms/video)

## 결과 출력

실험 결과는 다음 위치에 저장됩니다:

- `checkpoints/`: 모델 체크포인트
- `results/`: 실험 결과 (JSON, CSV)

결과에는 다음 정보가 포함됩니다:

- Baseline 모델 성능
- 각 Pruning 비율별 성능
- 파라미터 수, Sparsity, MAE, Latency 비교

## 주의사항

1. **GPU 사용**: CUDA가 사용 가능한 경우 자동으로 GPU를 사용합니다. CPU만 사용 가능한 경우 `config.py`에서 `DEVICE = "cpu"`로 설정하세요.

2. **데이터 경로**: `config.py`의 `DATA_ROOT`를 실제 데이터 경로에 맞게 수정하세요.

3. **메모리**: 비디오 데이터는 메모리를 많이 사용할 수 있습니다. 배치 크기를 조정하거나 `NUM_FRAMES`를 줄이세요.

## 참고사항

- 이 프로젝트는 연구 목적으로 작성되었습니다.
- 재현성을 위해 random seed를 고정하는 것을 권장합니다.
- Colab GPU 환경에서 실행 가능하도록 설계되었습니다.

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

