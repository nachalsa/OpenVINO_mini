# README.md 내용
"""
# OpenVINO Pose Estimation Package

OpenVINO를 사용한 인간 포즈 추정 패키지입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용
```python
from pose_estimation import PoseEstimator

# 포즈 추정기 초기화
estimator = PoseEstimator()

# 단일 이미지에서 포즈 추정
poses, scores, result = estimator.estimate_pose_single_image("image.jpg")

# 웹캠에서 실시간 포즈 추정
estimator.estimate_pose_webcam()
```

### 고급 사용
```python
# 커스텀 설정으로 초기화
estimator = PoseEstimator(
    model_name="human-pose-estimation-0001",
    precision="FP16-INT8",
    device="AUTO"
)

# 디코더 파라미터 조정
estimator.set_decoder_params(
    score_threshold=0.2,
    min_paf_alignment_score=0.1
)

# 시각화 설정
estimator.set_visualization_params(colors=custom_colors)
```

## 폴더 구조

```
pose_estimation/
├── __init__.py                 # 패키지 초기화
├── pose_estimator.py          # 메인 클래스
├── core/                      # 핵심 기능
│   ├── __init__.py
│   ├── decoder.py            # OpenPose 디코더 (후처리 포함)
│   └── visualizer.py         # 시각화
├── models/                   # 모델 관리
│   ├── __init__.py
│   └── model_manager.py      # 모델 다운로드/로드
└── utils/                    # 유틸리티
    ├── __init__.py
    └── video_utils.py        # 비디오 처리
```

## 예제

`example_usage.py` 파일을 참조하세요.

## 라이센스

MIT License
"""