# 사용 예제 파일: example_usage.py
"""
OpenVINO Pose Estimation 사용 예제

이 파일은 pose_estimation 패키지의 사용법을 보여줍니다.
"""

from pose_estimation import PoseEstimator
from pathlib import Path

def example_single_image():
    """단일 이미지 포즈 추정 예제"""
    print("=== 단일 이미지 포즈 추정 예제 ===")
    
    # 포즈 추정기 초기화
    estimator = PoseEstimator()
    
    # 이미지에서 포즈 추정
    image_path = "sample_image.jpg"
    output_path = "result_image.jpg"
    
    try:
        poses, scores, result_frame = estimator.estimate_pose_single_image(
            image_path=image_path,
            output_path=output_path,
            show_result=True
        )
        print(f"감지된 포즈 수: {len(poses)}")
        print(f"포즈 점수: {scores}")
    except Exception as e:
        print(f"오류 발생: {e}")

def example_video_file():
    """비디오 파일 포즈 추정 예제"""
    print("=== 비디오 파일 포즈 추정 예제 ===")
    
    # 포즈 추정기 초기화
    estimator = PoseEstimator(device="CPU")  # CPU 사용
    
    # 비디오 파일에서 포즈 추정
    video_path = "sample_video.mp4"
    output_path = "result_video.mp4"
    
    try:
        estimator.estimate_pose_video(
            source=video_path,
            output_path=output_path,
            use_popup=False,
            skip_first_frames=100,
            max_frames=500
        )
        print("비디오 처리 완료!")
    except Exception as e:
        print(f"오류 발생: {e}")

def example_webcam():
    """웹캠 실시간 포즈 추정 예제"""
    print("=== 웹캠 실시간 포즈 추정 예제 ===")
    
    # 포즈 추정기 초기화
    estimator = PoseEstimator()
    
    # 디코더 파라미터 조정
    estimator.set_decoder_params(
        score_threshold=0.2,
        min_paf_alignment_score=0.1
    )
    
    try:
        estimator.estimate_pose_webcam(camera_id=0, use_popup=True)
    except Exception as e:
        print(f"오류 발생: {e}")

def example_custom_settings():
    """커스텀 설정 예제"""
    print("=== 커스텀 설정 예제 ===")
    
    # 고정밀도 모델 사용
    estimator = PoseEstimator(
        model_name="human-pose-estimation-0001",
        precision="FP32",
        device="GPU"  # GPU 사용 (사용 가능한 경우)
    )
    
    # 시각화 색상 커스터마이징
    custom_colors = [(255, 0, 0)] * 17  # 모든 포인트를 빨간색으로
    estimator.set_visualization_params(colors=custom_colors)
    
    # 모델 정보 출력
    model_info = estimator.get_model_info()
    print(f"모델 정보: {model_info}")

if __name__ == "__main__":
    # 예제 실행
    print("OpenVINO Pose Estimation 예제\n")
    
    # 사용할 예제 선택
    example_choice = input(
        "실행할 예제를 선택하세요:\n"
        "1. 단일 이미지\n"
        "2. 비디오 파일\n"
        "3. 웹캠 실시간\n"
        "4. 커스텀 설정\n"
        "선택 (1-4): "
    )
    
    if example_choice == "1":
        example_single_image()
    elif example_choice == "2":
        example_video_file()
    elif example_choice == "3":
        example_webcam()
    elif example_choice == "4":
        example_custom_settings()
    else:
        print("잘못된 선택입니다.")
