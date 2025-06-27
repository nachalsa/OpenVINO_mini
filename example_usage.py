# 사용 예제 파일: example_usage.py
"""
OpenVINO Pose Estimation 사용 예제

이 파일은 pose_estimation 패키지의 사용법을 보여줍니다.
"""
#from pose_estimation.xdotool_usage import XdotoolPoseEstimator
from pose_estimation import PoseEstimator, run_proximity_trigger_example
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
        poses, sores, result_frame = estimator.estimate_pose_single_image(
            image_path=image_path,
            output_path=output_path,
            show_result=True
        )
        print(f"감지된 포즈 수: {len(poses)}")
        print(f"포즈 점수: {scores}")
    except FileNotFoundError:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다. 예제 이미지를 폴더에 추가해주세요.")
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
            use_popup=True, # 팝업 창으로 결과 보기
            skip_first_frames=100,
            max_frames=500
        )
        print("비디오 처리 완료!")
    except FileNotFoundError:
        print(f"오류: '{video_path}' 파일을 찾을 수 없습니다. 예제 비디오를 폴더에 추가해주세요.")
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
    try:
        model_info = estimator.get_model_info()
        print(f"모델 정보: {model_info}")
        # 간단한 테스트를 위해 단일 이미지 추정 실행
        example_single_image()
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    # 예제 실행
    print("OpenVINO Pose Estimation 예제\n")
    
    # [수정] 사용할 예제 선택 메뉴에 옵션 추가
    example_choice = input(5
        "실행할 예제를 선택하세요:\n"
        "1. 단일 이미지\n"
        "2. 비디오 파일\n"
        "3. 웹캠 실시간\n"
        "4. 커스텀 설정 보기\n"
        "5. 웹캠 근접 감지 트리거\n" # << 새로운 옵션 추가
        "선택 (1-5): "
    )
    
    if example_choice == "1":
        example_single_image()
    elif example_choice == "2":
        example_video_file()
    elif example_choice == "3":
        example_webcam()
    elif example_choice == "4":
        example_custom_settings()
    # [수정] 새로운 예제 함수 호출을 위한 elif 블록 추가
    elif example_choice == "5":
        run_proximity_trigger_example()
    else:
        print("잘못된 선택입니다.")
        