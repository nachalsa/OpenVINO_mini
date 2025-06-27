# 사용 예제 파일: example_usage.py
"""
OpenVINO Pose Estimation 사용 예제

이 파일은 pose_estimation 패키지의 사용법을 보여줍니다.
"""
#from pose_estimation.xdotool_usage import XdotoolPoseEstimator
from pose_estimation import PoseEstimator, run_proximity_trigger_example
from pathlib import Path


if __name__ == "__main__":
    # [수정] 메뉴를 보여주거나 입력을 받는 대신, 바로 함수를 호출합니다.
    
    # 예제 실행
    print("OpenVINO Pose Estimation - 웹캠 근접 감지 트리거를 시작합니다.\n")
    
    # 확장 기능 예제를 직접 실행
    run_proximity_trigger_example()

    print("\n프로그램이 종료되었습니다.")