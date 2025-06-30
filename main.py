# 사용 예제 파일: main.py
"""
OpenVINO Pose Estimation 사용 예제

이 파일은 dark_templar패키지의 사용법을 보여줍니다.
"""

from dark_templar import run_dark_templar


if __name__ == "__main__":
    
    # 예제 실행
    print("OpenVINO Dark Templar - 웹캠 근접 감지 트리거를 시작합니다.\n")
    
    # 확장 기능 예제를 직접 실행
    run_dark_templar(0.4)

    print("\n Adun Toridas ")