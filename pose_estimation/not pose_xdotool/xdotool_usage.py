# pose_estimation/xdotool_usage.py

import cv2
import numpy as np
import subprocess
import time
from pathlib import Path

from pose_estimation.pose_estimator import PoseEstimator

def _empty_callback(value):
    pass

class XdotoolPoseEstimator(PoseEstimator):
    """
    PoseEstimator를 상속받아 근접 감지 시 xdotool 명령을 실행하는 기능을 추가한 클래스.
    """
    def __init__(self, model_name="human-pose-estimation-0001", precision="FP16-INT8", device="AUTO",
                 proximity_threshold=0.20, 
                 trigger_command="xdotool key 'super+l'"):
        super().__init__(model_name=model_name, precision=precision, device=device)
        self.proximity_threshold = proximity_threshold
        self.trigger_command = trigger_command
        self.proximity_triggered = False

    def _check_proximity_and_trigger(self, poses, frame_shape):
        frame_height, frame_width, _ = frame_shape
        frame_area = frame_height * frame_width
        person_is_close = False
        largest_person_box = None
        max_area_ratio = 0

        if poses.size > 0:
            for pose in poses:
                valid_keypoints = pose[pose[:, 0] > 0]
                if valid_keypoints.shape[0] < 4: continue
                min_x, min_y = np.min(valid_keypoints[:, :2], axis=0)
                max_x, max_y = np.max(valid_keypoints[:, :2], axis=0)
                area_ratio = ((max_x - min_x) * (max_y - min_y)) / frame_area
                if area_ratio > max_area_ratio:
                    max_area_ratio = area_ratio
                    largest_person_box = (int(min_x), int(min_y), int(max_x), int(max_y))

        if max_area_ratio > self.proximity_threshold:
            person_is_close = True
            if not self.proximity_triggered:
                print(f"Proximity! Ratio: {max_area_ratio:.2f} > Thresh: {self.proximity_threshold:.2f}. Executing.")
                try:
                    subprocess.run(self.trigger_command, shell=True, check=True)
                    self.proximity_triggered = True
                except Exception as e:
                    print(f"Failed to execute trigger command: {e}")
        else:
            if self.proximity_triggered: print("Person moved away. Resetting trigger.")
            self.proximity_triggered = False
        return person_is_close, largest_person_box, max_area_ratio

    def estimate_pose_video(self, source=0, output_path=None, use_popup=False, 
                           skip_first_frames=0, max_frames=None):
        if self.compiled_model is None: self.initialize_model()
        cap = cv2.VideoCapture(str(source)) if isinstance(source, (str, Path)) else cv2.VideoCapture(source)
        if not cap.isOpened(): raise ValueError(f"Could not open video source: {source}")
        
        writer = None
        if use_popup:
            title = "Proximity Trigger - Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar("Threshold %", title, int(self.proximity_threshold * 100), 100, _empty_callback)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                if use_popup: self.proximity_threshold = cv2.getTrackbarPos("Threshold %", title) / 100.0
                
                input_img, _ = self.preprocess_image(frame, max_size=None)
                start_time = time.time()
                results = self.compiled_model([input_img])
                self.video_utils.update_processing_time(time.time() - start_time)
                
                pafs = results[self.pafs_output_key]
                heatmaps = results[self.heatmaps_output_key]
                poses, _ = self._postprocess(pafs, heatmaps, frame)
                
                is_close, box, area_ratio = self._check_proximity_and_trigger(poses, frame.shape)
                result_frame = self.visualizer.draw_poses(frame, poses, 0.1)
                
                if box:
                    color = (0, 0, 255) if is_close else (0, 255, 0)
                    cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(result_frame, f"Area: {area_ratio:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                status_text = f"Armed: {not self.proximity_triggered} (Thresh: {self.proximity_threshold:.2f})"
                status_color = (0, 255, 0) if not self.proximity_triggered else (0, 0, 255)
                cv2.putText(result_frame, status_text, (15, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, status_color, 2)
                
                avg_time_ms, fps = self.video_utils.get_avg_processing_time_ms(), self.video_utils.get_fps()
                result_frame = self.visualizer.add_performance_info(result_frame, avg_time_ms, fps)

                if use_popup:
                    cv2.imshow(title, result_frame)
                    if cv2.waitKey(1) & 0xFF == 27: break
        finally:
            cap.release()
            if use_popup: cv2.destroyAllWindows()


# --- 이 파일의 기능을 실행하는 함수 ---
def run_proximity_trigger_example():
    """
    웹캠 근접 감지 트리거 예제를 실행합니다.
    이 함수는 XdotoolPoseEstimator의 생성과 실행을 모두 캡슐화합니다.
    """
    print("=== 웹캠 근접 감지 트리거 예제 ===")
    print("사람이 카메라에 가까이 접근하면 화면 잠금(super+l) 명령이 실행됩니다.")
    print("이 기능을 사용하려면 'xdotool'이 설치되어 있어야 합니다. (e.g., sudo apt install xdotool)")
    
    # 세부 파라미터 설정은 이 함수 내에서 이루어집니다.
    estimator = XdotoolPoseEstimator(
        proximity_threshold=0.5,
        trigger_command="xdotool set_desktop 1"
    )
    
    try:
        estimator.estimate_pose_webcam(camera_id=0, use_popup=True)
    except Exception as e:
        print(f"오류 발생: {e}")