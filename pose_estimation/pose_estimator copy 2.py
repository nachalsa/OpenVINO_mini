# pose_estimation/pose_estimator.py

import cv2
import time
import numpy as np
import subprocess
from pathlib import Path
from IPython import display
from numpy.lib.stride_tricks import as_strided

from .core.decoder import OpenPoseDecoder
from .core.visualizer import PoseVisualizer
from .models import ModelManager
from .utils.video_utils import VideoUtils

def _empty_callback(value):
    """트랙바를 위한 빈 콜백 함수. 아무 작업도 하지 않습니다."""
    pass

class PoseEstimator:
    """메인 포즈 추정 클래스"""
    
    def __init__(self, model_name="human-pose-estimation-0001", precision="FP16-INT8", device="AUTO",
                 enable_proximity_trigger=False, 
                 proximity_threshold=0.15, 
                 trigger_command="xdotool key 'super+l'"):
        self.model_manager = ModelManager()
        self.decoder = OpenPoseDecoder()
        self.visualizer = PoseVisualizer()
        self.video_utils = VideoUtils()
        self.model_name = model_name
        self.precision = precision
        self.device = device
        self.compiled_model = None
        self.model_info = None
        self.pafs_output_key = None
        self.heatmaps_output_key = None

        # [추가] 이 부분도 __init__ 메서드 안에 포함되어야 합니다.
        self.enable_proximity_trigger = enable_proximity_trigger
        self.proximity_threshold = proximity_threshold
        self.trigger_command = trigger_command
        self.proximity_triggered = False
        
    def initialize_model(self):
        """모델 다운로드, 로드 및 초기화"""
        model_path = self.model_manager.download_model(self.model_name, self.precision)
        self.compiled_model = self.model_manager.load_model(model_path, self.device)
        self.model_info = self.model_manager.get_input_output_info()
        self.pafs_output_key = self.compiled_model.output("Mconv7_stage2_L1")
        self.heatmaps_output_key = self.compiled_model.output("Mconv7_stage2_L2")
        print("Model initialized successfully!")

    def preprocess_image(self, frame, max_size=1280):
        """이미지 전처리 (리사이즈 및 정규화)"""
        scale = max_size / max(frame.shape) if max_size else 1.0
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        input_img = cv2.resize(frame, (self.model_info['width'], self.model_info['height']), interpolation=cv2.INTER_AREA)
        input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]
        
        return input_img, frame
    
    @staticmethod
    def _pool2d(A, kernel_size, stride, padding, pool_mode="max"):
        """2D 풀링 연산 (decoder에서 이동)"""
        A = np.pad(A, padding, mode="constant")
        output_shape = ((A.shape[0] - kernel_size) // stride + 1, (A.shape[1] - kernel_size) // stride + 1)
        A_w = as_strided(A, shape=output_shape + (kernel_size, kernel_size), strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides)
        A_w = A_w.reshape(-1, kernel_size, kernel_size)
        if pool_mode == "max": return A_w.max(axis=(1, 2)).reshape(output_shape)
        elif pool_mode == "avg": return A_w.mean(axis=(1, 2)).reshape(output_shape)

    def _perform_heatmap_nms(self, heatmaps):
        """히트맵에 NMS 적용 (decoder에서 이동)"""
        pooled_heatmaps = np.array([
            [self._pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]
        ])
        return heatmaps * (heatmaps == pooled_heatmaps)

    def _postprocess(self, pafs, heatmaps, original_frame):
        """
        추론 결과를 후처리하여 최종 포즈를 얻습니다.
        NMS, 디코딩, 좌표 스케일링을 포함합니다.
        """
        # 1. Heatmap NMS
        nms_heatmaps = self._perform_heatmap_nms(heatmaps)

        # 2. Decode poses
        poses, scores = self.decoder(heatmaps, nms_heatmaps, pafs)

        # 3. 좌표를 원본 이미지 크기에 맞게 스케일링
        output_shape = self.model_info['output_layers'][0].partial_shape
        output_scale = (
            original_frame.shape[1] / output_shape[3].get_length(),
            original_frame.shape[0] / output_shape[2].get_length(),
        )
        if poses.size > 0:
            poses[:, :, :2] *= output_scale
        
        return poses, scores

    def estimate_pose_single_image(self, image_path, output_path=None, show_result=True):
        """단일 이미지에서 포즈 추정"""
        if self.compiled_model is None: self.initialize_model()
        
        frame = cv2.imread(str(image_path))
        if frame is None: raise ValueError(f"Could not load image: {image_path}")
        
        input_img, processed_frame = self.preprocess_image(frame)
        
        start_time = time.time()
        results = self.compiled_model([input_img])
        processing_time = (time.time() - start_time) * 1000
        
        pafs = results[self.pafs_output_key]
        heatmaps = results[self.heatmaps_output_key]
        poses, scores = self._postprocess(pafs, heatmaps, processed_frame)
        
        result_frame = self.visualizer.draw_poses(processed_frame, poses, 0.1)
        result_frame = self.visualizer.add_performance_info(result_frame, processing_time, 1000 / processing_time if processing_time > 0 else 0)
        
        if output_path:
            cv2.imwrite(str(output_path), result_frame)
            print(f"Result saved to: {output_path}")
        
        if show_result:
            cv2.imshow("Pose Estimation Result", result_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return poses, scores, result_frame
    
    def estimate_pose_video(self, source=0, output_path=None, use_popup=False, 
                           skip_first_frames=0, max_frames=None):
        """비디오/웹캠에서 포즈 추정"""
        if self.compiled_model is None: self.initialize_model()
        
        cap = cv2.VideoCapture(str(source)) if isinstance(source, (str, Path)) else cv2.VideoCapture(source)
        if not cap.isOpened(): raise ValueError(f"Could not open video source: {source}")
        
        writer = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        if use_popup:
            title = "Pose Estimation - Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
            # [트랙바 수정] 근접 감지 기능이 활성화되었을 때 트랙바 생성
            if self.enable_proximity_trigger:
                # 트랙바는 정수만 다루므로, 0~100 사이의 값으로 만든 후 100으로 나눔
                initial_threshold_int = int(self.proximity_threshold * 100)
                cv2.createTrackbar("Threshold %", title, initial_threshold_int, 100, _empty_callback)
        
        
        frame_count = 0
        try:
            for _ in range(skip_first_frames):
                if not cap.read()[0]: break
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                if max_frames and frame_count >= max_frames: break
                 # [트랙바 수정] 루프마다 트랙바의 현재 값을 읽어와 threshold 업데이트
                if self.enable_proximity_trigger and use_popup:
                    current_threshold_int = cv2.getTrackbarPos("Threshold %", title)
                    self.proximity_threshold = current_threshold_int / 100.0
            
                
                
                input_img, _ = self.preprocess_image(frame, max_size=None)
                
                start_time = time.time()
                results = self.compiled_model([input_img])
                self.video_utils.update_processing_time(time.time() - start_time)
                
                pafs = results[self.pafs_output_key]
                heatmaps = results[self.heatmaps_output_key]
                poses, _ = self._postprocess(pafs, heatmaps, frame)
                
                result_frame = self.visualizer.draw_poses(frame, poses, 0.1)
                avg_time_ms = self.video_utils.get_avg_processing_time_ms()
                fps = self.video_utils.get_fps()
                result_frame = self.visualizer.add_performance_info(result_frame, avg_time_ms, fps)
                
                if self.enable_proximity_trigger:
                    # _check_proximity_and_trigger는 이제 실시간으로 변경된 self.proximity_threshold 값을 사용
                    is_close, box, area_ratio = self._check_proximity_and_trigger(poses, frame.shape)
                    
                    if box:
                        color = (0, 0, 255) if is_close else (0, 255, 0)
                        cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                        # [트랙바 수정] 현재 면적 비율을 화면에 표시
                        ratio_text = f"Area: {area_ratio:.2f}"
                        cv2.putText(result_frame, ratio_text, (box[0], box[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # [트랙바 수정] 현재 threshold 값과 트리거 상태를 화면에 표시
                    status_text = f"Armed: {not self.proximity_triggered} (Thresh: {self.proximity_threshold:.2f})"
                    status_color = (0, 255, 0) if not self.proximity_triggered else (0, 0, 255)
                    cv2.putText(result_frame, status_text, (15, 120), cv2.FONT_HERSHEY_COMPLEX, 0.8, status_color, 2)
                
                
                if writer: writer.write(result_frame)
                
                if use_popup:
                    cv2.imshow(title, result_frame)
                    if cv2.waitKey(1) & 0xFF == 27: break
                else:
                    self.video_utils.display_frame_notebook(result_frame)
                
                frame_count += 1
                
        except KeyboardInterrupt: print("Interrupted by user")
        finally:
            cap.release()
            if writer: writer.release()
            if use_popup: cv2.destroyAllWindows()
            print(f"Processed {frame_count} frames.")
            
    # [트랙바 수정] _check_proximity_and_trigger 메서드가 현재 면적 비율(area_ratio)도 반환하도록 수정
    def _check_proximity_and_trigger(self, poses, frame_shape):
        """
        ... (기존 설명) ...
        반환값에 현재 감지된 최대 면적 비율을 추가합니다.
        """
        frame_height, frame_width, _ = frame_shape
        frame_area = frame_height * frame_width
        
        person_is_close = False
        largest_person_box = None
        max_area_ratio = 0

        if poses.size > 0:
            for pose in poses:
                valid_keypoints = pose[pose[:, 0] > 0]
                if valid_keypoints.shape[0] < 4:
                    continue
                
                min_x, min_y = np.min(valid_keypoints[:, :2], axis=0)
                max_x, max_y = np.max(valid_keypoints[:, :2], axis=0)
                
                box_area = (max_x - min_x) * (max_y - min_y)
                area_ratio = box_area / frame_area

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
        
        if not person_is_close:
            if self.proximity_triggered:
                print("Person moved away. Resetting trigger.")
            self.proximity_triggered = False
            
        # 반환값에 max_area_ratio 추가
        return person_is_close, largest_person_box, max_area_ratio
    
    
    def estimate_pose_webcam(self, camera_id=0, use_popup=True):
        """웹캠에서 실시간 포즈 추정"""
        self.estimate_pose_video(source=camera_id, use_popup=use_popup)
    
    def get_model_info(self):
        """모델 정보 반환"""
        if self.model_info is None: self.initialize_model()
        return self.model_info
    
    def set_decoder_params(self, **kwargs):
        """디코더 파라미터 설정"""
        self.decoder = OpenPoseDecoder(**kwargs)
    
    def set_visualization_params(self, colors=None, skeleton=None):
        """시각화 파라미터 설정"""
        if colors: self.visualizer.colors = colors
        if skeleton: self.visualizer.default_skeleton = skeleton