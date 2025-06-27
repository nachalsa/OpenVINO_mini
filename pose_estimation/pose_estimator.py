# pose_estimation/pose_estimator.py

import cv2
import time
import numpy as np
from pathlib import Path
from IPython import display
from numpy.lib.stride_tricks import as_strided

from .core.decoder import OpenPoseDecoder
from .core.visualizer import PoseVisualizer
from .models import ModelManager
from .utils.video_utils import VideoUtils


class PoseEstimator:
    """메인 포즈 추정 클래스"""
    
    def __init__(self, model_name="human-pose-estimation-0001", precision="FP16-INT8", device="AUTO"):
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
        
        # 비디오 소스 열기
        cap = cv2.VideoCapture(str(source)) if isinstance(source, (str, Path)) else cv2.VideoCapture(source)
        if not cap.isOpened(): 
            raise ValueError(f"Could not open video source: {source}")
        
        # 비디오 라이터 초기화
        writer = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            # 비디오 프레임 크기를 가져올 때, 원본 영상의 크기를 사용하도록 보장
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # 팝업 윈도우 설정
        if use_popup:
            title = "Pose Estimation - Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        try:
            # 초기 프레임 스킵
            for _ in range(skip_first_frames):
                ret, _ = cap.read()
                if not ret:
                    print("Could not skip frames, video source might be too short.")
                    break
            
            while True:
                # 프레임 읽기
                ret, frame = cap.read()
                if not ret:
                    print("Video stream ended or failed to read frame.")
                    break
                
                # 최대 프레임 제한
                if max_frames and frame_count >= max_frames:
                    print(f"Reached max frames limit: {max_frames}")
                    break
                
                # 원본 프레임 복사본으로 추론 및 시각화 진행
                display_frame = frame.copy()

                input_img, _ = self.preprocess_image(display_frame, max_size=None)
                
                start_time = time.time()
                results = self.compiled_model([input_img])
                self.video_utils.update_processing_time(time.time() - start_time)
                
                pafs = results[self.pafs_output_key]
                heatmaps = results[self.heatmaps_output_key]
                poses, _ = self._postprocess(pafs, heatmaps, display_frame)
                
                result_frame = self.visualizer.draw_poses(display_frame, poses, 0.1)
                avg_time_ms = self.video_utils.get_avg_processing_time_ms()
                fps = self.video_utils.get_fps()
                result_frame = self.visualizer.add_performance_info(result_frame, avg_time_ms, fps)
                
                if writer:
                    # 저장할 때는 원본 해상도와 맞는지 확인
                    if result_frame.shape[0] != h or result_frame.shape[1] != w:
                         # 만약 디스플레이 프레임 크기가 다르다면 원본 크기로 리사이즈
                         result_frame_to_write = cv2.resize(result_frame, (w, h))
                         writer.write(result_frame_to_write)
                    else:
                         writer.write(result_frame)

                if use_popup:
                    cv2.imshow(title, result_frame)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC 키
                        print("ESC key pressed. Exiting.")
                        break
                else:
                    self.video_utils.display_frame_notebook(result_frame)
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("Interrupted by user.")
        except Exception as e:
            # 파이썬 레벨에서 잡을 수 있는 다른 예외 처리
            print(f"An unexpected error occurred: {e}")
        finally:
            print("Releasing resources...")
            if cap.isOpened():
                cap.release()
            if writer:
                writer.release()
            if use_popup:
                cv2.destroyAllWindows()
            print(f"Processing finished. Total frames processed: {frame_count}.")

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