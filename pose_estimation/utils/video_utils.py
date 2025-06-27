# pose_estimation/utils/video_utils.py
import cv2
import collections
import time
from IPython import display


class VideoUtils:
    """비디오 처리 유틸리티"""
    
    def __init__(self):
        self.processing_times = collections.deque()
    
    def update_processing_time(self, processing_time):
        """처리 시간 업데이트"""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 200:
            self.processing_times.popleft()
    
    def get_fps(self):
        """FPS 계산"""
        if not self.processing_times:
            return 0
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0
    
    def get_avg_processing_time_ms(self):
        """평균 처리 시간(ms) 반환"""
        if not self.processing_times:
            return 0
        return (sum(self.processing_times) / len(self.processing_times)) * 1000
    
    @staticmethod
    def display_frame_notebook(frame, quality=90):
        """노트북에서 프레임 표시"""
        _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, quality])
        i = display.Image(data=encoded_img)
        display.clear_output(wait=True)
        display.display(i)
