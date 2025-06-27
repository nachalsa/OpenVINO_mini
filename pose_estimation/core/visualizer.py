# pose_estimation/core/visualizer.py
import cv2
import numpy as np


class PoseVisualizer:
    """포즈 시각화 클래스"""
    
    def __init__(self):
        self.colors = (
            (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
            (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
            (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
            (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
            (0, 170, 255),
        )
        
        self.default_skeleton = (
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
            (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
            (3, 5), (4, 6),
        )

    def draw_poses(self, img, poses, point_score_threshold, skeleton=None):
        """포즈 그리기"""
        if poses.size == 0:
            return img

        if skeleton is None:
            skeleton = self.default_skeleton
            
        img_limbs = np.copy(img)
        
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            
            # 관절 그리기
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > point_score_threshold:
                    cv2.circle(img, tuple(p), 1, self.colors[i], 2)
                    
            # 연결선 그리기
            for i, j in skeleton:
                if (points_scores[i] > point_score_threshold and 
                    points_scores[j] > point_score_threshold):
                    cv2.line(
                        img_limbs,
                        tuple(points[i]),
                        tuple(points[j]),
                        color=self.colors[j],
                        thickness=4,
                    )
                    
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img

    def add_performance_info(self, frame, processing_time, fps):
        """성능 정보 추가"""
        _, f_width = frame.shape[:2]
        cv2.putText(
            frame,
            f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            (20, 40),
            cv2.FONT_HERSHEY_COMPLEX,
            f_width / 1000,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

