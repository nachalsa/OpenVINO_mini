# pose_estimation/core/decoder.py

import numpy as np

class OpenPoseDecoder:
    """
    OpenPose 디코더 클래스.
    모델 출력(heatmaps, pafs)을 인간 포즈 키포인트 데이터로 변환합니다.
    NMS나 좌표 스케일링 같은 전/후처리는 이 클래스 외부에서 처리됩니다.
    """
    
    BODY_PARTS_KPT_IDS = (
        (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
        (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
        (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17),
    )
    
    BODY_PARTS_PAF_IDS = (
        12, 20, 14, 16, 22, 24, 0, 2, 4, 6, 8, 10, 28, 30, 34, 32, 36, 18, 26,
    )

    def __init__(self, num_joints=18, skeleton=None, paf_indices=None,
                 max_points=100, score_threshold=0.1, 
                 min_paf_alignment_score=0.05, delta=0.5):
        """
        Args:
            num_joints: 관절 수
            skeleton: 스켈레톤 연결 정보
            paf_indices: PAF 인덱스
            max_points: 최대 포인트 수
            score_threshold: 점수 임계값
            min_paf_alignment_score: 최소 PAF 정렬 점수
            delta: 오프셋 값
        """
        self.num_joints = num_joints
        self.skeleton = skeleton or self.BODY_PARTS_KPT_IDS
        self.paf_indices = paf_indices or self.BODY_PARTS_PAF_IDS
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.min_paf_alignment_score = min_paf_alignment_score
        self.delta = delta
        
        self.points_per_limb = 10
        self.grid = np.arange(self.points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    def __call__(self, heatmaps, nms_heatmaps, pafs):
        """히트맵과 PAF를 포즈로 디코딩"""
        batch_size, _, h, w = heatmaps.shape
        assert batch_size == 1, "Batch size of 1 only supported"

        keypoints = self.extract_points(heatmaps, nms_heatmaps)
        pafs_t = np.transpose(pafs, (0, 2, 3, 1))

        if self.delta > 0:
            for kpts in keypoints:
                if kpts.size > 0:
                    kpts[:, :2] += self.delta
                    np.clip(kpts[:, 0], 0, w - 1, out=kpts[:, 0])
                    np.clip(kpts[:, 1], 0, h - 1, out=kpts[:, 1])

        pose_entries, all_keypoints = self.group_keypoints(
            keypoints, pafs_t, pose_entry_size=self.num_joints + 2
        )
        poses, scores = self.convert_to_coco_format(pose_entries, all_keypoints)
        
        if len(poses) > 0:
            poses = np.asarray(poses, dtype=np.float32)
            poses = poses.reshape((poses.shape[0], -1, 3))
        else:
            poses = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)

        return poses, scores

    def extract_points(self, heatmaps, nms_heatmaps):
        """히트맵에서 키포인트 추출"""
        batch_size, channels_num, h, w = heatmaps.shape
        assert batch_size == 1, "Batch size of 1 only supported"
        assert channels_num >= self.num_joints

        xs, ys, scores = self.top_k(nms_heatmaps)
        masks = scores > self.score_threshold
        all_keypoints = []
        keypoint_id = 0
        
        for k in range(self.num_joints):
            mask = masks[0, k]
            x = xs[0, k][mask].ravel()
            y = ys[0, k][mask].ravel()
            score = scores[0, k][mask].ravel()
            n = len(x)
            
            if n == 0:
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                continue
                
            x, y = self.refine(heatmaps[0, k], x, y)
            np.clip(x, 0, w - 1, out=x)
            np.clip(y, 0, h - 1, out=y)
            
            keypoints = np.empty((n, 4), dtype=np.float32)
            keypoints[:, 0] = x
            keypoints[:, 1] = y
            keypoints[:, 2] = score
            keypoints[:, 3] = np.arange(keypoint_id, keypoint_id + n)
            keypoint_id += n
            all_keypoints.append(keypoints)
            
        return all_keypoints

    def top_k(self, heatmaps):
        """상위 k개 포인트 추출"""
        N, K, _, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        ind = heatmaps.argpartition(-self.max_points, axis=2)[:, :, -self.max_points:]
        scores = np.take_along_axis(heatmaps, ind, axis=2)
        subind = np.argsort(-scores, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        scores = np.take_along_axis(scores, subind, axis=2)
        y, x = np.divmod(ind, W)
        return x, y, scores

    @staticmethod
    def refine(heatmap, x, y):
        """키포인트 위치 정제"""
        h, w = heatmap.shape[-2:]
        valid = np.logical_and(
            np.logical_and(x > 0, x < w - 1),
            np.logical_and(y > 0, y < h - 1)
        )
        xx = x[valid]
        yy = y[valid]
        dx = np.sign(heatmap[yy, xx + 1] - heatmap[yy, xx - 1], dtype=np.float32) * 0.25
        dy = np.sign(heatmap[yy + 1, xx] - heatmap[yy - 1, xx], dtype=np.float32) * 0.25
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x[valid] += dx
        y[valid] += dy
        return x, y

    @staticmethod
    def is_disjoint(pose_a, pose_b):
        """두 포즈가 분리되어 있는지 확인"""
        pose_a = pose_a[:-2]
        pose_b = pose_b[:-2]
        return np.all(np.logical_or.reduce((pose_a == pose_b, pose_a < 0, pose_b < 0)))

    def update_poses(self, kpt_a_id, kpt_b_id, all_keypoints, connections, pose_entries, pose_entry_size):
        """포즈 업데이트"""
        for connection in connections:
            pose_a_idx, pose_b_idx = -1, -1
            for j, pose in enumerate(pose_entries):
                if pose[kpt_a_id] == connection[0]:
                    pose_a_idx = j
                if pose[kpt_b_id] == connection[1]:
                    pose_b_idx = j
                    
            if pose_a_idx < 0 and pose_b_idx < 0:
                pose_entry = np.full(pose_entry_size, -1, dtype=np.float32)
                pose_entry[kpt_a_id] = connection[0]
                pose_entry[kpt_b_id] = connection[1]
                pose_entry[-1] = 2
                pose_entry[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
                pose_entries.append(pose_entry)
            elif pose_a_idx >= 0 and pose_b_idx >= 0 and pose_a_idx != pose_b_idx:
                pose_a, pose_b = pose_entries[pose_a_idx], pose_entries[pose_b_idx]
                if self.is_disjoint(pose_a, pose_b):
                    pose_a += pose_b
                    pose_a[:-2] += 1
                    pose_a[-2] += connection[2]
                    del pose_entries[pose_b_idx]
            elif pose_a_idx >= 0 and pose_b_idx >= 0:
                pose_entries[pose_a_idx][-2] += connection[2]
            elif pose_a_idx >= 0:
                pose = pose_entries[pose_a_idx]
                if pose[kpt_b_id] < 0:
                    pose[-2] += all_keypoints[connection[1], 2]
                pose[kpt_b_id] = connection[1]
                pose[-2] += connection[2]
                pose[-1] += 1
            elif pose_b_idx >= 0:
                pose = pose_entries[pose_b_idx]
                if pose[kpt_a_id] < 0:
                    pose[-2] += all_keypoints[connection[0], 2]
                pose[kpt_a_id] = connection[0]
                pose[-2] += connection[2]
                pose[-1] += 1
        return pose_entries

    @staticmethod
    def connections_nms(a_idx, b_idx, affinity_scores):
        """연결 NMS"""
        order = affinity_scores.argsort()[::-1]
        a_idx, b_idx, affinity_scores = a_idx[order], b_idx[order], affinity_scores[order]
        idx = []
        has_kpt_a, has_kpt_b = set(), set()
        for t, (i, j) in enumerate(zip(a_idx, b_idx)):
            if i not in has_kpt_a and j not in has_kpt_b:
                idx.append(t)
                has_kpt_a.add(i)
                has_kpt_b.add(j)
        idx = np.asarray(idx, dtype=np.int32)
        return a_idx[idx], b_idx[idx], affinity_scores[idx]

    def group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20):
        """키포인트 그룹화"""
        all_keypoints = np.concatenate(all_keypoints_by_type, axis=0)
        pose_entries = []
        
        for part_id, paf_channel in enumerate(self.paf_indices):
            kpt_a_id, kpt_b_id = self.skeleton[part_id]
            kpts_a, kpts_b = all_keypoints_by_type[kpt_a_id], all_keypoints_by_type[kpt_b_id]
            n, m = len(kpts_a), len(kpts_b)
            if n == 0 or m == 0: continue

            a = kpts_a[:, :2]
            a = np.broadcast_to(a[None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)
            steps = 1 / (self.points_per_limb - 1) * vec_raw
            points = (steps * self.grid + a.reshape(-1, 1, 2)).round().astype(np.int32)
            x, y = points[..., 0].ravel(), points[..., 1].ravel()

            part_pafs = pafs[0, :, :, paf_channel:paf_channel + 2]
            field = part_pafs[y, x].reshape(-1, self.points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field * vec).sum(-1).reshape(-1, self.points_per_limb)
            valid_affinity_scores = affinity_scores > self.min_paf_alignment_score
            valid_num = valid_affinity_scores.sum(1)
            affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
            success_ratio = valid_num / self.points_per_limb

            valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            if len(valid_limbs) == 0: continue
            
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity_scores = affinity_scores[valid_limbs]
            a_idx, b_idx, affinity_scores = self.connections_nms(a_idx, b_idx, affinity_scores)
            connections = list(zip(kpts_a[a_idx, 3].astype(np.int32), kpts_b[b_idx, 3].astype(np.int32), affinity_scores))
            if len(connections) == 0: continue

            pose_entries = self.update_poses(kpt_a_id, kpt_b_id, all_keypoints, connections, pose_entries, pose_entry_size)

        pose_entries = np.asarray(pose_entries, dtype=np.float32).reshape(-1, pose_entry_size)
        valid_poses = pose_entries[:, -1] >= 3
        return pose_entries[valid_poses], all_keypoints

    @staticmethod
    def convert_to_coco_format(pose_entries, all_keypoints):
        """COCO 형식으로 변환"""
        coco_keypoints, scores = [], []
        num_joints = 17
        for pose in pose_entries:
            if len(pose) == 0: continue
            
            keypoints = np.zeros(num_joints * 3)
            reorder_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            person_score = pose[-2]
            
            for keypoint_id_int, target_id in zip(pose[:-2].astype(int), reorder_map):
                if target_id < 0 or keypoint_id_int < 0 or keypoint_id_int >= len(all_keypoints):
                    continue
                cx, cy, score = all_keypoints[keypoint_id_int, 0:3]
                keypoints[target_id * 3 + 0] = cx
                keypoints[target_id * 3 + 1] = cy
                keypoints[target_id * 3 + 2] = score
                
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose[-1] - 1)))
            
        return np.asarray(coco_keypoints), np.asarray(scores)