"""
数据增强：保持坐标尺度一致性
"""
import numpy as np
import torch
from typing import Tuple


class CoordinateAwareAugmenter:
    """坐标感知的增强器"""

    def __init__(
            self,
            noise_scale: float = 0.01,  # 相对标准化后的尺度
            mask_ratio: float = 0.15,
            rotate_angle: float = 0.1
    ):
        self.noise_scale = noise_scale
        self.mask_ratio = mask_ratio
        self.rotate_angle = rotate_angle

    def __call__(
            self,
            coords: np.ndarray,  # 已标准化的坐标 [T, 2]
            valid_length: int
    ) -> np.ndarray:
        """应用增强（保持标准化后的尺度）"""
        coords = coords.copy()
        valid_coords = coords[:valid_length]

        # 1. 对 xy 通道添加相对噪声；速度通道加更小的噪声（保护绝对速度信息）
        if np.random.rand() > 0.5:
            noise = np.zeros_like(valid_coords)
            noise[:, :2] = np.random.normal(0, self.noise_scale, (len(valid_coords), 2))
            noise[:, 2:] = np.random.normal(0, self.noise_scale * 0.1, (len(valid_coords), valid_coords.shape[1] - 2))
            valid_coords = valid_coords + noise

        # 2. 随机掩码（用邻近点插值，而非0）
        if np.random.rand() > 0.5:
            num_mask = max(1, int(valid_length * self.mask_ratio))
            mask_indices = np.random.choice(valid_length, num_mask, replace=False)

            for idx in mask_indices:
                # 用前后点的平均值替代
                if idx > 0 and idx < valid_length - 1:
                    valid_coords[idx] = (valid_coords[idx - 1] + valid_coords[idx + 1]) / 2
                elif idx == 0 and valid_length > 1:
                    valid_coords[idx] = valid_coords[idx + 1]
                elif idx == valid_length - 1 and valid_length > 1:
                    valid_coords[idx] = valid_coords[idx - 1]

        # 3. 小角度旋转（仅对 xy 坐标，不影响速度通道）
        if np.random.rand() > 0.7:
            angle = np.random.uniform(-self.rotate_angle, self.rotate_angle)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            center_xy = valid_coords[:, :2].mean(axis=0)
            coords_rotated = valid_coords.copy()
            coords_rotated[:, :2] = (valid_coords[:, :2] - center_xy) @ rotation_matrix.T + center_xy
            valid_coords = coords_rotated

        coords[:valid_length] = valid_coords
        return coords
