"""
数据增强模块
提供多种轨迹数据增强方法
"""

import numpy as np
from typing import Callable, List


class TrajectoryAugmenter:
    """基础轨迹增强器"""

    def __init__(self,
                 noise_scale: float = 0.0001,
                 mask_ratio: float = 0.15,
                 rotate_angle: float = 0.1):
        """
        初始化增强器

        Args:
            noise_scale: 噪声标准差
            mask_ratio: 掩码比例
            rotate_angle: 旋转角度范围（弧度）
        """
        self.noise_scale = noise_scale
        self.mask_ratio = mask_ratio
        self.rotate_angle = rotate_angle

    def __call__(self, coords: np.ndarray, valid_length: int) -> np.ndarray:
        """
        应用增强

        Args:
            coords: 坐标数组 [max_len, 2]
            valid_length: 有效长度

        Returns:
            增强后的坐标
        """
        coords = coords.copy()
        valid_coords = coords[:valid_length]

        # 随机应用增强
        if np.random.rand() > 0.5:
            valid_coords = self._add_noise(valid_coords)
        if np.random.rand() > 0.5:
            valid_coords = self._random_mask(valid_coords)
        if np.random.rand() > 0.7:
            valid_coords = self._random_rotation(valid_coords)

        coords[:valid_length] = valid_coords
        return coords

    def _add_noise(self, coords: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_scale, coords.shape)
        return coords + noise

    def _random_mask(self, coords: np.ndarray) -> np.ndarray:
        """随机掩码（邻居插值替代 0）"""
        n = len(coords)
        num_mask = max(1, int(n * self.mask_ratio))
        idx = np.random.choice(n, size=num_mask, replace=False)
        out = coords.copy()
        if n > 2:
            prev = np.vstack([coords[0], coords[:-1]])
            nxt = np.vstack([coords[1:], coords[-1]])
            rep = 0.5 * (prev + nxt)
        else:
            rep = coords
        noise = np.random.normal(0, self.noise_scale * 0.5, size=out.shape)
        out[idx] = rep[idx] + noise[idx]
        return out

    def _random_rotation(self, coords: np.ndarray) -> np.ndarray:
        """随机旋转（仅对前两维 lat/lon 旋转）"""
        center = coords.mean(axis=0)
        coords_centered = coords - center
        angle = np.random.uniform(-self.rotate_angle, self.rotate_angle)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        result = coords_centered.copy()
        result[:, :2] = coords_centered[:, :2] @ rotation_matrix.T
        return result + center


class MultiScaleAugmenter(TrajectoryAugmenter):
    """多尺度增强器 - 组合多种增强方法"""

    def __init__(self):
        super().__init__()
        self.augmentation_pool: List[Callable] = [
            self._temporal_subsample,
            self._spatial_distortion,
            self._rotation_scaling,
            self._speed_perturbation
        ]

    def __call__(self, coords: np.ndarray, length: int) -> np.ndarray:
        """随机组合2-3种增强方法"""
        num_augs = np.random.randint(2, 4)
        selected_augs = np.random.choice(
            self.augmentation_pool,
            num_augs,
            replace=False
        )

        augmented = coords.copy()
        for aug_func in selected_augs:
            augmented = aug_func(augmented, length)

        return augmented

    def _temporal_subsample(self, coords: np.ndarray, length: int) -> np.ndarray:
        """时间子采样"""
        if length < 10:
            return coords

        sample_ratio = np.random.uniform(0.5, 0.9)
        num_samples = max(5, int(length * sample_ratio))
        indices = np.sort(np.random.choice(length, num_samples, replace=False))

        sampled = coords.copy()
        sampled[:num_samples] = coords[indices]
        sampled[num_samples:length] = 0

        return sampled

    def _spatial_distortion(self, coords: np.ndarray, length: int) -> np.ndarray:
        """空间扭曲"""
        distorted = coords.copy()
        center_idx = np.random.randint(0, length)
        radius = np.random.randint(5, min(20, length // 2))

        for i in range(max(0, center_idx - radius), min(length, center_idx + radius)):
            distance = abs(i - center_idx)
            distortion_strength = (1 - distance / radius) * 0.001
            distorted[i] += np.random.normal(0, distortion_strength, coords.shape[1])

        return distorted

    def _rotation_scaling(self, coords: np.ndarray, length: int) -> np.ndarray:
        """旋转和缩放"""
        center = coords[:length].mean(axis=0)
        coords_centered = coords.copy()
        coords_centered[:length] -= center

        angle = np.random.uniform(-0.2, 0.2)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        scale = np.random.uniform(0.9, 1.1)
        coords_centered[:length, :2] = (coords_centered[:length, :2] @ rotation.T) * scale
        coords_centered[:length] += center

        return coords_centered

    def _speed_perturbation(self, coords: np.ndarray, length: int) -> np.ndarray:
        """速度扰动"""
        if length < 3:
            return coords

        perturbed = coords.copy()
        velocities = np.diff(coords[:length], axis=0)
        speed_factors = np.random.uniform(0.8, 1.2, len(velocities))
        velocities_perturbed = velocities * speed_factors[:, np.newaxis]

        perturbed[0] = coords[0]
        for i in range(len(velocities_perturbed)):
            perturbed[i + 1] = perturbed[i] + velocities_perturbed[i]

        return perturbed
