# data/dataset.py
"""
PyTorch Dataset和DataLoader工具
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional
from .augmentation import MultiScaleAugmenter
from .augmentation_v2 import CoordinateAwareAugmenter  # 新增导入

class TrajDataset(Dataset):
    """轨迹数据集"""

    def __init__(self,
                 trajectories: List[Dict],
                 augment: bool = True,
                 augmenter: Optional[MultiScaleAugmenter] = None):  # ✅ 添加此参数
        """
        初始化数据集

        Args:
            trajectories: 轨迹列表
            augment: 是否应用数据增强
            augmenter: 数据增强器实例（可选，如果未提供则自动创建）
        """
        self.trajs = trajectories
        self.augment = augment

        # ✅ 支持传入 augmenter 或自动创建
        if augment:
            self.augmenter = augmenter if augmenter is not None else CoordinateAwareAugmenter()
        else:
            self.augmenter = None

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本

        Args:
            idx: 索引

        Returns:
            样本字典
        """
        traj = self.trajs[idx]
        coords = traj['raw_coords'].copy()
        length = traj['original_length']
        label = traj['label'] if traj['label'] is not None else -1

        # 应用增强
        if self.augment and self.augmenter:
            coords = self.augmenter(coords, length)

        return {
            'coords': coords,
            'features': traj['features'],
            'length': length,
            'label': label,
            'index': idx
        }

    def update_labels(self, new_labels: np.ndarray):
        """
        更新数据集中的标签（用于伪标签）

        Args:
            new_labels: 新标签数组
        """
        assert len(new_labels) == len(self.trajs), \
            f"标签数量 ({len(new_labels)}) 与轨迹数量 ({len(self.trajs)}) 不匹配"

        for i, label in enumerate(new_labels):
            self.trajs[i]['label'] = int(label) if label >= 0 else -1

    def get_label_distribution(self) -> Dict[int, int]:
        """获取标签分布"""
        from collections import Counter
        labels = [t['label'] for t in self.trajs if t['label'] >= 0]
        return dict(Counter(labels))


def traj_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数

    Args:
        batch: 样本列表

    Returns:
        批次字典
    """
    coords = [torch.FloatTensor(item['coords']) for item in batch]
    features = torch.stack([torch.FloatTensor(item['features']) for item in batch])
    lengths = torch.LongTensor([item['length'] for item in batch])
    labels = torch.LongTensor([
        item['label'] if item['label'] is not None else -1
        for item in batch
    ])
    indices = torch.LongTensor([item['index'] for item in batch])

    # 填充坐标序列
    coords_padded = pad_sequence(coords, batch_first=True)

    return {
        'coords': coords_padded,
        'features': features,
        'lengths': lengths,
        'labels': labels,
        'indices': indices
    }
