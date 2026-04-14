"""
GeoLife数据集加载器
支持时间重叠标签匹配和多用户轨迹加载
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple


class ImprovedGeoLifeDataLoader:
    """改进的GeoLife数据集加载器 - 支持时间重叠标签匹配"""

    def __init__(self, data_root: str, label_mapping: Dict[int, int] = None,
                 min_overlap: float = 0.35):
        """
        初始化数据加载器

        Args:
            data_root: GeoLife数据集根目录
            label_mapping: 标签映射字典（原始标签 -> 新标签，-1表示忽略）
        """
        self.data_root = data_root

        # 原始标签映射（用于读取labels.txt）
        self.original_label_mapping = {
            'walk': 0, 'bike': 1, 'bus': 2, 'car': 3, 'subway': 4,
            'train': 5, 'airplane': 6, 'boat': 7, 'run': 8,
            'motorcycle': 9, 'taxi': 10
        }
        self.min_overlap = float(min_overlap)

        # 标签转换映射（原始 -> 合并后）
        self.label_mapping = label_mapping or {
            0: 0,  # walk
            1: 1,  # bike
            2: 2,  # bus
            3: 3,  # car
            4: 4,  # subway
            5: -1,  # train -> 忽略
            6: -1,  # airplane -> 忽略
            7: -1,  # boat -> 忽略
            8: -1,  # run -> 忽略
            9: -1,  # motorcycle -> 忽略
            10: 3  # taxi -> car
        }

    def load_all_data(self,
                      max_users: Optional[int] = None,
                      min_points: int = 20,
                      only_labeled_users: bool = True,
                      require_valid_label: bool = True) -> List[Dict]:
        """
        加载所有用户的轨迹数据

        Args:
            max_users: 最大用户数量限制
            min_points: 轨迹最小点数阈值
            only_labeled_users: 是否只加载有标签的用户
            require_valid_label:  是否要求轨迹必须有有效标签

        Returns:
            轨迹列表，每个元素为轨迹字典
        """
        trajectories = []
        user_dirs = sorted([
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
        ])

        # 如果只要有标签的用户，先过滤
        if only_labeled_users:
            labeled_users = []
            for user_id in user_dirs:
                labels_file = os.path.join(self.data_root, user_id, 'labels.txt')
                if os.path.exists(labels_file):
                    labeled_users.append(user_id)
            user_dirs = labeled_users
            print(f"📌 只使用有标签的用户: {len(user_dirs)}/{len(os.listdir(self.data_root))} 个用户")

        if max_users:
            user_dirs = user_dirs[:max_users]

        print(f"🔄 加载 {len(user_dirs)} 个用户的数据...")

        total_loaded = 0
        total_filtered = 0

        for user_id in tqdm(user_dirs, desc="加载用户"):
            user_trajs = self._load_user_trajectories(
                user_id,
                min_points,
                require_valid_label=require_valid_label  # ✅ 传递参数
            )

            # 统计加载和过滤的数量
            loaded_count = len(user_trajs)
            total_loaded += loaded_count

            trajectories.extend(user_trajs)

        if require_valid_label:
            original_count = len(trajectories)
            trajectories = [t for t in trajectories if t['label'] is not None and t['label'] >= 0]
            excluded_count = original_count - len(trajectories)

            if excluded_count > 0:
                print(f"⚠️  二次过滤排除了 {excluded_count} 条无效标签轨迹")

            print(f"✅ 成功加载 {len(trajectories)} 条有效轨迹")
        else:
            print(f"✅ 成功加载 {len(trajectories)} 条轨迹（含无标签轨迹）")

        print(f"📊 总加载: {total_loaded}, 最终保留: {len(trajectories)}")

        # 统计标签分布
        self._print_label_distribution(trajectories)

        return trajectories

    def _load_user_trajectories(self,
                                user_id: str,
                                min_points: int,
                                require_valid_label: bool = True) -> List[Dict]:  # ✅ 新增参数
        """
        加载单个用户的轨迹（只保留有有效标签的轨迹）

        Args:
            user_id: 用户ID
            min_points: 最小点数
            require_valid_label: 是否要求必须有有效标签

        Returns:
            用户轨迹列表（只包含有效标签的轨迹）
        """
        trajectories = []
        trajectory_dir = os.path.join(self.data_root, user_id, 'Trajectory')

        if not os.path.exists(trajectory_dir):
            return trajectories

        # 加载标签列表（基于时间范围）
        labels_list = self._load_labels(user_id)

        # 如果要求有效标签但用户没有标签，直接返回
        if require_valid_label and not labels_list:
            return trajectories

        # 遍历plt文件
        plt_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.plt')]

        for plt_file in plt_files:
            file_path = os.path.join(trajectory_dir, plt_file)

            try:
                # 读取plt文件（跳过前6行）
                df = pd.read_csv(
                    file_path,
                    skiprows=6,
                    header=None,
                    names=['latitude', 'longitude', 'zero', 'altitude',
                           'days', 'date', 'time']
                )

                if len(df) < min_points:
                    continue

                # 合并日期和时间
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                df['timestamp'] = df['datetime'].astype(np.int64) // 10 ** 9

                # 获取轨迹开始/结束时间
                traj_start = df['datetime'].iloc[0]
                traj_end = df['datetime'].iloc[-1]

                # 查找最佳匹配标签
                label, mode_name = self._find_best_label(
                    labels_list, traj_start, traj_end
                )

                # 关键过滤：只保留有有效标签的轨迹
                if require_valid_label and (label is None or label < 0):
                    continue  # 跳过无标签或无效标签的轨迹

                trajectories.append({
                    'user_id': user_id,
                    'trajectory_id': plt_file.replace('.plt', ''),
                    'data': df[['latitude', 'longitude', 'timestamp', 'datetime']],
                    'label': label if label is not None and label >= 0 else -1,
                    'mode_name': mode_name
                })

            except Exception as e:
                # 静默忽略损坏的文件
                continue

        return trajectories

    def _load_labels(self, user_id: str) -> List[Dict]:
        """
        加载用户的标签文件 - 返回时间范围列表（应用标签合并）

        Args:
            user_id: 用户ID

        Returns:
            标签信息列表
        """
        labels_list = []
        labels_file = os.path.join(self.data_root, user_id, 'labels.txt')

        if not os.path.exists(labels_file):
            return labels_list

        try:
            with open(labels_file, 'r') as f:
                lines = f.readlines()[1:]  # 跳过表头

            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    start_time_str = parts[0]
                    end_time_str = parts[1]
                    mode = parts[2].lower()

                    # 获取原始标签
                    if mode in self.original_label_mapping:
                        original_label = self.original_label_mapping[mode]

                        # 转换为合并后的标签
                        merged_label = self.label_mapping.get(original_label, -1)

                        # 只保留有效标签（不是-1）
                        if merged_label >= 0:
                            start_time = pd.to_datetime(start_time_str)
                            end_time = pd.to_datetime(end_time_str)

                            labels_list.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'label': merged_label,
                                'mode_name': mode,
                                'original_label': original_label
                            })
        except Exception:
            pass

        return labels_list

    def _find_best_label(self,
                        labels_list: List[Dict],
                        traj_start: pd.Timestamp,
                        traj_end: pd.Timestamp) -> Tuple[Optional[int], Optional[str]]:
        """
        查找最佳匹配的标签 - 基于时间重叠度

        Args:
            labels_list: 标签列表
            traj_start: 轨迹开始时间
            traj_end: 轨迹结束时间

        Returns:
            (标签ID, 模式名称) 或 (None, None)
        """
        if not labels_list:
            return None, None

        best_overlap = 0
        best_label = None
        best_mode_name = None

        traj_duration = (traj_end - traj_start).total_seconds()

        for label_info in labels_list:
            label_start = label_info['start_time']
            label_end = label_info['end_time']

            # 计算时间重叠
            overlap_start = max(traj_start, label_start)
            overlap_end = min(traj_end, label_end)

            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                overlap_ratio = overlap_duration / max(traj_duration, 1)

                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_label = label_info['label']
                    best_mode_name = label_info['mode_name']

        # 遍历所有区间后，选最大重叠的标签
        if best_overlap > self.min_overlap:
            return best_label, best_mode_name
        else:
            return None, None

    def _print_label_distribution(self, trajectories: List[Dict]):
        """打印标签分布统计（显示合并后的标签）"""
        from collections import Counter

        labels = [t['label'] for t in trajectories if t['label'] is not None]
        if labels:
            label_counts = Counter(labels)

            # 使用合并后的标签名称
            merged_label_names = {
                0: 'walk', 1: 'bike', 2: 'bus',
                3: 'car', 4: 'subway'
            }

            print(f"\n📊 标签分布统计（合并后）:")
            for label_id, count in sorted(label_counts.items()):
                mode_name = merged_label_names.get(label_id, f"Unknown_{label_id}")
                print(f"   {mode_name}: {count}")
