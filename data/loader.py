"""
GeoLife数据集加载器
支持时间边界切分、多模态轨迹过滤和多用户轨迹加载
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class ImprovedGeoLifeDataLoader:
    """改进的GeoLife数据集加载器 - 支持时间边界切分与 purity 过滤"""

    def __init__(
        self,
        data_root: str,
        label_mapping: Dict[int, int] = None,
        min_overlap: float = 0.35,
        segment_by_label: bool = True,
        min_label_purity: float = 0.80,
        min_segment_points: int = 20,
        drop_mixed_segments: bool = True,
        keep_unlabeled_segments: bool = False,
        label_names: Dict[int, str] = None,
    ):
        self.data_root = data_root
        self.original_label_mapping = {
            'walk': 0, 'bike': 1, 'bus': 2, 'car': 3, 'subway': 4,
            'train': 5, 'airplane': 6, 'boat': 7, 'run': 8,
            'motorcycle': 9, 'taxi': 10
        }
        self.min_overlap = float(min_overlap)
        self.segment_by_label = bool(segment_by_label)
        self.min_label_purity = float(min_label_purity)
        self.min_segment_points = int(min_segment_points)
        self.drop_mixed_segments = bool(drop_mixed_segments)
        self.keep_unlabeled_segments = bool(keep_unlabeled_segments)
        self.label_mapping = label_mapping or {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
            5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: 3
        }
        self.label_names = label_names or {
            v: str(v) for v in sorted(set(x for x in self.label_mapping.values() if x >= 0))
        }

    def load_all_data(
        self,
        max_users: Optional[int] = None,
        min_points: int = 20,
        only_labeled_users: bool = True,
        require_valid_label: bool = True,
    ) -> List[Dict]:
        trajectories = []
        user_dirs = sorted([
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
        ])

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

        for user_id in tqdm(user_dirs, desc="加载用户"):
            user_trajs = self._load_user_trajectories(
                user_id,
                min_points,
                require_valid_label=require_valid_label,
            )
            total_loaded += len(user_trajs)
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
        self._print_label_distribution(trajectories)
        return trajectories

    def _load_user_trajectories(
        self,
        user_id: str,
        min_points: int,
        require_valid_label: bool = True,
    ) -> List[Dict]:
        trajectories = []
        trajectory_dir = os.path.join(self.data_root, user_id, 'Trajectory')
        if not os.path.exists(trajectory_dir):
            return trajectories

        labels_list = self._load_labels(user_id)
        if require_valid_label and not labels_list:
            return trajectories

        plt_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.plt')]
        for plt_file in plt_files:
            file_path = os.path.join(trajectory_dir, plt_file)
            try:
                df = pd.read_csv(
                    file_path,
                    skiprows=6,
                    header=None,
                    names=['latitude', 'longitude', 'zero', 'altitude', 'days', 'date', 'time']
                )
                if len(df) < min_points:
                    continue
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                df['timestamp'] = df['datetime'].astype(np.int64) // 10 ** 9
                df = df[['latitude', 'longitude', 'altitude', 'timestamp', 'datetime']].sort_values('datetime').reset_index(drop=True)

                if self.segment_by_label and labels_list:
                    # In segmented mode, each valid label interval becomes one kept sample.
                    # We remove multimodal mixing by temporal cutting, not by later purity dropping.
                    segments = self._segment_trajectory_by_labels(
                        user_id=user_id,
                        plt_file=plt_file,
                        df=df,
                        labels_list=labels_list,
                        require_valid_label=require_valid_label,
                    )
                    trajectories.extend([
                        seg for seg in segments
                        if len(seg['data']) >= self.min_segment_points
                    ])
                    continue

                traj_start = df['datetime'].iloc[0]
                traj_end = df['datetime'].iloc[-1]
                label, mode_name, purity = self._find_best_label(labels_list, traj_start, traj_end)
                if require_valid_label and (label is None or label < 0):
                    continue
                if label is not None and label >= 0 and self.drop_mixed_segments and purity < self.min_label_purity:
                    continue
                trajectories.append({
                    'user_id': user_id,
                    'trajectory_id': plt_file.replace('.plt', ''),
                    'data': df,
                    'label': label if label is not None and label >= 0 else -1,
                    'mode_name': mode_name,
                    'purity': purity,
                })
            except Exception:
                continue
        return trajectories

    def _segment_trajectory_by_labels(
        self,
        user_id: str,
        plt_file: str,
        df: pd.DataFrame,
        labels_list: List[Dict],
        require_valid_label: bool,
    ) -> List[Dict]:
        traj_start = df['datetime'].iloc[0]
        traj_end = df['datetime'].iloc[-1]
        segments = []
        seg_idx = 0

        for label_info in labels_list:
            overlap_start = max(traj_start, label_info['start_time'])
            overlap_end = min(traj_end, label_info['end_time'])
            if overlap_start >= overlap_end:
                continue

            seg_df = df[(df['datetime'] >= overlap_start) & (df['datetime'] <= overlap_end)].copy()
            seg_df = seg_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            if len(seg_df) < self.min_segment_points:
                continue

            purity = self._compute_segment_purity(overlap_start, overlap_end, label_info['label'], labels_list)

            segments.append({
                'user_id': user_id,
                'trajectory_id': f"{plt_file.replace('.plt', '')}_seg{seg_idx:02d}",
                'data': seg_df,
                'label': label_info['label'],
                'mode_name': label_info['mode_name'],
                'purity': purity,
                'segment_start': overlap_start,
                'segment_end': overlap_end,
            })
            seg_idx += 1

        if segments:
            return segments

        if require_valid_label or not self.keep_unlabeled_segments:
            return []

        return [{
            'user_id': user_id,
            'trajectory_id': plt_file.replace('.plt', ''),
            'data': df,
            'label': -1,
            'mode_name': None,
            'purity': 0.0,
        }]

    def _compute_segment_purity(
        self,
        seg_start: pd.Timestamp,
        seg_end: pd.Timestamp,
        target_label: int,
        labels_list: List[Dict],
    ) -> float:
        total = max((seg_end - seg_start).total_seconds(), 1.0)
        target_overlap = 0.0
        all_overlap = 0.0
        for label_info in labels_list:
            overlap_start = max(seg_start, label_info['start_time'])
            overlap_end = min(seg_end, label_info['end_time'])
            if overlap_start >= overlap_end:
                continue
            overlap = (overlap_end - overlap_start).total_seconds()
            all_overlap += overlap
            if label_info['label'] == target_label:
                target_overlap += overlap
        denom = max(all_overlap, total, 1.0)
        return float(target_overlap / denom)

    def _load_labels(self, user_id: str) -> List[Dict]:
        labels_list = []
        labels_file = os.path.join(self.data_root, user_id, 'labels.txt')
        if not os.path.exists(labels_file):
            return labels_list
        try:
            with open(labels_file, 'r') as f:
                lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split('	')
                if len(parts) < 3:
                    continue
                start_time_str, end_time_str, mode = parts[0], parts[1], parts[2].lower()
                if mode not in self.original_label_mapping:
                    continue
                original_label = self.original_label_mapping[mode]
                merged_label = self.label_mapping.get(original_label, -1)
                if merged_label < 0:
                    continue
                labels_list.append({
                    'start_time': pd.to_datetime(start_time_str),
                    'end_time': pd.to_datetime(end_time_str),
                    'label': merged_label,
                    'mode_name': mode,
                    'original_label': original_label,
                })
        except Exception:
            pass
        return labels_list

    def _find_best_label(
        self,
        labels_list: List[Dict],
        traj_start: pd.Timestamp,
        traj_end: pd.Timestamp,
    ) -> Tuple[Optional[int], Optional[str], float]:
        if not labels_list:
            return None, None, 0.0
        best_overlap = 0.0
        best_label = None
        best_mode_name = None
        traj_duration = max((traj_end - traj_start).total_seconds(), 1.0)
        for label_info in labels_list:
            overlap_start = max(traj_start, label_info['start_time'])
            overlap_end = min(traj_end, label_info['end_time'])
            if overlap_start >= overlap_end:
                continue
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            overlap_ratio = overlap_duration / traj_duration
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_label = label_info['label']
                best_mode_name = label_info['mode_name']
        if best_overlap > self.min_overlap:
            return best_label, best_mode_name, float(best_overlap)
        return None, None, float(best_overlap)

    def _print_label_distribution(self, trajectories: List[Dict]):
        labels = [t['label'] for t in trajectories if t['label'] is not None and t['label'] >= 0]
        if not labels:
            return
        label_counts = Counter(labels)
        print('📈 标签分布:')
        for label_id, count in sorted(label_counts.items()):
            mode_name = self.label_names.get(label_id, f'Unknown_{label_id}')
            print(f'   {mode_name}: {count}')
