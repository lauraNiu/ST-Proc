# data/preprocessor.py
import numpy as np
from geopy.distance import great_circle  # 保留，不再使用 great_circle 计算距离
from tqdm import tqdm
from typing import List, Dict, Tuple

class AdvancedTrajectoryPreprocessor:
    """高级轨迹预处理器 - 统一本地平面尺度，扩展动态与停靠特征"""

    # 速度通道全局固定缩放参数（m/s）：
    #   walk ~1.5, bike ~4, bus/car ~10-30, subway ~15-25
    #   clip 到 [-80, 80] m/s 后除以 VXY_SCALE，保留类间绝对速度差异
    VXY_SCALE = 40.0   # 典型最大速度的一半，使输出大致落在 [-2, 2]
    VXY_CLIP  = 80.0   # 超过此速度（m/s）的点视为噪声，截断

    # 统计特征全局归一化参数（由 fit() 填充，或使用默认值）
    _feat_mean: np.ndarray = None
    _feat_std:  np.ndarray = None

    def __init__(self, max_len: int = 200):
        self.max_len = max_len

    def process(self, raw_trajectories: List[Dict]) -> List[Dict]:
        trajectories = []
        print(f"🔧 处理 {len(raw_trajectories)} 条原始轨迹...")

        for raw_traj in tqdm(raw_trajectories, desc='预处理'):
            df = raw_traj['data']
            if len(df) < 2:
                continue

            coords_ll = df[['latitude', 'longitude']].values
            times = df['timestamp'].values

            if not self._validate_coords(coords_ll):
                continue

            # 1) 本地平面坐标 (米)
            coords_xy = self._to_local_xy(coords_ll)

            # 2) 速度通道 (vx, vy) [N,2]
            if times is not None and len(times) > 1:
                dt = np.diff(times).astype(np.float32)
                dt[dt <= 0] = 1.0
                dxy = np.diff(coords_xy, axis=0)
                vxy = dxy / dt[:, None]                 # [N-1,2]
                vxy = np.vstack([vxy[0:1], vxy])        # [N,2] 与坐标长度对齐
            else:
                vxy = np.zeros_like(coords_xy, dtype=np.float32)

            # 3) 分别处理并填充到 max_len，再拼为 4 通道
            # xy: per-trajectory z-score（保留相对形状）
            coords_xy_std, valid_length = self._process_coords(coords_xy)  # [max_len,2]
            # vxy: 全局固定缩放（保留绝对速度信息，这是区分交通方式的关键）
            vxy_std, _ = self._process_velocity(vxy)                       # [max_len,2]
            coords4 = np.concatenate([coords_xy_std, vxy_std], axis=1)     # [max_len,4]

            # 4) 54维特征（本地平面/秒尺度）
            features = self._extract_features(coords_ll, times)  # 内部自行转 xy

            trajectories.append({
                'raw_coords': coords4,                   # [max_len, 4]
                'features': features.astype(np.float32), # [54,]
                'original_length': valid_length,
                'label': raw_traj['label'],
                'mode_name': raw_traj.get('mode_name'),
                'metadata': {
                    'user_id': raw_traj['user_id'],
                    'trajectory_id': raw_traj['trajectory_id']
                }
            })

        print(f"✅ 成功处理 {len(trajectories)} 条有效轨迹")

        # 统计特征全局归一化（消除量纲差异，如 total_dist 1e5 vs dwell_ratio 0-1）
        if len(trajectories) > 0:
            all_feats = np.stack([t['features'] for t in trajectories])  # [N, 48]
            feat_mean = all_feats.mean(axis=0)
            feat_std  = all_feats.std(axis=0) + 1e-6
            for t in trajectories:
                t['features'] = (t['features'] - feat_mean) / feat_std
            # 保存归一化参数供推理时使用
            self._feat_mean = feat_mean
            self._feat_std  = feat_std

        return trajectories

    def _to_local_xy(self, latlon: np.ndarray) -> np.ndarray:
        if len(latlon) == 0:
            return latlon
        R = 6371000.0
        lat0 = np.deg2rad(latlon[0, 0])
        lon0 = np.deg2rad(latlon[0, 1])
        lat = np.deg2rad(latlon[:, 0])
        lon = np.deg2rad(latlon[:, 1])
        x = (lon - lon0) * np.cos((lat + lat0) / 2.0) * R
        y = (lat - lat0) * R
        return np.stack([x, y], axis=1)

    def _validate_coords(self, coords: np.ndarray) -> bool:
        if len(coords) < 2:
            return False
        if not (np.all((coords[:, 0] > -90) & (coords[:, 0] < 90))):
            return False
        if not (np.all((coords[:, 1] > -180) & (coords[:, 1] < 180))):
            return False
        return True

    def _sample_and_pad(self, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        """均匀采样 + 零填充，返回 [max_len, D] 和有效长度"""
        D = arr.shape[1]
        if len(arr) > self.max_len:
            idx = np.linspace(0, len(arr) - 1, self.max_len, dtype=int)
            sampled = arr[idx]
            valid_length = self.max_len
        else:
            sampled = arr
            valid_length = len(arr)
        if len(sampled) < self.max_len:
            pad = np.zeros((self.max_len - len(sampled), D), dtype=np.float32)
            padded = np.vstack([sampled, pad])
        else:
            padded = sampled.copy()
        return padded.astype(np.float32), valid_length

    def _process_coords(self, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        """xy坐标：per-trajectory z-score（保留相对形状）"""
        padded, valid_length = self._sample_and_pad(arr)
        if valid_length > 0:
            valid = padded[:valid_length]
            mean = valid.mean(axis=0, keepdims=True)
            std = valid.std(axis=0, keepdims=True) + 1e-6
            padded[:valid_length] = (valid - mean) / std
        return padded, valid_length

    def _process_velocity(self, vxy: np.ndarray) -> Tuple[np.ndarray, int]:
        """速度通道：全局固定缩放，保留绝对速度信息（区分交通方式的关键）"""
        padded, valid_length = self._sample_and_pad(vxy)
        if valid_length > 0:
            # clip 噪声后除以固定尺度，不做 per-trajectory 归一化
            padded[:valid_length] = np.clip(
                padded[:valid_length], -self.VXY_CLIP, self.VXY_CLIP
            ) / self.VXY_SCALE
        return padded, valid_length

    def _extract_features(self, coords: np.ndarray, times: np.ndarray = None) -> np.ndarray:
        """
        提取 48 维特征（全部在本地平面/秒尺度；含停靠/jerk/阈值加速度）
        """
        FEAT_DIM = 54
        features = np.zeros(FEAT_DIM, dtype=np.float32)
        n = len(coords)
        if n < 2:
            return features

        eps = 1e-6
        # 本地平面
        xy = self._to_local_xy(coords)      # [N,2]
        seg_vec = np.diff(xy, axis=0)       # [N-1,2]
        seg_dist = np.linalg.norm(seg_vec, axis=1).astype(np.float32)
        total_dist = float(seg_dist.sum())

        # 时间/速度/加速度/jerk
        speed = np.array([], dtype=np.float32)
        acc = np.array([], dtype=np.float32)
        jerk = np.array([], dtype=np.float32)
        duration = 0.0
        if times is not None and len(times) > 1:
            dt = np.diff(times).astype(np.float32)
            dt[dt <= 0] = 1.0
            duration = float(times[-1] - times[0])
            speed = seg_dist / dt
            if len(speed) > 1:
                acc = np.diff(speed) / dt[:-1]
                if len(acc) > 1:
                    jerk = np.diff(acc) / dt[:-2]

        # 转弯角（基于本地平面）
        angles = np.array(self._calculate_turning_angles(xy), dtype=np.float32) if n > 2 else np.array([], dtype=np.float32)

        # 距离统计 (0-5)
        features[0] = total_dist
        if len(seg_dist) > 0:
            features[1] = float(seg_dist.mean())
            features[2] = float(seg_dist.std())
            features[3] = float(seg_dist.max())
            features[4] = float(seg_dist.min())
            features[5] = float(np.median(seg_dist))

        # 速度/加速度统计 (6-16)
        if len(speed) > 0:
            features[6]  = float(speed.mean())
            features[7]  = float(speed.std())
            features[8]  = float(np.percentile(speed, 95))
            features[9]  = float(speed.max())
            features[10] = float(speed.min())
            features[11] = float(np.median(speed))
            features[12] = float(np.percentile(speed, 25))
            features[13] = float(np.percentile(speed, 75))
            if len(acc) > 0:
                abs_acc = np.abs(acc)
                features[14] = float(abs_acc.mean())
                features[15] = float(abs_acc.max())
                features[16] = float(abs_acc.std())

        # 转弯角统计 (17-21)
        if angles.size > 0:
            features[17] = float(angles.mean())
            features[18] = float(angles.std())
            features[19] = float(angles.max())
            features[20] = float((angles > 90.0).sum() / len(angles))
            features[21] = float(np.median(angles))

        # 速度分布 (22-25)
        if len(speed) > 0:
            stop_mask = speed < 0.5
            slow_mask = (speed >= 0.5) & (speed < 5.0)
            med_mask  = (speed >= 5.0) & (speed < 20.0)
            fast_mask = speed >= 20.0
            features[22] = float(stop_mask.mean())
            features[23] = float(slow_mask.mean())
            features[24] = float(med_mask.mean())
            features[25] = float(fast_mask.mean())

        # 空间特征 (26-29) — 本地平面
        if n > 0:
            range_x = float(np.ptp(xy[:, 0]))
            range_y = float(np.ptp(xy[:, 1]))
            centroid = xy.mean(axis=0)
            gyration = float(np.sqrt(((xy - centroid) ** 2).sum(axis=1).mean()))
            diag_len = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
            features[26] = range_x
            features[27] = range_y
            features[28] = gyration
            features[29] = diag_len

        # 形状/时长 (30-35)
        straight = float(np.linalg.norm(xy[-1] - xy[0]))
        features[30] = float(n)
        features[31] = float(straight / (total_dist + eps))   # 直线度
        features[32] = float(total_dist / (straight + eps))   # 曲折度
        features[33] = float(np.mean(np.abs(np.diff(angles))) / n) if angles.size > 1 else 0.0
        features[34] = straight
        features[35] = float(duration)

        # 新增：停靠/jerk 等 (36-47)
        stops_per_km = 0.0; dwell_ratio = 0.0; avg_dwell = 0.0; p95_dwell = 0.0
        cv_speed = 0.0; p95_abs_jerk = 0.0
        brake_ratio = 0.0; accel_ratio = 0.0
        mean_acc_signed = 0.0; max_acc_signed = 0.0; min_acc_signed = 0.0; p95_abs_acc = 0.0

        if len(speed) > 0:
            stop_thr = 0.5
            stop_mask = speed < stop_thr
            diffs = np.diff(np.concatenate(([0], stop_mask.astype(np.int8), [0])))
            starts = np.where(diffs == 1)[0]; ends = np.where(diffs == -1)[0]
            n_stop = len(starts)

            if total_dist > 0:
                stops_per_km = float(n_stop / (total_dist / 1000.0 + eps))
            dwell_ratio = float(stop_mask.mean())

            if times is not None and len(times) > 1:
                dt = np.diff(times).astype(np.float32); dt[dt <= 0] = 1.0
                dwell_times = [float(dt[s:e].sum()) for s, e in zip(starts, ends)]
                if dwell_times:
                    avg_dwell = float(np.mean(dwell_times))
                    p95_dwell = float(np.percentile(dwell_times, 95))

            mu = speed.mean()
            cv_speed = float(speed.std() / (mu + eps))
            if len(jerk) > 0:
                p95_abs_jerk = float(np.percentile(np.abs(jerk), 95))

        if len(acc) > 0:
            a_thr = 1.5  # m/s^2，可按数据分布校准
            brake_ratio = float((acc < -a_thr).sum() / len(acc))
            accel_ratio = float((acc >  a_thr).sum() / len(acc))
            mean_acc_signed = float(acc.mean())
            max_acc_signed  = float(acc.max())
            min_acc_signed  = float(acc.min())
            p95_abs_acc     = float(np.percentile(np.abs(acc), 95))

        features[36] = stops_per_km
        features[37] = dwell_ratio
        features[38] = avg_dwell
        features[39] = p95_dwell
        features[40] = cv_speed
        features[41] = p95_abs_jerk
        features[42] = brake_ratio
        features[43] = accel_ratio
        features[44] = mean_acc_signed
        features[45] = max_acc_signed
        features[46] = min_acc_signed
        features[47] = p95_abs_acc

        # ===== 地铁专项特征 (48-53) =====
        # 地铁特征：隧道检测（GPS 信号丢失）、稳速巡行、站间规律性

        # [48] gap_ratio：GPS 时间间隔 > 30s 的段占比（隧道/地下信号中断）
        gap_ratio = 0.0
        max_gap_duration = 0.0
        if times is not None and len(times) > 1:
            dt_all = np.diff(times).astype(np.float32)
            dt_all[dt_all <= 0] = 0.0
            gap_mask = dt_all > 30.0
            gap_ratio = float(gap_mask.mean())
            if gap_mask.any():
                max_gap_duration = float(dt_all[gap_mask].max())

        # [49] speed_stability：高速段（>5 m/s）速度的稳定性 = 1 / (1 + cv)
        # 地铁稳速巡行 → cv 极小 → speed_stability 高（接近 1）
        # 公交/汽车加减速多 → cv 大 → speed_stability 低
        speed_stability = 0.0
        if len(speed) > 0:
            high_speed_mask = speed > 5.0
            if high_speed_mask.sum() > 2:
                hs = speed[high_speed_mask]
                cv_hs = float(hs.std() / (hs.mean() + eps))
                speed_stability = float(1.0 / (1.0 + cv_hs))

        # [50] stop_interval_cv：停靠间隔变异系数（站间距规律性）
        # 地铁站间距较均匀 → cv 小；公交/出租变化大 → cv 大
        stop_interval_cv = 0.0
        if len(speed) > 1 and times is not None and len(times) > 1:
            dt_sp = np.diff(times).astype(np.float32)
            dt_sp[dt_sp <= 0] = 1.0
            stop_mask_sp = speed < 0.5
            diffs_sp = np.diff(np.concatenate(([0], stop_mask_sp.astype(np.int8), [0])))
            starts_sp = np.where(diffs_sp == 1)[0]
            ends_sp = np.where(diffs_sp == -1)[0]
            if len(starts_sp) >= 2:
                stop_centers = [(s + e) / 2.0 for s, e in zip(starts_sp, ends_sp)]
                intervals = np.diff(stop_centers)
                if len(intervals) > 0:
                    stop_interval_cv = float(np.std(intervals) / (np.mean(intervals) + eps))

        # [51] high_speed_ratio_15：速度 > 15 m/s 的点比例（地铁/高速公路特征）
        high_speed_ratio_15 = 0.0
        if len(speed) > 0:
            high_speed_ratio_15 = float((speed > 15.0).mean())

        # [52] transit_cruise_ratio：10-30 m/s 且加速度绝对值 < 1 m/s² 的段占比
        # 地铁在此速度区间且非常平稳
        transit_cruise_ratio = 0.0
        if len(speed) > 1 and len(acc) > 0:
            n_min = min(len(speed) - 1, len(acc))
            cruise_mask = (
                (speed[1:n_min + 1] > 10.0) &
                (speed[1:n_min + 1] < 30.0) &
                (np.abs(acc[:n_min]) < 1.0)
            )
            transit_cruise_ratio = float(cruise_mask.mean())

        # [53] max_gap_duration：最大单次 GPS 信号中断时长（秒）
        # 地铁隧道内可长达 60-300s，地面交通通常 < 10s

        features[48] = gap_ratio
        features[49] = speed_stability
        features[50] = stop_interval_cv
        features[51] = high_speed_ratio_15
        features[52] = transit_cruise_ratio
        features[53] = max_gap_duration

        return features

    def _calculate_turning_angles(self, coords: np.ndarray) -> List[float]:
        """在本地平面坐标上计算转角（度）"""
        angles = []
        for i in range(1, len(coords) - 1):
            v1 = coords[i] - coords[i - 1]
            v2 = coords[i + 1] - coords[i]
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos = np.dot(v1, v2) / (n1 * n2)
                cos = np.clip(cos, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cos)))
        return angles
