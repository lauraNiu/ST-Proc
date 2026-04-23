# data/preprocessor.py
import numpy as np
from geopy.distance import great_circle  # 保留，不再使用 great_circle 计算距离
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional


class AdvancedTrajectoryPreprocessor:
    """高级轨迹预处理器 - 统一本地平面尺度，扩展动态与停靠特征"""

    VXY_SCALE = 40.0
    VXY_CLIP = 80.0
    _feat_mean: np.ndarray = None
    _feat_std: np.ndarray = None
    FEAT_DIM = 82

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
            altitudes = df['altitude'].values.astype(np.float32) if 'altitude' in df.columns else None

            if not self._validate_coords(coords_ll):
                continue

            coords_xy = self._to_local_xy(coords_ll)
            if times is not None and len(times) > 1:
                dt = np.diff(times).astype(np.float32)
                dt[dt <= 0] = 1.0
                dxy = np.diff(coords_xy, axis=0)
                vxy = dxy / dt[:, None]
                vxy = np.vstack([vxy[0:1], vxy])
            else:
                vxy = np.zeros_like(coords_xy, dtype=np.float32)

            coords_xy_std, valid_length = self._process_coords(coords_xy)
            vxy_std, _ = self._process_velocity(vxy)
            coords4 = np.concatenate([coords_xy_std, vxy_std], axis=1)
            features = self._extract_features(coords_ll, times, altitudes)

            trajectories.append({
                'raw_coords': coords4,
                'features': features.astype(np.float32),
                'original_length': valid_length,
                'label': raw_traj['label'],
                'mode_name': raw_traj.get('mode_name'),
                'metadata': {
                    'user_id': raw_traj['user_id'],
                    'trajectory_id': raw_traj['trajectory_id'],
                    'purity': raw_traj.get('purity', 1.0),
                }
            })

        print(f"✅ 成功处理 {len(trajectories)} 条有效轨迹")
        return trajectories

    def _to_local_xy(self, latlon: np.ndarray) -> np.ndarray:
        if len(latlon) == 0:
            return latlon
        r = 6371000.0
        lat0 = np.deg2rad(latlon[0, 0])
        lon0 = np.deg2rad(latlon[0, 1])
        lat = np.deg2rad(latlon[:, 0])
        lon = np.deg2rad(latlon[:, 1])
        x = (lon - lon0) * np.cos((lat + lat0) / 2.0) * r
        y = (lat - lat0) * r
        return np.stack([x, y], axis=1)

    def _validate_coords(self, coords: np.ndarray) -> bool:
        if len(coords) < 2:
            return False
        if not np.all((coords[:, 0] > -90) & (coords[:, 0] < 90)):
            return False
        if not np.all((coords[:, 1] > -180) & (coords[:, 1] < 180)):
            return False
        return True

    def _sample_and_pad(self, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        d = arr.shape[1]
        if len(arr) > self.max_len:
            idx = np.linspace(0, len(arr) - 1, self.max_len, dtype=int)
            sampled = arr[idx]
            valid_length = self.max_len
        else:
            sampled = arr
            valid_length = len(arr)
        if len(sampled) < self.max_len:
            pad = np.zeros((self.max_len - len(sampled), d), dtype=np.float32)
            padded = np.vstack([sampled, pad])
        else:
            padded = sampled.copy()
        return padded.astype(np.float32), valid_length

    def _process_coords(self, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        padded, valid_length = self._sample_and_pad(arr)
        if valid_length > 0:
            valid = padded[:valid_length]
            mean = valid.mean(axis=0, keepdims=True)
            std = valid.std(axis=0, keepdims=True) + 1e-6
            padded[:valid_length] = (valid - mean) / std
        return padded, valid_length

    def _process_velocity(self, vxy: np.ndarray) -> Tuple[np.ndarray, int]:
        padded, valid_length = self._sample_and_pad(vxy)
        if valid_length > 0:
            padded[:valid_length] = np.clip(
                padded[:valid_length], -self.VXY_CLIP, self.VXY_CLIP
            ) / self.VXY_SCALE
        return padded, valid_length

    def _extract_features(
        self,
        coords: np.ndarray,
        times: Optional[np.ndarray] = None,
        altitudes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        features = np.zeros(self.FEAT_DIM, dtype=np.float32)
        n = len(coords)
        if n < 2:
            return features

        eps = 1e-6
        xy = self._to_local_xy(coords)
        seg_vec = np.diff(xy, axis=0)
        seg_dist = np.linalg.norm(seg_vec, axis=1).astype(np.float32)
        total_dist = float(seg_dist.sum())

        speed = np.array([], dtype=np.float32)
        acc = np.array([], dtype=np.float32)
        jerk = np.array([], dtype=np.float32)
        duration = 0.0
        dt = np.array([], dtype=np.float32)
        if times is not None and len(times) > 1:
            dt = np.diff(times).astype(np.float32)
            dt[dt <= 0] = 1.0
            duration = float(times[-1] - times[0])
            speed = seg_dist / dt
            if len(speed) > 1:
                acc = np.diff(speed) / dt[:-1]
                if len(acc) > 1:
                    jerk = np.diff(acc) / dt[:-2]

        angles = np.array(self._calculate_turning_angles(xy), dtype=np.float32) if n > 2 else np.array([], dtype=np.float32)
        headings = np.degrees(np.arctan2(seg_vec[:, 1], seg_vec[:, 0])) if len(seg_vec) > 0 else np.array([], dtype=np.float32)
        headings = (headings + 360.0) % 360.0 if len(headings) > 0 else headings

        features[0] = total_dist
        if len(seg_dist) > 0:
            features[1] = float(seg_dist.mean())
            features[2] = float(seg_dist.std())
            features[3] = float(seg_dist.max())
            features[4] = float(seg_dist.min())
            features[5] = float(np.median(seg_dist))

        if len(speed) > 0:
            features[6] = float(speed.mean())
            features[7] = float(speed.std())
            features[8] = float(np.percentile(speed, 95))
            features[9] = float(speed.max())
            features[10] = float(speed.min())
            features[11] = float(np.median(speed))
            features[12] = float(np.percentile(speed, 25))
            features[13] = float(np.percentile(speed, 75))
            if len(acc) > 0:
                abs_acc = np.abs(acc)
                features[14] = float(abs_acc.mean())
                features[15] = float(abs_acc.max())
                features[16] = float(abs_acc.std())

        if angles.size > 0:
            features[17] = float(angles.mean())
            features[18] = float(angles.std())
            features[19] = float(angles.max())
            features[20] = float((angles > 90.0).sum() / len(angles))
            features[21] = float(np.median(angles))

        stop_mask = np.zeros_like(speed, dtype=bool)
        if len(speed) > 0:
            stop_mask = speed < 0.5
            slow_mask = (speed >= 0.5) & (speed < 5.0)
            med_mask = (speed >= 5.0) & (speed < 20.0)
            fast_mask = speed >= 20.0
            features[22] = float(stop_mask.mean())
            features[23] = float(slow_mask.mean())
            features[24] = float(med_mask.mean())
            features[25] = float(fast_mask.mean())

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

        straight = float(np.linalg.norm(xy[-1] - xy[0]))
        features[30] = float(n)
        features[31] = float(straight / (total_dist + eps))
        features[32] = float(total_dist / (straight + eps))
        features[33] = float(np.mean(np.abs(np.diff(angles))) / n) if angles.size > 1 else 0.0
        features[34] = straight
        features[35] = float(duration)

        dwell_times = []
        move_run_lengths = []
        long_stop_per_km = 0.0
        first_stop_delay_ratio = 0.0
        end_stop_ratio = 0.0
        if len(speed) > 0:
            diffs = np.diff(np.concatenate(([0], stop_mask.astype(np.int8), [0])))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            n_stop = len(starts)
            stops_per_km = float(n_stop / (total_dist / 1000.0 + eps)) if total_dist > 0 else 0.0
            dwell_ratio = float(stop_mask.mean())
            avg_dwell = 0.0
            p95_dwell = 0.0
            if dt.size > 0:
                dwell_times = [float(dt[s:e].sum()) for s, e in zip(starts, ends)]
                if dwell_times:
                    avg_dwell = float(np.mean(dwell_times))
                    p95_dwell = float(np.percentile(dwell_times, 95))
                    long_stop_per_km = float(sum(x >= 30.0 for x in dwell_times) / (total_dist / 1000.0 + eps)) if total_dist > 0 else 0.0
                    first_stop_delay_ratio = float(np.sum(dt[:starts[0]]) / (duration + eps)) if len(starts) > 0 else 0.0
                    end_stop_ratio = float(np.sum(dt[starts[-1]:ends[-1]]) / (duration + eps)) if len(starts) > 0 else 0.0
            mu = speed.mean()
            cv_speed = float(speed.std() / (mu + eps))
            p95_abs_jerk = float(np.percentile(np.abs(jerk), 95)) if len(jerk) > 0 else 0.0
        else:
            stops_per_km = dwell_ratio = avg_dwell = p95_dwell = cv_speed = p95_abs_jerk = 0.0

        if len(speed) > 0 and dt.size > 0:
            move_mask = ~stop_mask
            diffs_mv = np.diff(np.concatenate(([0], move_mask.astype(np.int8), [0])))
            mv_starts = np.where(diffs_mv == 1)[0]
            mv_ends = np.where(diffs_mv == -1)[0]
            move_run_lengths = [float(dt[s:e].sum()) for s, e in zip(mv_starts, mv_ends)]

        if len(acc) > 0:
            a_thr = 1.5
            brake_ratio = float((acc < -a_thr).sum() / len(acc))
            accel_ratio = float((acc > a_thr).sum() / len(acc))
            mean_acc_signed = float(acc.mean())
            max_acc_signed = float(acc.max())
            min_acc_signed = float(acc.min())
            p95_abs_acc = float(np.percentile(np.abs(acc), 95))
        else:
            brake_ratio = accel_ratio = mean_acc_signed = max_acc_signed = min_acc_signed = p95_abs_acc = 0.0

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

        gap_ratio = 0.0
        max_gap_duration = 0.0
        if dt.size > 0:
            gap_mask = dt > 30.0
            gap_ratio = float(gap_mask.mean())
            if gap_mask.any():
                max_gap_duration = float(dt[gap_mask].max())

        speed_stability = 0.0
        if len(speed) > 0:
            high_speed_mask = speed > 5.0
            if high_speed_mask.sum() > 2:
                hs = speed[high_speed_mask]
                cv_hs = float(hs.std() / (hs.mean() + eps))
                speed_stability = float(1.0 / (1.0 + cv_hs))

        stop_interval_cv = 0.0
        if len(speed) > 1 and dt.size > 0:
            diffs_sp = np.diff(np.concatenate(([0], stop_mask.astype(np.int8), [0])))
            starts_sp = np.where(diffs_sp == 1)[0]
            ends_sp = np.where(diffs_sp == -1)[0]
            if len(starts_sp) >= 2:
                stop_centers = [(s + e) / 2.0 for s, e in zip(starts_sp, ends_sp)]
                intervals = np.diff(stop_centers)
                if len(intervals) > 0:
                    stop_interval_cv = float(np.std(intervals) / (np.mean(intervals) + eps))

        high_speed_ratio_15 = float((speed > 15.0).mean()) if len(speed) > 0 else 0.0
        transit_cruise_ratio = 0.0
        if len(speed) > 1 and len(acc) > 0:
            n_min = min(len(speed) - 1, len(acc))
            cruise_mask = (
                (speed[1:n_min + 1] > 10.0) &
                (speed[1:n_min + 1] < 30.0) &
                (np.abs(acc[:n_min]) < 1.0)
            )
            transit_cruise_ratio = float(cruise_mask.mean())

        features[48] = gap_ratio
        features[49] = speed_stability
        features[50] = stop_interval_cv
        features[51] = high_speed_ratio_15
        features[52] = transit_cruise_ratio
        features[53] = max_gap_duration

        features[54] = self._autocorr(speed, 1)
        features[55] = self._autocorr(speed, 3)
        dom_freq, spec_entropy = self._spectral_features(speed)
        features[56] = dom_freq
        features[57] = spec_entropy
        features[58] = float(np.mean(np.diff(stop_mask.astype(np.int8)) != 0)) if len(stop_mask) > 1 else 0.0
        features[59] = self._sign_change_rate(acc)
        features[60] = self._hist_entropy(headings, bins=12, value_range=(0.0, 360.0))

        if len(seg_dist) > 1 and angles.size > 0:
            curv_len = min(len(angles), len(seg_dist) - 1)
            curvature = np.abs(angles[:curv_len]) / (seg_dist[1:1 + curv_len] + eps)
        else:
            curvature = np.array([], dtype=np.float32)
        features[61] = float(np.percentile(curvature, 50)) if curvature.size > 0 else 0.0
        features[62] = float(np.percentile(curvature, 90)) if curvature.size > 0 else 0.0
        features[63] = self._turn_burst_ratio(angles)
        features[64] = float((angles < 10.0).mean()) if angles.size > 0 else 0.0

        if dwell_times:
            features[65] = float(np.median(dwell_times))
            features[66] = float(np.std(dwell_times))
        features[67] = long_stop_per_km
        features[68] = float(np.std(move_run_lengths) / (np.mean(move_run_lengths) + eps)) if move_run_lengths else 0.0
        features[69] = first_stop_delay_ratio
        features[70] = end_stop_ratio

        features[71] = float(straight / (duration + eps)) if duration > 0 else 0.0
        features[72] = float(1.0 - dwell_ratio)
        features[73] = self._max_run_ratio(speed > 10.0)
        seg_speed_mean_mean, seg_speed_mean_std, seg_stop_mean, seg_stop_std = self._segment_pooling(speed, stop_mask, segments=4)
        features[74] = seg_speed_mean_mean
        features[75] = seg_speed_mean_std
        features[76] = seg_stop_mean
        features[77] = seg_stop_std

        alt_range, mean_abs_climb, slope_p90, alt_std = self._altitude_features(altitudes, dt, seg_dist)
        features[78] = alt_range
        features[79] = mean_abs_climb
        features[80] = slope_p90
        features[81] = alt_std
        return features

    def _calculate_turning_angles(self, coords: np.ndarray) -> List[float]:
        angles = []
        for i in range(1, len(coords) - 1):
            v1 = coords[i] - coords[i - 1]
            v2 = coords[i + 1] - coords[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                cos = np.dot(v1, v2) / (n1 * n2)
                cos = np.clip(cos, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cos)))
        return angles

    def _autocorr(self, x: np.ndarray, lag: int) -> float:
        if x is None or len(x) <= lag or lag <= 0:
            return 0.0
        x1 = x[:-lag]
        x2 = x[lag:]
        x1 = x1 - x1.mean()
        x2 = x2 - x2.mean()
        denom = np.sqrt((x1 ** 2).sum() * (x2 ** 2).sum()) + 1e-6
        return float((x1 * x2).sum() / denom)

    def _spectral_features(self, x: np.ndarray) -> Tuple[float, float]:
        if x is None or len(x) < 4:
            return 0.0, 0.0
        centered = x - x.mean()
        spec = np.abs(np.fft.rfft(centered)) ** 2
        if len(spec) <= 1:
            return 0.0, 0.0
        spec = spec[1:]
        power = spec / (spec.sum() + 1e-6)
        dom_idx = int(np.argmax(power)) + 1
        dom_freq = float(dom_idx / max(len(x), 1))
        entropy = float(-(power * np.log(power + 1e-12)).sum() / np.log(len(power) + 1e-6))
        return dom_freq, entropy

    def _hist_entropy(self, x: np.ndarray, bins: int, value_range: Tuple[float, float]) -> float:
        if x is None or len(x) == 0:
            return 0.0
        hist, _ = np.histogram(x, bins=bins, range=value_range)
        prob = hist.astype(np.float32) / (hist.sum() + 1e-6)
        nz = prob[prob > 0]
        if len(nz) == 0:
            return 0.0
        return float(-(nz * np.log(nz)).sum() / np.log(len(prob) + 1e-6))

    def _sign_change_rate(self, x: np.ndarray) -> float:
        if x is None or len(x) < 2:
            return 0.0
        x = x[np.abs(x) > 1e-3]
        if len(x) < 2:
            return 0.0
        return float(np.mean(np.sign(x[1:]) != np.sign(x[:-1])))

    def _turn_burst_ratio(self, angles: np.ndarray) -> float:
        if angles is None or len(angles) < 3:
            return 0.0
        hard_turn = angles > 45.0
        burst = hard_turn[:-1] & hard_turn[1:]
        return float(burst.mean()) if len(burst) > 0 else 0.0

    def _max_run_ratio(self, mask: np.ndarray) -> float:
        if mask is None or len(mask) == 0:
            return 0.0
        best = cur = 0
        for v in mask.astype(np.int8):
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return float(best / max(len(mask), 1))

    def _segment_pooling(self, speed: np.ndarray, stop_mask: np.ndarray, segments: int = 4) -> Tuple[float, float, float, float]:
        if speed is None or len(speed) == 0:
            return 0.0, 0.0, 0.0, 0.0
        chunks = np.array_split(np.arange(len(speed)), segments)
        speed_means = []
        stop_means = []
        for idx in chunks:
            if len(idx) == 0:
                continue
            speed_means.append(float(speed[idx].mean()))
            stop_means.append(float(stop_mask[idx].mean()))
        if not speed_means:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(np.mean(speed_means)),
            float(np.std(speed_means)),
            float(np.mean(stop_means)),
            float(np.std(stop_means)),
        )

    def _altitude_features(self, altitudes: Optional[np.ndarray], dt: np.ndarray, seg_dist: np.ndarray) -> Tuple[float, float, float, float]:
        if altitudes is None or len(altitudes) < 2:
            return 0.0, 0.0, 0.0, 0.0
        alt = np.asarray(altitudes, dtype=np.float32)
        valid_mask = np.isfinite(alt) & (alt > -500.0) & (alt < 10000.0)
        if valid_mask.sum() < 2:
            return 0.0, 0.0, 0.0, 0.0
        valid_vals = alt[valid_mask]
        median_val = np.median(valid_vals)
        alt = np.where(valid_mask, alt, median_val)
        alt = self._median_smooth(alt, kernel=5)
        dz = np.diff(alt)
        climb_rate = np.abs(dz / (dt + 1e-6)) if len(dt) == len(dz) else np.array([], dtype=np.float32)
        slope = np.abs(dz / (seg_dist + 1e-6)) if len(seg_dist) == len(dz) else np.array([], dtype=np.float32)
        alt_range = float(np.percentile(alt, 95) - np.percentile(alt, 5))
        mean_abs_climb = float(climb_rate.mean()) if len(climb_rate) > 0 else 0.0
        slope_p90 = float(np.percentile(slope, 90)) if len(slope) > 0 else 0.0
        alt_std = float(np.std(alt))
        return alt_range, mean_abs_climb, slope_p90, alt_std

    def _median_smooth(self, x: np.ndarray, kernel: int = 5) -> np.ndarray:
        if len(x) < 3 or kernel <= 1:
            return x
        pad = kernel // 2
        xp = np.pad(x, (pad, pad), mode='edge')
        out = np.empty_like(x)
        for i in range(len(x)):
            out[i] = np.median(xp[i:i + kernel])
        return out
