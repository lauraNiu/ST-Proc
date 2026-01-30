"""
详细分析模块 - 增强版
提供轨迹和聚类的深度分析，支持特征逆标准化
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from collections import Counter
import warnings
import joblib

warnings.filterwarnings('ignore')


class DetailedClusterAnalyzer:
    def __init__(self, label_names=None, save_dir='results', scaler=None, feature_schema: str = 'auto'):
        """
        feature_schema:
          - 'auto'  : 运行时根据样本特征长度自动选择（48 或 36）
          - '48'    : 强制使用新 48 维索引
          - '36'    : 使用旧 36 维索引（向后兼容）
        """
        self.save_dir = Path(save_dir);
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = scaler
        self.feature_schema = feature_schema
        # 给个默认映射，真正生效会在 analyze_clusters 里根据样本长度覆盖
        self.feature_info = self._get_feature_info(schema='48')
        self.analysis_results = {}
        self.label_names = label_names or {
            0: 'walk', 1: 'bike', 2: 'bus', 3: 'car',
            4: 'subway', 5: 'train', 6: 'airplane',
            7: 'boat', 8: 'run', 9: 'motorcycle', 10: 'taxi'
        }
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_feature_info(self, schema: str = '48') -> Dict:
        """
        返回特征索引/名称/单位映射。
        schema='48' 对应新的 48 维定义（推荐）；
        schema='36' 对应你之前的 36 维定义（向后兼容）。
        """
        if schema == '36':
            return {
                'total_distance':     {'index': 0,  'unit': 'm',    'name': 'Total Distance',       'description': '总距离'},
                'avg_speed':          {'index': 6,  'unit': 'm/s',  'name': 'Average Speed',        'description': '平均速度'},
                'max_speed':          {'index': 7,  'unit': 'm/s',  'name': 'Maximum Speed',        'description': '最大速度'},
                'stop_ratio':         {'index': 22, 'unit': '',     'name': 'Stop Ratio',           'description': '停止比例'},
                'straightness':       {'index': 31, 'unit': '',     'name': 'Straightness',         'description': '直线度'},
                'avg_acceleration':   {'index': 14, 'unit': 'm/s²', 'name': 'Avg Acceleration Abs', 'description': '平均加速度(绝对值)'},
                'turn_angle_mean':    {'index': 17, 'unit': 'deg',  'name': 'Turning Angle Mean',    'description': '平均转角'}
            }

        return {
            # 0-5: 步长距离统计（米）
            'total_distance':      {'index': 0,  'unit': 'm',    'name': 'Total Distance',        'description': '总距离'},
            'seg_len_mean':        {'index': 1,  'unit': 'm',    'name': 'SegLen Mean',           'description': '步长均值'},
            'seg_len_std':         {'index': 2,  'unit': 'm',    'name': 'SegLen Std',            'description': '步长标准差'},
            'seg_len_max':         {'index': 3,  'unit': 'm',    'name': 'SegLen Max',            'description': '步长最大值'},
            'seg_len_min':         {'index': 4,  'unit': 'm',    'name': 'SegLen Min',            'description': '步长最小值'},
            'seg_len_median':      {'index': 5,  'unit': 'm',    'name': 'SegLen Median',         'description': '步长中位数'},

            # 6-13: 速度统计（m/s）
            'avg_speed':           {'index': 6,  'unit': 'm/s',  'name': 'Average Speed',         'description': '平均速度'},
            'std_speed':           {'index': 7,  'unit': 'm/s',  'name': 'Std Speed',             'description': '速度标准差'},
            'p95_speed':           {'index': 8,  'unit': 'm/s',  'name': 'P95 Speed',             'description': '速度95分位'},
            'max_speed':           {'index': 9,  'unit': 'm/s',  'name': 'Maximum Speed',         'description': '最大速度'},
            'min_speed':           {'index': 10, 'unit': 'm/s',  'name': 'Minimum Speed',         'description': '最小速度'},
            'median_speed':        {'index': 11, 'unit': 'm/s',  'name': 'Median Speed',          'description': '速度中位数'},
            'p25_speed':           {'index': 12, 'unit': 'm/s',  'name': 'P25 Speed',             'description': '速度25分位'},
            'p75_speed':           {'index': 13, 'unit': 'm/s',  'name': 'P75 Speed',             'description': '速度75分位'},

            # 14-16: 绝对加速度统计（m/s²）
            'avg_abs_acc':         {'index': 14, 'unit': 'm/s²', 'name': 'Avg |Acc|',             'description': '绝对加速度均值'},
            'max_abs_acc':         {'index': 15, 'unit': 'm/s²', 'name': 'Max |Acc|',             'description': '绝对加速度最大'},
            'std_abs_acc':         {'index': 16, 'unit': 'm/s²', 'name': 'Std |Acc|',             'description': '绝对加速度标准差'},

            # 17-21: 转弯角统计（度）
            'turn_angle_mean':     {'index': 17, 'unit': 'deg',  'name': 'Turning Angle Mean',    'description': '平均转角'},
            'turn_angle_std':      {'index': 18, 'unit': 'deg',  'name': 'Turning Angle Std',     'description': '转角标准差'},
            'turn_angle_max':      {'index': 19, 'unit': 'deg',  'name': 'Turning Angle Max',     'description': '最大转角'},
            'turn_angle_gt90':     {'index': 20, 'unit': '',     'name': 'Angle>90 Ratio',        'description': '转角>90°比例'},
            'turn_angle_median':   {'index': 21, 'unit': 'deg',  'name': 'Turning Angle Median',  'description': '转角中位数'},

            # 22-25: 速度区间占比
            'stop_ratio':          {'index': 22, 'unit': '',     'name': 'Stop Ratio (<0.5m/s)',  'description': '停止比例'},
            'slow_ratio':          {'index': 23, 'unit': '',     'name': 'Slow Ratio',            'description': '低速比例'},
            'medium_ratio':        {'index': 24, 'unit': '',     'name': 'Medium Ratio',          'description': '中速比例'},
            'fast_ratio':          {'index': 25, 'unit': '',     'name': 'Fast Ratio',            'description': '高速比例'},

            # 26-29: 空间量纲（米）
            'range_x':             {'index': 26, 'unit': 'm',    'name': 'X Range',               'description': 'X范围'},
            'range_y':             {'index': 27, 'unit': 'm',    'name': 'Y Range',               'description': 'Y范围'},
            'gyration_radius':     {'index': 28, 'unit': 'm',    'name': 'Gyration Radius',       'description': '回转半径'},
            'diag_range':          {'index': 29, 'unit': 'm',    'name': 'Diagonal Range',        'description': '对角线长度'},

            # 30-35: 形状/时长
            'num_points':          {'index': 30, 'unit': '',     'name': 'Num Points',            'description': '轨迹点数'},
            'straightness':        {'index': 31, 'unit': '',     'name': 'Straightness',          'description': '直线度'},
            'tortuosity':          {'index': 32, 'unit': '',     'name': 'Tortuosity',            'description': '曲折度'},
            'turn_intensity':      {'index': 33, 'unit': '1/pt', 'name': 'Turn Intensity',        'description': '角变化密度'},
            'straight_line_dist':  {'index': 34, 'unit': 'm',    'name': 'Endpoint Distance',     'description': '端点直线距离'},
            'duration':            {'index': 35, 'unit': 's',    'name': 'Duration',              'description': '持续时间'},

            # 36-39: 停靠
            'stops_per_km':        {'index': 36, 'unit': '/km',  'name': 'Stops per km',          'description': '每公里停靠数'},
            'dwell_ratio':         {'index': 37, 'unit': '',     'name': 'Dwell Ratio',           'description': '停靠时间占比'},
            'avg_dwell':           {'index': 38, 'unit': 's',    'name': 'Avg Dwell',             'description': '平均驻停时长'},
            'p95_dwell':           {'index': 39, 'unit': 's',    'name': 'P95 Dwell',             'description': '95分位驻停时长'},

            # 40-47: jerk / 强制加速度 / 有符号加速度
            'cv_speed':            {'index': 40, 'unit': '',     'name': 'CV Speed',              'description': '速度变异系数'},
            'p95_abs_jerk':        {'index': 41, 'unit': 'm/s³', 'name': 'P95 |Jerk|',            'description': '95分位绝对jerk'},
            'brake_ratio':         {'index': 42, 'unit': '',     'name': 'Hard Brake Ratio',      'description': '强制制动比例'},
            'accel_ratio':         {'index': 43, 'unit': '',     'name': 'Hard Accel Ratio',      'description': '强制加速比例'},
            'mean_acc_signed':     {'index': 44, 'unit': 'm/s²', 'name': 'Mean Acc',              'description': '有符号加速度均值'},
            'max_acc_signed':      {'index': 45, 'unit': 'm/s²', 'name': 'Max Acc',               'description': '有符号加速度最大'},
            'min_acc_signed':      {'index': 46, 'unit': 'm/s²', 'name': 'Min Acc',               'description': '有符号加速度最小'},
            'p95_abs_acc':         {'index': 47, 'unit': 'm/s²', 'name': 'P95 |Acc|',             'description': '95分位绝对加速度'}
        }

    def set_scaler(self, scaler):
        """设置标准化器"""
        self.scaler = scaler

    def load_scaler(self, scaler_path: str):
        """从文件加载标准化器"""
        self.scaler = joblib.load(scaler_path)
        print(f"   ✅ 已加载标准化器: {scaler_path}")

    def inverse_transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        特征逆标准化

        Args:
            features: 标准化后的特征 [N, D]

        Returns:
            原始尺度的特征 [N, D]
        """
        if self.scaler is None:
            print("   ⚠️  未设置标准化器，返回原始特征")
            return features

        return self.scaler.inverse_transform(features)

    def analyze_clusters(self, embeddings, clusters, true_labels, trajectories=None, save_dir=None):
        """
        执行完整的聚类分析（包含逆标准化）

        Args:
            embeddings: 样本嵌入向量
            clusters: 聚类结果
            true_labels: 真实标签
            trajectories: 轨迹数据（可选）
            save_dir: 结果保存目录（可选）

        Returns:
            dict: 分析结果字典
        """
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # === 自适应 schema：根据样本特征长度动态选择 48/36 的索引映射 ===
        if getattr(self, 'feature_schema', 'auto') == 'auto':
            if trajectories and len(trajectories) > 0 and 'features' in trajectories[0]:
                feat_dim = len(trajectories[0]['features'])
                schema = '48' if feat_dim >= 48 else '36'
            else:
                # 若拿不到 trajectories，就默认按新 48 维
                schema = '48'
            self.feature_info = self._get_feature_info(schema=schema)
        # === 自适应 schema 结束 ===

        print("\n" + "=" * 80)
        print("🔬 详细聚类分析（含特征逆标准化）")
        print("=" * 80)

        # 1. 簇纯度分析
        purity_results = self._analyze_cluster_purity(clusters, true_labels)
        self.analysis_results['purity'] = purity_results

        # 2. 交通方式分布分析
        distribution_results = self._analyze_transport_distribution(clusters, true_labels)
        self.analysis_results['distribution'] = distribution_results

        # 3. 混淆模式分析
        confusion_results = self._analyze_confusion_patterns(clusters, true_labels)
        self.analysis_results['confusion'] = confusion_results

        # 4. 簇特征统计
        if trajectories is not None:
            feature_results = self._analyze_cluster_features(clusters, trajectories)
            self.analysis_results['features'] = feature_results

            # 可视化簇特征对比
            self._visualize_cluster_features(feature_results)

        # 5. 簇大小分析
        size_results = self._analyze_cluster_sizes(clusters, true_labels)
        self.analysis_results['sizes'] = size_results

        # 6. 保存分析报告
        self._save_analysis_report()

        return self.analysis_results

    def _analyze_cluster_purity(self, clusters, true_labels):
        """分析每个簇的纯度"""
        print("\n📊 簇纯度分析:")
        print("-" * 80)

        cluster_purities = []

        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            cluster_labels = true_labels[mask]
            cluster_labels = cluster_labels[cluster_labels >= 0]

            if len(cluster_labels) == 0:
                continue

            label_counts = np.bincount(cluster_labels, minlength=len(self.label_names))
            most_common_label = label_counts.argmax()
            purity = label_counts[most_common_label] / len(cluster_labels)

            label_probs = label_counts[label_counts > 0] / len(cluster_labels)
            entropy = -np.sum(label_probs * np.log2(label_probs))

            cluster_info = {
                'cluster_id': int(cluster_id),
                'total_size': int(mask.sum()),
                'labeled_size': int(len(cluster_labels)),
                'dominant_label': int(most_common_label),
                'dominant_label_name': self.label_names.get(most_common_label, 'unknown'),
                'purity': float(purity),
                'entropy': float(entropy),
                'label_distribution': {
                    int(label): int(count)
                    for label, count in enumerate(label_counts) if count > 0
                }
            }

            cluster_purities.append(cluster_info)

        cluster_purities.sort(key=lambda x: x['purity'], reverse=True)

        print(f"\n{'簇ID':<8}{'总大小':<10}{'标签数':<10}{'主导类型':<15}{'纯度':<10}{'熵':<10}")
        print("-" * 80)
        for cp in cluster_purities:
            print(f"{cp['cluster_id']:<8}{cp['total_size']:<10}{cp['labeled_size']:<10}"
                  f"{cp['dominant_label_name']:<15}{cp['purity']:<10.2%}{cp['entropy']:<10.3f}")

        avg_purity = np.mean([cp['purity'] for cp in cluster_purities])
        print(f"\n✅ 平均纯度: {avg_purity:.2%}")

        return {
            'clusters': cluster_purities,
            'average_purity': float(avg_purity)
        }

    def _analyze_transport_distribution(self, clusters, true_labels):
        """分析交通方式在簇中的分布"""
        print(f"\n🚗 交通方式分布分析:")
        print("-" * 80)

        valid_labels = true_labels[true_labels >= 0]
        if len(valid_labels) == 0:
            print("⚠️  没有有效标签，跳过交通方式分布分析")
            return {}

        labeled_mask = true_labels >= 0
        labeled_clusters = clusters[labeled_mask]
        labeled_true = true_labels[labeled_mask]

        transport_stats = []

        for label_id in np.unique(labeled_true):
            label_mask = labeled_true == label_id
            label_clusters = labeled_clusters[label_mask]

            total_count = len(label_clusters)
            cluster_dist = np.bincount(label_clusters, minlength=clusters.max() + 1)

            dominant_cluster = cluster_dist.argmax()
            concentration = cluster_dist[dominant_cluster] / total_count
            spread = len(np.where(cluster_dist > 0)[0])

            transport_info = {
                'label': int(label_id),
                'label_name': self.label_names.get(label_id, f'label_{label_id}'),
                'total_samples': int(total_count),
                'dominant_cluster': int(dominant_cluster),
                'concentration': float(concentration),
                'spread': int(spread),
                'cluster_distribution': {
                    int(c): int(count)
                    for c, count in enumerate(cluster_dist) if count > 0
                }
            }

            transport_stats.append(transport_info)

        transport_stats.sort(key=lambda x: x['concentration'], reverse=True)

        print(f"\n{'交通方式':<15}{'样本数':<10}{'主导簇':<12}{'集中度':<12}{'分散度':<10}")
        print("-" * 80)
        for ts in transport_stats:
            print(f"{ts['label_name']:<15}{ts['total_samples']:<10}{ts['dominant_cluster']:<12}"
                  f"{ts['concentration']:<12.2%}{ts['spread']:<10}")

        return {
            'transport_modes': transport_stats,
            'n_transport_modes': len(transport_stats),
            'n_clusters': len(np.unique(clusters))
        }

    def _analyze_confusion_patterns(self, clusters, true_labels):
        """分析混淆模式"""
        print(f"\n❌ 混淆模式分析:")
        print("-" * 80)

        labeled_mask = true_labels >= 0
        if labeled_mask.sum() == 0:
            print("⚠️  没有标签数据进行混淆分析")
            return {}

        labeled_clusters = clusters[labeled_mask]
        labeled_true = true_labels[labeled_mask]

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labeled_true, labeled_clusters)

        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)

        top_k = 10
        flat_indices = np.argsort(cm_no_diag.ravel())[::-1][:top_k]
        confusion_pairs = np.unravel_index(flat_indices, cm_no_diag.shape)

        confusion_patterns = []

        print("\nTop 10 混淆对 (真实标签 -> 预测簇):")
        print(f"{'排名':<6}{'真实标签':<15}{'预测簇':<10}{'混淆数量':<10}{'占比':<10}")
        print("-" * 80)

        for rank, (true_idx, pred_idx) in enumerate(zip(confusion_pairs[0], confusion_pairs[1]), 1):
            count = cm[true_idx, pred_idx]

            if count == 0:
                break

            total_true = cm[true_idx].sum()
            ratio = count / total_true if total_true > 0 else 0

            pattern_info = {
                'rank': rank,
                'true_label': int(true_idx),
                'true_label_name': self.label_names.get(true_idx, f'label_{true_idx}'),
                'predicted_cluster': int(pred_idx),
                'confusion_count': int(count),
                'confusion_ratio': float(ratio)
            }

            confusion_patterns.append(pattern_info)

            print(f"{rank:<6}{pattern_info['true_label_name']:<15}{pred_idx:<10}"
                  f"{count:<10}{ratio:<10.2%}")

        return {
            'confusion_patterns': confusion_patterns,
            'confusion_matrix': cm.tolist()
        }

    def _analyze_cluster_features(self, clusters, trajectories):
        """
        分析每个簇的特征统计（含逆标准化）
        """
        print(f"\n📐 簇特征统计（真实物理量）:")
        print("-" * 80)

        cluster_features = []

        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            cluster_trajs = [trajectories[i] for i in np.where(mask)[0]]

            if len(cluster_trajs) == 0:
                continue

            # 提取标准化的特征
            cluster_feats_normalized = np.array([t['features'] for t in cluster_trajs])

            # 逆标准化
            if self.scaler is not None:
                cluster_feats_original = self.inverse_transform_features(cluster_feats_normalized)
            else:
                cluster_feats_original = cluster_feats_normalized

            stats = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_trajs),
                'features': {}
            }

            # 统计每个特征
            for feat_name, feat_config in self.feature_info.items():
                feat_idx = feat_config['index']
                feat_values = cluster_feats_original[:, feat_idx]

                stats['features'][feat_name] = {
                    'mean': float(np.mean(feat_values)),
                    'std': float(np.std(feat_values)),
                    'min': float(np.min(feat_values)),
                    'max': float(np.max(feat_values)),
                    'median': float(np.median(feat_values)),
                    'unit': feat_config['unit'],
                    'description': feat_config['description']
                }

            cluster_features.append(stats)

            # 打印关键特征
            print(f"\n📍 簇 {cluster_id} (大小={len(cluster_trajs)}):")

            # 速度相关
            avg_speed = stats['features']['avg_speed']
            max_speed = stats['features']['max_speed']
            print(f"  🚄 平均速度: {avg_speed['mean']:.2f} ± {avg_speed['std']:.2f} {avg_speed['unit']}")
            print(f"  🚄 最大速度: {max_speed['mean']:.2f} {max_speed['unit']}")

            # 距离
            total_dist = stats['features']['total_distance']
            print(f"  📏 平均距离: {total_dist['mean']:.2f} {total_dist['unit']}")

            # 其他特征
            stop_ratio = stats['features']['stop_ratio']
            straightness = stats['features']['straightness']
            print(f"  ⏸️  停止比例: {stop_ratio['mean']:.2%}")
            print(f"  ➡️  直线度: {straightness['mean']:.3f}")

        return cluster_features

    def _analyze_cluster_sizes(self, clusters, true_labels):
        """分析簇大小分布"""
        print(f"\n📏 簇大小分析:")
        print("-" * 80)

        cluster_sizes = []

        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            total_size = mask.sum()

            labeled_in_cluster = (true_labels[mask] >= 0).sum()

            cluster_sizes.append({
                'cluster_id': int(cluster_id),
                'total_size': int(total_size),
                'labeled_size': int(labeled_in_cluster),
                'labeled_ratio': float(labeled_in_cluster / total_size) if total_size > 0 else 0
            })

        cluster_sizes.sort(key=lambda x: x['total_size'], reverse=True)

        print(f"\n{'簇ID':<10}{'总大小':<12}{'有标签':<12}{'标签比例':<12}")
        print("-" * 80)
        for cs in cluster_sizes:
            print(f"{cs['cluster_id']:<10}{cs['total_size']:<12}{cs['labeled_size']:<12}"
                  f"{cs['labeled_ratio']:<12.2%}")

        return cluster_sizes

    def _visualize_cluster_features(self, cluster_features: List[Dict]):
        """
        可视化簇特征对比
        """
        print("\n📊 生成特征可视化...")

        # 准备数据
        cluster_ids = [cf['cluster_id'] for cf in cluster_features]
        sizes = [cf['size'] for cf in cluster_features]

        # 选择关键特征进行可视化
        key_features = ['avg_speed', 'total_distance', 'stop_ratio', 'straightness']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        for idx, feat_name in enumerate(key_features):
            ax = axes[idx]

            means = [cf['features'][feat_name]['mean'] for cf in cluster_features]
            stds = [cf['features'][feat_name]['std'] for cf in cluster_features]
            unit = cluster_features[0]['features'][feat_name]['unit']
            description = cluster_features[0]['features'][feat_name]['description']

            # 柱状图 + 误差棒
            x = np.arange(len(cluster_ids))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                          color='skyblue', edgecolor='black')

            ax.set_xlabel('Cluster ID', fontsize=11)
            ylabel = f'{description}'
            if unit:
                ylabel += f' ({unit})'
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Cluster Feature: {description}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(cluster_ids)
            ax.grid(axis='y', alpha=0.3)

            # 标注数值
            for i, (bar, mean_val) in enumerate(zip(bars, means)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{mean_val:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        save_path = self.save_dir / 'cluster_features_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ 特征对比图已保存: {save_path.name}")

        # 额外生成热图
        self._plot_feature_heatmap(cluster_features)

    def _plot_feature_heatmap(self, cluster_features: List[Dict]):
        """绘制特征热图"""
        cluster_ids = [cf['cluster_id'] for cf in cluster_features]

        # 选择所有特征
        feature_names = list(self.feature_info.keys())

        # 构建矩阵
        feature_matrix = []
        for cf in cluster_features:
            row = [cf['features'][fn]['mean'] for fn in feature_names]
            feature_matrix.append(row)

        feature_matrix = np.array(feature_matrix)

        # 归一化到[0, 1]用于显示
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        feature_matrix_norm = scaler.fit_transform(feature_matrix.T).T

        fig, ax = plt.subplots(figsize=(14, 8))

        # 绘制热图
        im = ax.imshow(feature_matrix_norm, cmap='YlOrRd', aspect='auto')

        # 设置刻度
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(cluster_ids)))
        ax.set_xticklabels([self.feature_info[fn]['description'] for fn in feature_names],
                           rotation=45, ha='right')
        ax.set_yticklabels([f'Cluster {cid}' for cid in cluster_ids])

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value', rotation=270, labelpad=20)

        # 添加数值标注
        for i in range(len(cluster_ids)):
            for j in range(len(feature_names)):
                text = ax.text(j, i, f'{feature_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Cluster Feature Heatmap (Original Scale)',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        save_path = self.save_dir / 'cluster_features_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✅ 特征热图已保存: {save_path.name}")

    def _save_analysis_report(self):
        """保存分析报告为JSON和Excel"""
        # 1. JSON格式
        report_path = self.save_dir / 'cluster_analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=4, ensure_ascii=False)
        print(f"\n   ✅ 分析报告已保存: {report_path.name}")

        if 'features' in self.analysis_results:
            self._save_features_to_excel()

    def _save_features_to_excel(self):
        """将簇特征保存为Excel"""
        try:
            cluster_features = self.analysis_results['features']

            # 构建DataFrame
            rows = []
            for cf in cluster_features:
                cluster_id = cf['cluster_id']
                size = cf['size']

                row = {'Cluster ID': cluster_id, 'Size': size}

                for feat_name, feat_stats in cf['features'].items():
                    desc = feat_stats['description']
                    unit = f" ({feat_stats['unit']})" if feat_stats['unit'] else ""

                    row[f'{desc}{unit} - Mean'] = feat_stats['mean']
                    row[f'{desc}{unit} - Std'] = feat_stats['std']
                    row[f'{desc}{unit} - Min'] = feat_stats['min']
                    row[f'{desc}{unit} - Max'] = feat_stats['max']

                rows.append(row)

            df = pd.DataFrame(rows)

            excel_path = self.save_dir / 'cluster_features_detailed.xlsx'
            df.to_excel(excel_path, index=False, sheet_name='Cluster Features')

            print(f"   ✅ 特征详情已保存: {excel_path.name}")

        except Exception as e:
            print(f"   ⚠️  Excel保存失败: {e}")