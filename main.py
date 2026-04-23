# main.py
"""
轨迹交通方式识别
"""

import os
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime

import torch

# 导入项目模块
from config.config import Config
from data.loader import ImprovedGeoLifeDataLoader
from data.preprocessor import AdvancedTrajectoryPreprocessor
from data.dataset import TrajDataset, traj_collate_fn
from data.augmentation import MultiScaleAugmenter
from models.encoders import AdaptiveTrajectoryEncoder, GraphAggregationLayer
from models.projectors import ProjectionHead, ClassifierHead
from models.learners import SemiSupervisedPrototypicalLearner, PrototypicalContrastiveLearner
from training.trainer import SemiSupervisedTrainer
from training.loss import compute_prototype_logits
from training.pseudo_label import AdvancedPseudoLabelGenerator, PseudoLabelGenerator
from evaluation.clustering import (
    ClusteringEvaluator,
    ClusterLabelMapper,
    perform_clustering
)
from evaluation.metrics import TransportModeEvaluator

# 修改: 使用正确的可视化类
from utils.visualization import (
    EmbeddingVisualizer,
    TrainingVisualizer,
    ClusterVisualizer
)
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# 修改: 使用新的日志系统
from utils.logger import get_logger
from utils.helper import (
    set_seed,
    seed_worker,
    create_torch_generator,
    ensure_dir,
    get_device,
    Timer,
    make_json_serializable  # 添加这个导入
)
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')




class TransportModeRecognitionPipeline:
    """交通方式识别"""

    def __init__(self, config: Config, mode='train'):
        """
        Args:
            config: Config对象
            mode: 运行模式 ('train', 'eval', 'inference')
        """
        self.config = config
        self.mode = mode

        self.device = get_device(use_cuda=(config.experiment.device == 'cuda'))
        self.exp_dir = Path(config.exp_dir)
        self.logger = get_logger(
            exp_name=config.experiment.exp_name,
            log_dir=str(self.exp_dir.parent)
        )

        self.logger.log_section(f"🚀 初始化交通方式识别系统 (模式: {mode})")
        self.logger.info(f"📁 实验目录: {self.exp_dir}")
        self.logger.info(f"🖥️  设备: {self.device}")
        self.seed = int(config.data.random_seed)
        self.deterministic = bool(getattr(config.experiment, 'deterministic', True))
        set_seed(self.seed, deterministic=self.deterministic)

        # 标签名称映射
        self.label_names = config.label_names

        self.data_loader = None
        self.preprocessor = None
        self.train_dataset = None
        self.val_dataset = None
        self.encoder = None
        self.projector = None
        self.graph_agg = None
        self.trainer = None
        self.scaler = None

        # 初始化可视化器
        self.embedding_viz = EmbeddingVisualizer()
        self.training_viz = TrainingVisualizer()
        self.cluster_viz = ClusterVisualizer()
        from utils.visualization import UnifiedVisualizer
        self.checkpoint_dir=self.config.checkpoint_dir

        self.visualizer = UnifiedVisualizer(
            save_dir=str(self.exp_dir / 'results')
        )

    def _split_dataset(self, trajectories: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        数据集划分（考虑标签分布）
        """
        from sklearn.model_selection import train_test_split
        from collections import Counter

        labels_array = np.array([
            t['label'] if t['label'] is not None else -1
            for t in trajectories
        ])

        # 智能判断是否使用分层划分
        use_stratify = False
        stratify_array = None
        labeled_mask = labels_array >= 0

        if labeled_mask.sum() > 10:
            label_counts = Counter(labels_array[labeled_mask])
            min_samples_per_class = min(label_counts.values())

            if min_samples_per_class >= 2:
                use_stratify = True
                stratify_array = labels_array
                self.logger.info(f"✅ 使用分层划分（最小类别样本数: {min_samples_per_class}）")
            else:
                self.logger.warning("⚠️  类别样本不足，使用随机划分")
        else:
            self.logger.warning("⚠️  有标签样本不足，使用随机划分")

        # 数据划分
        train_indices, val_indices = train_test_split(
            np.arange(len(trajectories)),
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_seed,
            stratify=stratify_array if use_stratify else None
        )

        train_trajectories = [trajectories[i] for i in train_indices]
        val_trajectories = [trajectories[i] for i in val_indices]

        return train_trajectories, val_trajectories

    def setup_data(self):
        """步骤1: 设置数据加载和预处理"""
        self.logger.log_section("📂 步骤1: 数据加载与预处理")

        with Timer("数据加载"):
            self.data_loader = ImprovedGeoLifeDataLoader(
                data_root=self.config.data.data_root,
                label_mapping=self.config.label_mapping,
                min_overlap=self.config.data.get('min_label_overlap', 0.35),
                segment_by_label=self.config.data.get('segment_by_label', True),
                min_label_purity=self.config.data.get('min_label_purity', 0.80),
                min_segment_points=self.config.data.get('min_segment_points', self.config.data.min_points),
                drop_mixed_segments=self.config.data.get('drop_mixed_segments', True),
                keep_unlabeled_segments=self.config.data.get('keep_unlabeled_segments', False),
                label_names=self.config.label_names,
            )

            # 只加载有标签的用户
            include_real_unlabeled = bool(
                getattr(self.config.data, 'include_real_unlabeled_in_train', False)
            )
            effective_only_labeled_users = (
                self.config.data.only_labeled_users or (not include_real_unlabeled)
            )
            raw_trajectories = self.data_loader.load_all_data(
                max_users=self.config.data.max_users,
                min_points=self.config.data.min_points,
                only_labeled_users=effective_only_labeled_users,
                require_valid_label=self.config.data.require_valid_label
            )

            if len(raw_trajectories) == 0:
                raise ValueError("❌ 未加载到任何轨迹数据！")

            self.logger.info(f"✅ 成功加载 {len(raw_trajectories)} 条原始轨迹")

        with Timer("数据预处理"):
            self.preprocessor = AdvancedTrajectoryPreprocessor(
                max_len=self.config.data.max_len
            )
            trajectories = self.preprocessor.process(raw_trajectories)

            labeled_trajectories = [
                t for t in trajectories
                if t['label'] is not None and t['label'] >= 0
            ]
            unlabeled_trajectories = [
                t for t in trajectories
                if t['label'] is None or t['label'] < 0
            ]
            self.logger.info(
                f"✅ 预处理后共有 {len(trajectories)} 条轨迹: "
                f"{len(labeled_trajectories)} 条有标签, {len(unlabeled_trajectories)} 条无标签"
            )

        # 当前默认使用 sample split；如需更严格的跨用户协议可切到 user_disjoint。
        split_mode = getattr(self.config.data, 'split_mode', 'sample')
        if split_mode == 'sample':
            self.logger.info('当前使用 sample-wise split（同一用户轨迹可同时出现在训练/验证中）')
            train_labeled, val_trajectories = self._split_dataset(labeled_trajectories)
        else:
            train_labeled, val_trajectories = self._split_dataset_stratified(labeled_trajectories)
        include_real_unlabeled = bool(
            getattr(self.config.data, 'include_real_unlabeled_in_train', False)
        )
        if include_real_unlabeled:
            train_trajectories = train_labeled + unlabeled_trajectories
            self.logger.info(
                f"   协议: real-world SSL | 训练集中额外并入 {len(unlabeled_trajectories)} 条真实无标签轨迹"
            )
        else:
            train_trajectories = train_labeled
            self.logger.info(
                "   协议: benchmark SSL | 仅使用原始有标签样本，并通过 label masking 构造未标注样本"
            )

        # 在训练集上 fit，再分别 transform
        with Timer("特征标准化"):
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()

            train_feats = np.array([t['features'] for t in train_trajectories])
            val_feats = np.array([t['features'] for t in val_trajectories])

            self.scaler.fit(train_feats)
            train_feats_scaled = self.scaler.transform(train_feats)
            val_feats_scaled = self.scaler.transform(val_feats)

            for i, t in enumerate(train_trajectories):
                t['features'] = train_feats_scaled[i]
            for i, t in enumerate(val_trajectories):
                t['features'] = val_feats_scaled[i]

        # 保存完整标签；真实无标签用户记为 -1，避免误入监督评估
        self.train_full_labels = np.array([
            t['label'] if t['label'] is not None and t['label'] >= 0 else -1
            for t in train_trajectories
        ])
        self.val_full_labels = np.array([t['label'] for t in val_trajectories])

        self.logger.info(
            f"   训练集完整标签: {int((self.train_full_labels >= 0).sum())} 条已知 / {len(self.train_full_labels)} 条总样本"
        )
        self.logger.info(f"   验证集完整标签: {len(self.val_full_labels)} 条")

        # 3. 应用标签掩码(仅对原本有标签的训练样本生效)
        import copy
        train_trajectories_masked = self._apply_label_masking(
            copy.deepcopy(train_trajectories),
            label_ratio=self.config.data.labeled_ratio
        )

        # 创建数据集
        if self.config.experiment.use_augmentation:
            augmenter = MultiScaleAugmenter()
            self.train_dataset = TrajDataset(train_trajectories_masked, augment=True, augmenter=augmenter)
        else:
            self.train_dataset = TrajDataset(train_trajectories_masked, augment=False)

        self.val_dataset = TrajDataset(val_trajectories, augment=False)

        # 5. 验证数据一致性
        assert len(self.train_dataset) == len(self.train_full_labels), "训练集大小不匹配!"
        assert len(self.val_dataset) == len(self.val_full_labels), "验证集大小不匹配!"

        self.logger.info(f"   训练集: {len(self.train_dataset)} 条轨迹")
        self.logger.info(f"   验证集: {len(self.val_dataset)} 条轨迹")

        # 记录标签统计
        self._log_label_statistics(train_trajectories_masked, val_trajectories)
        self._save_processed_data(trajectories, self.scaler)

        return trajectories

    # main.py (TransportModeRecognitionPipeline)
    def _extract_projected_embeddings_from_dataset(self, dataset):
        loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )
        all_z = []
        with torch.no_grad():
            for batch in loader:
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                emb = self.encoder(features, coords, lengths)
                emb = self._apply_graph_aggregation(emb)
                z = self.projector(emb)
                all_z.append(z.cpu().numpy())

        return np.vstack(all_z)

    def _split_dataset_stratified(
            self,
            trajectories: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        User-disjoint 分层划分：按 user_id 分组后划分，避免用户泄漏。
        """
        from sklearn.model_selection import train_test_split
        from collections import defaultdict

        # 按 user_id 分组
        user_trajs = defaultdict(list)
        for t in trajectories:
            metadata = t.get('metadata') or {}
            uid = t.get('user_id') or metadata.get('user_id', 'unknown')
            user_trajs[uid].append(t)

        user_ids = list(user_trajs.keys())

        # 用每个用户的众数标签做分层依据
        def user_majority_label(uid):
            from collections import Counter
            lbls = [t['label'] for t in user_trajs[uid] if t['label'] is not None]
            return Counter(lbls).most_common(1)[0][0] if lbls else 0

        user_labels = [user_majority_label(u) for u in user_ids]

        train_users, val_users = train_test_split(
            user_ids,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_seed,
            stratify=user_labels
        )

        val_set = set(val_users)
        get_uid = lambda t: t.get('user_id') or t.get('metadata', {}).get('user_id', 'unknown')
        train_trajectories = [t for t in trajectories if get_uid(t) not in val_set]
        val_trajectories   = [t for t in trajectories if get_uid(t) in val_set]

        self.logger.info(
            f"   User-disjoint split: {len(train_users)} train users / "
            f"{len(val_users)} val users"
        )
        return train_trajectories, val_trajectories


    def _apply_labeled_ratio(self, trajectories: List[Dict]) -> List[Dict]:
        """
        控制标签数据比例

        Args:
            trajectories: 轨迹列表

        Returns:
            调整标签后的轨迹列表
        """
        labeled_ratio = self.config.data.labeled_ratio

        if labeled_ratio >= 1.0:
            # 保留所有标签
            return trajectories

        # 找出有标签的轨迹
        labeled_indices = [
            i for i, t in enumerate(trajectories)
            if t['label'] is not None and t['label'] >= 0
        ]

        total_labeled = len(labeled_indices)
        keep_labeled = int(total_labeled * labeled_ratio)

        if keep_labeled < total_labeled:
            # 随机选择要保留标签的轨迹（分层采样）
            from sklearn.model_selection import train_test_split
            from collections import Counter

            labels = [trajectories[i]['label'] for i in labeled_indices]
            label_counts = Counter(labels)

            # 检查是否可以分层
            min_samples = min(label_counts.values())

            if min_samples >= 2:
                # 分层采样
                keep_indices, drop_indices = train_test_split(
                    labeled_indices,
                    train_size=keep_labeled,
                    random_state=self.config.data.random_seed,
                    stratify=labels
                )
            else:
                # 随机采样
                np.random.seed(self.config.data.random_seed)
                keep_indices = np.random.choice(
                    labeled_indices,
                    size=keep_labeled,
                    replace=False
                ).tolist()
                drop_indices = list(set(labeled_indices) - set(keep_indices))

            # 移除不保留的标签
            for idx in drop_indices:
                trajectories[idx]['label'] = None

            self.logger.info(f"   ✅ 标签数据: {keep_labeled}/{total_labeled} "
                             f"({labeled_ratio:.1%})")

        return trajectories

    def _log_label_statistics(self, train_trajs, val_trajs):
        """记录标签统计（区分可见/隐藏标签）"""
        train_visible = [t['label'] for t in train_trajs if t['label'] >= 0]
        train_hidden = sum(1 for t in train_trajs if t['label'] < 0)
        val_labels = [t['label'] for t in val_trajs if t['label'] >= 0]

        from collections import Counter

        self.logger.info("\n📊 训练集标签统计:")
        self.logger.info(f"   可见标签: {len(train_visible)} 条")
        self.logger.info(f"   隐藏标签: {train_hidden} 条")

        train_dist = Counter(train_visible)
        for label_id, count in sorted(train_dist.items()):
            label_name = self.label_names.get(label_id, f"Label {label_id}")
            self.logger.info(f"   {label_name}: {count}")

        self.logger.info("\n📊 验证集标签统计:")
        val_dist = Counter(val_labels)
        for label_id, count in sorted(val_dist.items()):
            label_name = self.label_names.get(label_id, f"Label {label_id}")
            self.logger.info(f"   {label_name}: {count}")

    def _apply_label_masking(self, trajectories: List[Dict], label_ratio: float,
                             min_samples_per_class: int = None):
        """隐藏部分标签"""
        from collections import Counter

        if label_ratio >= 1.0:
            return trajectories

        if label_ratio >= 1.0:
            return trajectories

        masked_trajectories = [t.copy() for t in trajectories]
        if min_samples_per_class is None:
            min_samples_per_class = int(
                getattr(self.config.data, 'min_samples_per_class', 5)
            )
        labels = [t['label'] for t in masked_trajectories if t['label'] is not None and t['label'] >= 0]
        label_counts = Counter(labels)
        rng = np.random.default_rng(self.config.data.random_seed)

        self.logger.info(f"   📊 原始标签分布: {dict(label_counts)}")

        # 为每个类别保留指定比例的标签
        for label_id, count in label_counts.items():
            indices = [
                i for i, t in enumerate(masked_trajectories)
                if t['label'] is not None and t['label'] == label_id
            ]
            # 确保每类至少保留min_samples_per_class个样本
            n_keep = max(
                min_samples_per_class,  # 至少5个
                int(count * label_ratio)  # 或按比例
            )
            n_keep = min(n_keep, count)  # 不超过总数

            keep_indices = rng.choice(indices, size=n_keep, replace=False)

            # 隐藏未选中的标签
            for idx in indices:
                if idx not in keep_indices:
                    masked_trajectories[idx]['label'] = -1

        # 统计可见标签
        visible_labels = [t['label'] for t in masked_trajectories if t['label'] >= 0]
        self.logger.info(
            f"   🎯 可见标签: {len(visible_labels)}/{len(labels)} "
            f"({len(visible_labels) / len(labels):.1%})"
        )

        return masked_trajectories

    def _log_data_statistics(self, train_trajs, val_trajs):
        """记录数据统计信息"""
        train_labels = [t['label'] for t in train_trajs if t.get('label') is not None and t['label'] >= 0]
        val_labels = [t['label'] for t in val_trajs if t.get('label') is not None and t['label'] >= 0]

        from collections import Counter
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)

        # 使用logger的log_class_distribution方法
        train_dist_dict = {self.label_names.get(k, f'Label {k}'): v
                           for k, v in train_dist.items()}
        val_dist_dict = {self.label_names.get(k, f'Label {k}'): v
                         for k, v in val_dist.items()}

        self.logger.log_class_distribution(train_dist_dict, title="训练集类别分布")
        self.logger.log_class_distribution(val_dist_dict, title="验证集类别分布")

        for label_id, count in sorted(train_dist.items()):
            label_name = self.label_names.get(label_id, f"Label {label_id}")
            self.logger.info(f"   {label_name}: {count}")

        for label_id, count in sorted(val_dist.items()):
            label_name = self.label_names.get(label_id, f"Label {label_id}")
            self.logger.info(f"   {label_name}: {count}")

    def _build_ratio_aware_training_overrides(self, ratio: float = None) -> Dict:
        """按 labeled_ratio 生成训练期覆盖项，低标签率使用更保守的 SSL 日程。"""
        ratio = float(self.config.data.labeled_ratio if ratio is None else ratio)
        tcfg = self.config.training
        overrides = {
            'labeled_ratio': ratio,
            'prototype_per_class_map_active': dict(tcfg.get('prototype_per_class_map', {}) or {}),
            'use_ssl_pretrain': False,
        }

        low_cutoff = float(tcfg.get('low_ratio_cutoff', 0.10))
        mid_cutoff = float(tcfg.get('mid_ratio_cutoff', 0.20))

        if ratio <= low_cutoff:
            overrides.update({
                'pseudo_warmup_epochs': int(tcfg.get('low_ratio_pseudo_warmup_epochs', tcfg.get('pseudo_warmup_epochs', 10))),
                'pseudo_label_interval': int(tcfg.get('low_ratio_pseudo_label_interval', tcfg.get('pseudo_label_interval', 5))),
                'pseudo_label_threshold': float(tcfg.get('low_ratio_pseudo_label_threshold', tcfg.get('pseudo_label_threshold', 0.88))),
                'pseudo_threshold_min': float(tcfg.get('low_ratio_pseudo_threshold_min', tcfg.get('pseudo_threshold_min', 0.86))),
                'pseudo_max_adoption_rate': float(tcfg.get('low_ratio_initial_pseudo_max_adoption_rate', tcfg.get('pseudo_max_adoption_rate', 0.25))),
                'pseudo_target_adoption_rate': float(tcfg.get('low_ratio_target_pseudo_max_adoption_rate', tcfg.get('pseudo_max_adoption_rate', 0.25))),
                'pseudo_cap_ramp_epochs': int(tcfg.get('low_ratio_cap_ramp_epochs', 30)),
                'pseudo_cap_ramp_min_quality': float(tcfg.get('low_ratio_cap_ramp_min_quality', 0.88)),
                'pseudo_cap_ramp_max_quality': float(tcfg.get('low_ratio_cap_ramp_max_quality', 0.94)),
                'pseudo_class_balance_min_count': int(tcfg.get('low_ratio_pseudo_class_balance_min_count', 0)),
                'hard_negative_weight_scale_low_ratio': float(tcfg.get('low_ratio_hard_negative_weight_scale', 0.35)),
                'coarse_aux_weight_scale_low_ratio': float(tcfg.get('low_ratio_coarse_aux_weight_scale', 0.50)),
                'low_ratio_aux_ramp_epochs': int(tcfg.get('low_ratio_aux_ramp_epochs', 35)),
                'prototype_per_class_map_active': dict(tcfg.get('prototype_per_class_map_low_ratio', tcfg.get('prototype_per_class_map', {})) or {}),
                'use_ssl_pretrain': bool(tcfg.get('use_ssl_pretrain', True)),
            })
        elif ratio <= mid_cutoff:
            overrides.update({
                'pseudo_warmup_epochs': int(tcfg.get('mid_ratio_pseudo_warmup_epochs', tcfg.get('pseudo_warmup_epochs', 10))),
                'pseudo_label_interval': int(tcfg.get('mid_ratio_pseudo_label_interval', tcfg.get('pseudo_label_interval', 5))),
                'pseudo_label_threshold': float(tcfg.get('mid_ratio_pseudo_label_threshold', tcfg.get('pseudo_label_threshold', 0.88))),
                'pseudo_threshold_min': float(tcfg.get('mid_ratio_pseudo_threshold_min', tcfg.get('pseudo_threshold_min', 0.86))),
                'pseudo_max_adoption_rate': float(tcfg.get('mid_ratio_initial_pseudo_max_adoption_rate', tcfg.get('pseudo_max_adoption_rate', 0.25))),
                'pseudo_target_adoption_rate': float(tcfg.get('mid_ratio_target_pseudo_max_adoption_rate', tcfg.get('pseudo_max_adoption_rate', 0.25))),
                'hard_negative_weight_scale_low_ratio': 1.0,
                'coarse_aux_weight_scale_low_ratio': 1.0,
                'low_ratio_aux_ramp_epochs': int(tcfg.get('low_ratio_aux_ramp_epochs', 35)),
            })
        else:
            overrides.update({
                'pseudo_label_threshold': float(tcfg.get('pseudo_label_threshold', 0.88)),
                'pseudo_threshold_min': float(tcfg.get('pseudo_threshold_min', 0.86)),
                'pseudo_target_adoption_rate': float(tcfg.get('pseudo_max_adoption_rate', 0.25)),
                'hard_negative_weight_scale_low_ratio': 1.0,
                'coarse_aux_weight_scale_low_ratio': 1.0,
                'low_ratio_aux_ramp_epochs': int(tcfg.get('low_ratio_aux_ramp_epochs', 35)),
            })

        self.logger.info(f"Ratio-aware training overrides (ratio={ratio:.2f}): {overrides}")
        return overrides

    def setup_model(self):
        """步骤2: 初始化模型"""
        self.logger.log_section("🤖 步骤2: 初始化模型")

        # 2.1 创建编码器
        self.encoder = AdaptiveTrajectoryEncoder(
            feat_dim=self.config.model.feat_dim,
            coord_dim=self.config.model.coord_dim,
            hidden_dim=self.config.model.hidden_dim,
            dropout=self.config.model.dropout,
            num_heads=self.config.model.get('num_attention_heads', 8),
            num_layers=self.config.model.get('num_encoder_layers', 3),
            encoder_mode=self.config.model.encoder_mode
        ).to(self.device)

        # 2.2 创建投影头
        self.projector = ProjectionHead(
            input_dim=self.config.model.hidden_dim,
            hidden_dim=self.config.model.hidden_dim,
            output_dim=self.config.model.projection_dim
        ).to(self.device)

        # 2.3 创建分类头（直接在 h 上分类，推理时主用）
        self.classifier = ClassifierHead(
            input_dim=self.config.model.hidden_dim,
            num_classes=self.config.experiment.num_classes,
            dropout=self.config.model.dropout
        ).to(self.device)

        if self.config.training.get('use_gnn_aggregation', True):
            self.graph_agg = GraphAggregationLayer(
                hidden_dim=self.config.model.hidden_dim,
                dropout=0.1
            ).to(self.device)
        else:
            self.graph_agg = None

        # 2.4 统计参数量
        from utils.helper import count_parameters

        encoder_params = count_parameters(self.encoder)
        projector_params = count_parameters(self.projector)
        classifier_params = count_parameters(self.classifier)
        graph_params = count_parameters(self.graph_agg)['total'] if self.graph_agg is not None else 0
        total_params = (
            encoder_params['total'] + projector_params['total'] + classifier_params['total'] + graph_params
        )

        self.logger.info(f"   编码器参数量: {encoder_params['total']:,}")
        self.logger.info(f"   投影头参数量: {projector_params['total']:,}")
        self.logger.info(f"   分类头参数量: {classifier_params['total']:,}")
        if self.graph_agg is not None:
            self.logger.info(f"   图聚合层参数量: {graph_params:,}")
        self.logger.info(f"   总参数量: {total_params:,}")

        return self.encoder, self.projector

    def train(self):
        """步骤3: 训练模型"""
        self.logger.log_section("🏋️  步骤3: 开始训练")

        ratio_overrides = self._build_ratio_aware_training_overrides()

        def tget(key, default=None):
            if key in ratio_overrides:
                return ratio_overrides[key]
            return self.config.training.get(key, default)

        labels = np.array([t['label'] for t in self.train_dataset.trajs])
        labeled_mask = labels >= 0

        unlabeled_weight = float(self.config.training.get('sampler_unlabeled_weight', 0.2))
        balance_power = float(self.config.training.get('sampler_class_balance_power', 0.5))
        min_class_weight = float(self.config.training.get('sampler_min_class_weight', 0.05))
        max_class_weight = float(self.config.training.get('sampler_max_class_weight', 4.0))
        hard_class_boost = dict(self.config.training.get('sampler_hard_class_boost', {}) or {})
        weights = np.full(len(labels), unlabeled_weight, dtype=np.float64)
        sampler_class_weights = {}
        use_weighted_sampler = bool(self.config.training.get('use_weighted_sampler', True))
        if use_weighted_sampler and labeled_mask.any():
            labeled_labels = labels[labeled_mask].astype(np.int64)
            class_counts = np.bincount(labeled_labels, minlength=self.config.experiment.num_classes)
            positive_counts = class_counts[class_counts > 0]
            max_count = float(positive_counts.max()) if positive_counts.size > 0 else 1.0
            for cls in np.unique(labeled_labels):
                cls = int(cls)
                count = max(int(class_counts[cls]), 1)
                cls_weight = (max_count / float(count)) ** balance_power
                cls_weight *= float(hard_class_boost.get(cls, 1.0))
                cls_weight = float(np.clip(cls_weight, min_class_weight, max_class_weight))
                sampler_class_weights[cls] = cls_weight
                weights[labels == cls] = cls_weight
            labeled_mean = float(weights[labeled_mask].mean())
            if (~labeled_mask).any():
                weights[~labeled_mask] = max(unlabeled_weight * labeled_mean, min_class_weight)
            unlabeled_display = float(weights[~labeled_mask][0]) if (~labeled_mask).any() else 0.0
            class_weight_str = ', '.join(
                f"{self.label_names.get(cls, cls)}:{sampler_class_weights[cls]:.3f}"
                for cls in sorted(sampler_class_weights)
            )
            self.logger.info(f"采样器类别权重: [{class_weight_str}] | unlabeled:{unlabeled_display:.3f}")
        else:
            self.logger.info("采样器: 使用普通随机打乱，不启用加权采样")

        data_seed = int(self.config.data.random_seed)
        sampler_generator = create_torch_generator(data_seed)
        loader_generator = create_torch_generator(data_seed + 1)
        val_generator = create_torch_generator(data_seed + 2)

        sampler = None
        shuffle = True
        if use_weighted_sampler:
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(self.train_dataset),
                replacement=True,
                generator=sampler_generator,
            )
            shuffle = False

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=traj_collate_fn,
            num_workers=self.config.experiment.get('num_workers', 0),
            pin_memory=True if self.device.type == 'cuda' else False,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=self.config.experiment.get('num_workers', 0),
            worker_init_fn=seed_worker,
            generator=val_generator,
        )

        # 创建伪标签生成器
        generator_name = self.config.training.pseudo_label_generator
        clean_baseline_mode = bool(tget('clean_supervised_baseline', self.config.training.clean_supervised_baseline))

        if generator_name == 'naive':
            if clean_baseline_mode:
                self.logger.info("Clean baseline: build naive pseudo generator for offline evaluation only")
            else:
                self.logger.info("Using Naive Pseudo-Label Generator (Fixed Threshold)")
            pseudo_label_gen = PseudoLabelGenerator(
                confidence_threshold=0.95,
                pseudo_temperature=tget('pseudo_confidence_temperature', 0.30),
                prototype_pooling=tget('prototype_pooling', 'max'),
                prototype_pool_temperature=tget('prototype_pool_temperature', 1.0),
            )
        elif generator_name == 'advanced_no_margin':
            if clean_baseline_mode:
                self.logger.info("Clean baseline: build advanced pseudo generator for offline evaluation only")
            else:
                self.logger.info("Using Advanced Generator (w/o Margin Filter)")
            pseudo_label_gen = AdvancedPseudoLabelGenerator(
                confidence_threshold=tget('pseudo_label_threshold', self.config.training.pseudo_label_threshold),
                progressive_threshold=tget('pseudo_progressive_threshold', False),
                consistency_check=True,
                top_k_consistency=3,
                margin_threshold=0.0,
                pseudo_temperature=tget('pseudo_confidence_temperature', 0.30),
                min_threshold=tget('pseudo_threshold_min', 0.65),
                threshold_decay=tget('pseudo_threshold_decay', 0.02),
                threshold_decay_interval=tget('pseudo_threshold_decay_interval', 10),
                low_margin_penalty=tget('pseudo_low_margin_penalty', 0.90),
                prototype_pooling=tget('prototype_pooling', 'max'),
                prototype_pool_temperature=tget('prototype_pool_temperature', 1.0),
                confidence_floor=tget('pseudo_confidence_floor', 0.0),
                distribution_alignment=tget('pseudo_distribution_alignment', False),
                distribution_momentum=tget('pseudo_distribution_momentum', 0.90),
                distribution_min_prob=tget('pseudo_distribution_min_prob', 1e-3),
                teacher_temperature=tget('pseudo_teacher_temperature', 1.0),
                proto_temperature=tget('pseudo_proto_temperature', 1.0),
                reliability_gate=tget('pseudo_reliability_gate', False),
                reliability_power=tget('pseudo_reliability_power', 1.0),
                reliability_floor=tget('pseudo_reliability_floor', 0.35),
                require_teacher_proto_agreement=tget('pseudo_require_teacher_proto_agreement', False),
            )
        else:
            teacher_clf_weight = float(tget('teacher_clf_pseudo_weight', 0.0))
            proto_weight = float(tget('proto_pseudo_weight', 1.0))
            use_graph_lp = bool(tget('use_graph_lp', False))
            if clean_baseline_mode:
                self.logger.info("Clean baseline: build advanced pseudo generator for offline evaluation only")
            elif teacher_clf_weight > 0.0 and proto_weight == 0.0 and not use_graph_lp:
                self.logger.info("Using conservative teacher-classifier pseudo labels")
            elif teacher_clf_weight > 0.0 and proto_weight > 0.0 and use_graph_lp:
                self.logger.info("Using full advanced pseudo-label fusion")
            else:
                self.logger.info("Using configurable advanced pseudo-label generator")
            pseudo_label_gen = AdvancedPseudoLabelGenerator(
                confidence_threshold=tget('pseudo_label_threshold', self.config.training.pseudo_label_threshold),
                progressive_threshold=tget('pseudo_progressive_threshold', False),
                consistency_check=True,
                top_k_consistency=3,
                margin_threshold=tget('proto_margin', 0.10),
                per_class_thresholds=tget('pseudo_per_class_thresholds', getattr(self.config.training, 'pseudo_per_class_thresholds', {})),
                per_class_margin=tget('pseudo_per_class_margin', getattr(self.config.training, 'pseudo_per_class_margin', {})),
                pseudo_temperature=tget('pseudo_confidence_temperature', 0.30),
                min_threshold=tget('pseudo_threshold_min', 0.65),
                threshold_decay=tget('pseudo_threshold_decay', 0.02),
                threshold_decay_interval=tget('pseudo_threshold_decay_interval', 10),
                low_margin_penalty=tget('pseudo_low_margin_penalty', 0.90),
                teacher_clf_weight=teacher_clf_weight,
                proto_weight=proto_weight,
                prototype_pooling=tget('prototype_pooling', 'max'),
                prototype_pool_temperature=tget('prototype_pool_temperature', 1.0),
                confidence_floor=tget('pseudo_confidence_floor', 0.0),
                distribution_alignment=tget('pseudo_distribution_alignment', False),
                distribution_momentum=tget('pseudo_distribution_momentum', 0.90),
                distribution_min_prob=tget('pseudo_distribution_min_prob', 1e-3),
                teacher_temperature=tget('pseudo_teacher_temperature', 1.0),
                proto_temperature=tget('pseudo_proto_temperature', 1.0),
                reliability_gate=tget('pseudo_reliability_gate', False),
                reliability_power=tget('pseudo_reliability_power', 1.0),
                reliability_floor=tget('pseudo_reliability_floor', 0.35),
                require_teacher_proto_agreement=tget('pseudo_require_teacher_proto_agreement', False),
            )

        config_dict = {
            'num_classes': self.config.experiment.num_classes,
            'label_names': dict(self.label_names),
            'projection_dim': self.config.model.projection_dim,
            'temperature': self.config.training.temperature,
            'use_contrastive': tget('use_contrastive', self.config.training.use_contrastive),
            'use_proto': tget('use_proto', self.config.training.use_proto),
            'contrast_weight': tget('contrast_weight', 1.0),
            'proto_weight': tget('proto_weight', self.config.training.proto_weight),
            'proto_weight_final': tget('proto_weight_final', self.config.training.proto_weight),
            'pseudo_weight': tget('pseudo_weight', self.config.training.pseudo_weight),
            'consistency_weight': tget('consistency_weight', self.config.training.consistency_weight),
            'optimizer': 'adamw',
            'lr': self.config.training.lr,
            'weight_decay': self.config.training.weight_decay,
            'scheduler': 'cosine',
            'warmup_epochs': 5,
            'epochs': self.config.training.epochs,
            'use_amp': True,
            'max_grad_norm': 1.0,
            'patience': self.config.training.patience,
            'save_interval': 10,
            'feat_dim': self.config.model.feat_dim,
            'coord_dim': self.config.model.coord_dim,
            'hidden_dim': self.config.model.hidden_dim,
            'dropout': self.config.model.dropout,
            'ema_decay': 0.999,
            'mask_ratio': 0.1,
            'random_seed': self.config.data.random_seed,
            'labeled_ratio': self.config.data.labeled_ratio,
            'low_ratio_cutoff': tget('low_ratio_cutoff', 0.10),
            'mid_ratio_cutoff': tget('mid_ratio_cutoff', 0.30),
            'pseudo_label_interval': tget('pseudo_label_interval', self.config.training.pseudo_label_interval),
            'pseudo_warmup_epochs': tget('pseudo_warmup_epochs', self.config.training.pseudo_warmup_epochs),
            'pseudo_ramp_epochs': tget('pseudo_ramp_epochs', self.config.training.pseudo_ramp_epochs),
            'consistency_ramp_epochs': tget('consistency_ramp_epochs', self.config.training.consistency_ramp_epochs),
            'proto_ema': tget('proto_ema', self.config.training.proto_ema),
            'pseudo_class_balance_power': tget('pseudo_class_balance_power', 0.5),
            'pseudo_class_balance_min_count': tget('pseudo_class_balance_min_count', 0),
            'pseudo_adaptive_per_class': tget('pseudo_adaptive_per_class', False),
            'pseudo_rare_class_gamma': tget('pseudo_rare_class_gamma', 0.0),
            'pseudo_margin_relax_gamma': tget('pseudo_margin_relax_gamma', 0.0),
            'pseudo_confidence_floor': tget('pseudo_confidence_floor', 0.0),
            'pseudo_quality_momentum': tget('pseudo_quality_momentum', 0.7),
            'pseudo_quality_relax_threshold': tget('pseudo_quality_relax_threshold', 0.93),
            'pseudo_quality_strict_threshold': tget('pseudo_quality_strict_threshold', 0.85),
            'pseudo_quality_strict_scale': tget('pseudo_quality_strict_scale', 0.20),
            'pseudo_hard_class_extra_strictness': tget('pseudo_hard_class_extra_strictness', {}),
            'pseudo_distribution_alignment': tget('pseudo_distribution_alignment', False),
            'pseudo_distribution_momentum': tget('pseudo_distribution_momentum', 0.90),
            'pseudo_distribution_min_prob': tget('pseudo_distribution_min_prob', 1e-3),
            'pseudo_teacher_temperature': tget('pseudo_teacher_temperature', 1.0),
            'pseudo_proto_temperature': tget('pseudo_proto_temperature', 1.0),
            'pseudo_reliability_gate': tget('pseudo_reliability_gate', False),
            'pseudo_reliability_power': tget('pseudo_reliability_power', 1.0),
            'pseudo_reliability_floor': tget('pseudo_reliability_floor', 0.35),
            'pseudo_reliability_warmup_epochs': tget('pseudo_reliability_warmup_epochs', 10),
            'proto_ema_conf_thr': tget('proto_ema_conf_thr', self.config.training.proto_ema_conf_thr),
            'pseudo_max_adoption_rate': tget('pseudo_max_adoption_rate', 1.0),
            'pseudo_target_adoption_rate': tget('pseudo_target_adoption_rate', tget('pseudo_max_adoption_rate', 1.0)),
            'pseudo_cap_ramp_epochs': tget('pseudo_cap_ramp_epochs', 0),
            'pseudo_cap_ramp_min_quality': tget('pseudo_cap_ramp_min_quality', 0.88),
            'pseudo_cap_ramp_max_quality': tget('pseudo_cap_ramp_max_quality', 0.94),
            'pseudo_max_adoption_count': tget('pseudo_max_adoption_count', 0),
            'pseudo_lp_threshold': tget('pseudo_lp_threshold', 0.92),
            'pseudo_lp_agree_threshold_offset': tget('pseudo_lp_agree_threshold_offset', 0.03),
            'pseudo_lp_conf_power': tget('pseudo_lp_conf_power', 0.75),
            'pseudo_lp_min_purity': tget('pseudo_lp_min_purity', 0.65),
            'pseudo_conflict_threshold': tget('pseudo_conflict_threshold', 0.95),
            'pseudo_conflict_margin': tget('pseudo_conflict_margin', 0.15),
            'pseudo_allow_lp_only': tget('pseudo_allow_lp_only', False),
            'pseudo_lp_only_warmup_epochs': tget('pseudo_lp_only_warmup_epochs', 0),
            'pseudo_lp_max_adoption_rate': tget('pseudo_lp_max_adoption_rate', 0.0),
            'pseudo_lp_agree_bonus': tget('pseudo_lp_agree_bonus', 0.0),
            'lp_min_support': tget('lp_min_support', 0.60),
            'use_pseudo_for_proto_ema': tget('use_pseudo_for_proto_ema', False),
            'pseudo_proto_ema_agreement_only': tget('pseudo_proto_ema_agreement_only', True),
            'use_teacher_clf_pseudo': tget('use_teacher_clf_pseudo', True),
            'teacher_clf_pseudo_weight': tget('teacher_clf_pseudo_weight', self.config.training.teacher_clf_pseudo_weight),
            'proto_pseudo_weight': tget('proto_pseudo_weight', self.config.training.proto_pseudo_weight),
            'pseudo_require_teacher_proto_agreement': tget('pseudo_require_teacher_proto_agreement', self.config.training.pseudo_require_teacher_proto_agreement),
            'pseudo_class_quota_per_update': tget('pseudo_class_quota_per_update', 0),
            'pseudo_dynamic_class_quota': tget('pseudo_dynamic_class_quota', self.config.training.pseudo_dynamic_class_quota),
            'pseudo_class_quota_quality_floor': tget('pseudo_class_quota_quality_floor', self.config.training.pseudo_class_quota_quality_floor),
            'pseudo_class_quota_bootstrap_per_class': tget('pseudo_class_quota_bootstrap_per_class', self.config.training.pseudo_class_quota_bootstrap_per_class),
            'pseudo_class_quota_quality_bins': tget('pseudo_class_quota_quality_bins', self.config.training.pseudo_class_quota_quality_bins),
            'pseudo_class_quota_bin_values': tget('pseudo_class_quota_bin_values', self.config.training.pseudo_class_quota_bin_values),
            'patience_after_pseudo': tget('patience_after_pseudo', 20),
            'use_gnn_aggregation': tget('use_gnn_aggregation', True),
            'use_graph_lp': tget('use_graph_lp', False),
            'pseudo_supervision_target': tget('pseudo_supervision_target', self.config.training.pseudo_supervision_target),
            'clean_supervised_baseline': tget('clean_supervised_baseline', self.config.training.clean_supervised_baseline),
            'sampler_hard_class_boost': tget('sampler_hard_class_boost', {}),
            'supcon_weight': tget('supcon_weight', self.config.training.supcon_weight),
            'supcon_temperature': tget('supcon_temperature', self.config.training.supcon_temperature),
        }

        config_dict.update({
            'proto_margin': tget('proto_margin', 0.0),
            'class_weights': tget('class_weights', None),
            'hard_negative_pairs': tget('hard_negative_pairs', []),
            'hard_negative_margin': tget('hard_negative_margin', 0.25),
            'hard_negative_weight': tget('hard_negative_weight', 0.0),
            'hard_negative_weight_scale_low_ratio': tget('hard_negative_weight_scale_low_ratio', 1.0),
            'coarse_groups': tget('coarse_groups', [[0, 1], [2, 3], [4]]),
            'coarse_aux_weight': tget('coarse_aux_weight', 0.0),
            'coarse_aux_weight_scale_low_ratio': tget('coarse_aux_weight_scale_low_ratio', 1.0),
            'low_ratio_aux_ramp_epochs': tget('low_ratio_aux_ramp_epochs', 35),
            'num_attention_heads': self.config.model.get('num_attention_heads', 8),
            'num_encoder_layers': self.config.model.get('num_encoder_layers', 3),
            'encoder_mode': self.config.model.encoder_mode,
            'graph_k': tget('graph_k', self.config.training.graph_k),
            'lambda_graph_smooth': tget('lambda_graph_smooth', self.config.training.lambda_graph_smooth),
            'lambda_graph_contrast': tget('lambda_graph_contrast', self.config.training.lambda_graph_contrast),
            'graph_build_interval': tget('graph_build_interval', self.config.training.graph_build_interval),
            'lp_alpha': tget('lp_alpha', self.config.training.lp_alpha),
            'lp_iters': tget('lp_iters', self.config.training.lp_iters),
            'classifier_weight': tget('classifier_weight', 1.0),
            'classifier_weight_final': tget('classifier_weight_final', tget('classifier_weight', 1.0)),
            'classifier_label_smoothing': tget('classifier_label_smoothing', 0.0),
            'prototypes_per_class': tget('prototypes_per_class', 1),
            'prototype_per_class_map': tget('prototype_per_class_map_active', tget('prototype_per_class_map', {})),
            'prototype_per_class_map_low_ratio': tget('prototype_per_class_map_low_ratio', {}),
            'prototype_stage_low_ratio_cutoff': tget('prototype_stage_low_ratio_cutoff', 0.10),
            'prototype_expand_epoch': tget('prototype_expand_epoch', 0),
            'prototype_expand_quality_thr': tget('prototype_expand_quality_thr', 0.92),
            'prototype_pooling': tget('prototype_pooling', 'max'),
            'prototype_pool_temperature': tget('prototype_pool_temperature', 1.0),
            'contrast_weight_final': tget('contrast_weight_final', 1.0),
            'graph_weight_final': tget('graph_weight_final', 1.0),
            'use_stagewise_loss_schedule': tget('use_stagewise_loss_schedule', False),
            'classification_stage_start': tget('classification_stage_start', 0),
            'classification_stage_ramp': tget('classification_stage_ramp', 1),
            'selection_metric': tget('selection_metric', 'macro_f1'),
            'use_ssl_pretrain': tget('use_ssl_pretrain', False),
            'pretrain_epochs': tget('pretrain_epochs', 0),
            'pretrain_lr': tget('pretrain_lr', self.config.training.lr),
            'pretrain_weight_decay': tget('pretrain_weight_decay', self.config.training.weight_decay),
            'pretrain_graph_smooth_weight': tget('pretrain_graph_smooth_weight', 0.0),
            'pretrain_graph_contrast_weight': tget('pretrain_graph_contrast_weight', 0.0),
        })

        # 根据 use_semi_supervised 开关决定是否启用 SSL 组件
        if not self.config.experiment.use_semi_supervised:
            config_dict['pseudo_weight'] = 0.0
            config_dict['consistency_weight'] = 0.0
            config_dict['lambda_graph_smooth'] = 0.0
            config_dict['lambda_graph_contrast'] = 0.0
            self.logger.info("   use_semi_supervised=False: 禁用伪标签/一致性/图损失")

        if bool(config_dict.get('clean_supervised_baseline', False)):
            config_dict['use_contrastive'] = False
            config_dict['use_proto'] = False
            config_dict['pseudo_weight'] = 0.0
            config_dict['consistency_weight'] = 0.0
            config_dict['pseudo_label_interval'] = 0
            config_dict['use_teacher_clf_pseudo'] = False
            config_dict['teacher_clf_pseudo_weight'] = 0.0
            config_dict['proto_pseudo_weight'] = 0.0
            config_dict['lambda_graph_smooth'] = 0.0
            config_dict['lambda_graph_contrast'] = 0.0
            config_dict['use_ssl_pretrain'] = False
            config_dict['use_gnn_aggregation'] = False
            config_dict['use_graph_lp'] = False
            config_dict['hard_negative_weight'] = 0.0
            config_dict['coarse_aux_weight'] = 0.0
            config_dict['use_stagewise_loss_schedule'] = False
            config_dict['classifier_weight_final'] = config_dict.get('classifier_weight', 1.0)
            config_dict['prototypes_per_class'] = 1
            config_dict['prototype_per_class_map'] = {}
            config_dict['prototype_per_class_map_low_ratio'] = {}
            self.logger.info("   clean_supervised_baseline=True: 仅保留 encoder + classifier + class-weighted CE")

        # full-label setting 不再沿用重 SSL 约束，避免把监督上界压平
        if (self.config.data.labeled_ratio >= 1.0 and
                self.config.training.get('disable_ssl_for_full_label', True)):
            config_dict['pseudo_weight'] = self.config.training.get('full_label_pseudo_weight', 0.0)
            config_dict['consistency_weight'] = self.config.training.get('full_label_consistency_weight', 0.0)
            config_dict['lambda_graph_smooth'] = self.config.training.get('full_label_lambda_graph_smooth', 0.0)
            config_dict['lambda_graph_contrast'] = self.config.training.get('full_label_lambda_graph_contrast', 0.0)
            self.logger.info("   labeled_ratio=1.0: 自动切换到 full-label supervised mode")

        self.trainer = SemiSupervisedTrainer(
            encoder=self.encoder,
            projector=self.projector,
            classifier=self.classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_dict,
            device=self.device,
            experiment_dir=str(self.exp_dir),
            pseudo_label_generator=pseudo_label_gen,
            graph_agg=self.graph_agg
        )
        self.trainer.hidden_true_labels = self.train_full_labels.copy()

        # 开始训练
        with Timer("模型训练"):
            history = self.trainer.fit()  # 使用 fit() 方法

        # 保存训练历史
        self._save_training_history(history)

        # 使用新的可视化方法
        self.visualizer.plot_training_curves(
            train_losses=history.get('train_loss', []),
            val_losses=history.get('val_loss'),
            contrast_losses=history.get('contrast_loss'),
            proto_losses=history.get('proto_loss'),
            pseudo_losses=history.get('pseudo_loss'),
            filename='training_curves.png'
        )

        if 'metrics' in history:
            self.visualizer.plot_metrics(
                metrics=history['metrics'],
                filename='metrics_trends.png'
            )

        return history


    def evaluate(self, checkpoint_path=None):
        """统一使用投影空间z进行所有评估"""
        self.logger.log_section("📊 步骤4: 模型评估（分类头 + 投影空间）")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projector.load_state_dict(checkpoint['projector_state_dict'])
        if 'classifier_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        if self.graph_agg is not None and 'graph_agg_state_dict' in checkpoint:
            self.graph_agg.load_state_dict(checkpoint['graph_agg_state_dict'])
        prototypes = checkpoint.get('prototypes')
        # 记录 checkpoint 对应的 epoch，供伪标签评估用动态阈值
        self._best_ckpt_epoch = int(checkpoint.get('epoch', 0))
        self._eval_trainer_config = checkpoint.get('config', {}) or {}
        self._eval_history = checkpoint.get('history', {}) or {}

        self.encoder.eval()
        self.projector.eval()
        self.classifier.eval()
        if self.graph_agg is not None:
            self.graph_agg.eval()

        # 提取编码器输出 h（用于 classifier head）和投影空间 z
        with Timer("提取embeddings"):
            h_train, z_train = self._extract_both_embeddings(self.train_dataset)
            h_val, z_val = self._extract_both_embeddings(self.val_dataset)
            z_all = np.vstack([z_train, z_val])
            all_true_labels = np.hstack([
                self.train_full_labels,
                self.val_full_labels
            ])

        # === 0. Classifier Head 评估（主指标）===
        with Timer("分类头评估"):
            clf_preds = self._predict_with_classifier(h_val)
            mode_evaluator = TransportModeEvaluator(
                label_names=self.label_names,
                save_dir=str(self.exp_dir / 'results')
            )
            clf_results = mode_evaluator.evaluate(
                pred_labels=clf_preds,
                true_labels=self.val_full_labels,
                save_prefix='clf_',
                return_detailed=False,
                selected_metrics=['accuracy', 'macro_f1', 'balanced_accuracy']
            )
            self.logger.info(
                "[FINAL] Classifier Accuracy: {:.4f} | Macro F1: {:.4f} | BalAcc: {:.4f}".format(
                    clf_results['summary']['accuracy'],
                    clf_results['summary']['macro_f1'],
                    clf_results['summary']['balanced_accuracy']
                )
            )

        # === 1. 原型最近邻评估（辅助参考）===
        with Timer("原型分类评估"):
            if prototypes is not None:
                proto_preds = self._predict_with_prototypes(z_val, prototypes)
                proto_results = mode_evaluator.evaluate(
                    pred_labels=proto_preds,
                    true_labels=self.val_full_labels,
                    save_prefix='proto_',
                    return_detailed=False,
                    selected_metrics=['accuracy', 'macro_f1', 'balanced_accuracy']
                )
                self.logger.info(
                    "[REF]   Proto   Accuracy: {:.4f} | Macro F1: {:.4f} | BalAcc: {:.4f}".format(
                        proto_results['summary']['accuracy'],
                        proto_results['summary']['macro_f1'],
                        proto_results['summary']['balanced_accuracy']
                    )
                )
            else:
                proto_results = {}

        # === 2. KNN评估 ===
        with Timer("KNN评估"):
            knn_results = self._evaluate_knn(
                z_train, self.train_full_labels,
                z_val, self.val_full_labels,
                k=5
            )

        # === 3. 线性探测 ===
        with Timer("线性探测"):
            lp_results = self._evaluate_linear_probe(
                z_train, self.train_full_labels,
                z_val, self.val_full_labels
            )

        # === 4. 聚类分析 ===
        with Timer("聚类分析"):
            clusters, refined_clusters = perform_clustering(
                z_all,
                n_clusters=self.config.experiment.num_clusters,
                method='kmeans',
                refine=True,
                logger=self.logger
            )

            # 簇到标签映射
            mapper = ClusterLabelMapper(num_classes=self.config.experiment.num_classes)
            known_mask = all_true_labels >= 0
            mapped_labels, cluster_to_label_map = mapper.map_clusters_to_labels(
                refined_clusters[known_mask], all_true_labels[known_mask]
            )

            # 聚类评估
            cluster_evaluator = ClusteringEvaluator(
                save_dir=str(self.exp_dir / 'results')
            )
            known_mask = all_true_labels >= 0
            cluster_results = cluster_evaluator.evaluate(
                z_all[known_mask], refined_clusters[known_mask], all_true_labels[known_mask]
            )

        # === 5. 伪标签质量评估===
        with Timer("伪标签质量评估"):
            pseudo_eval_results = self._evaluate_pseudo_labels_correct(
                h_train,
                z_train,
                self.train_dataset,  # 传入dataset以获取观测标签
                self.train_full_labels,
                prototypes
            )

        # 保存结果...
        self._save_evaluation_results(
            cluster_results=cluster_results,
            mode_results=clf_results,       # 主指标：classifier head
            cluster_to_label_map=cluster_to_label_map,
            knn_results=knn_results,
            linear_probe_results=lp_results,
            pseudo_eval_results=pseudo_eval_results
        )

        return {
            'proto_results': proto_results,
            'knn_results': knn_results,
            'linear_probe_results': lp_results,
            'cluster_results': cluster_results
        }

    def _predict_with_prototypes(
            self,
            embeddings: np.ndarray,
            prototypes: torch.Tensor
    ) -> np.ndarray:
        """用原型进行预测"""
        import torch.nn.functional as F

        z = torch.from_numpy(embeddings).to(self.device)
        protos = prototypes.to(self.device)

        logits = compute_prototype_logits(
            z,
            protos,
            temperature=1.0,
            aggregation=self.config.training.get('prototype_pooling', 'max'),
            pool_temperature=self.config.training.get('prototype_pool_temperature', 1.0),
        )
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        return preds

    def _evaluate_predictions(
            self,
            predictions: np.ndarray,
            true_labels: np.ndarray,
            prefix: str = ''
    ) -> Dict:
        """评估预测结果"""
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score,
            recall_score, classification_report
        )

        results = {
            'accuracy': float(accuracy_score(true_labels, predictions)),
            'macro_f1': float(f1_score(true_labels, predictions, average='macro')),
            'weighted_f1': float(f1_score(true_labels, predictions, average='weighted')),
            'macro_precision': float(precision_score(true_labels, predictions, average='macro')),
            'macro_recall': float(recall_score(true_labels, predictions, average='macro')),
            'report': classification_report(
                true_labels, predictions,
                target_names=[self.label_names[i] for i in range(self.config.experiment.num_classes)],
                output_dict=True
            )
        }

        self.logger.info(
            f"{prefix}Accuracy: {results['accuracy']:.4f}, "
            f"Macro F1: {results['macro_f1']:.4f}"
        )

        return results

    def _build_eval_batch_adj(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.graph_agg is None or embeddings.size(0) <= 1:
            return torch.zeros(
                embeddings.size(0),
                embeddings.size(0),
                device=embeddings.device,
                dtype=embeddings.dtype,
            )

        import torch.nn.functional as F

        z = F.normalize(embeddings.detach(), dim=1)
        sim = torch.matmul(z, z.t())
        sim.fill_diagonal_(0.0)
        k = min(int(self.config.training.graph_k), embeddings.size(0) - 1)
        topk = sim.topk(k, dim=1).indices
        adj = torch.zeros(embeddings.size(0), embeddings.size(0), device=embeddings.device)
        adj.scatter_(1, topk, torch.ones_like(topk, dtype=adj.dtype))
        return torch.maximum(adj, adj.t())

    def _apply_graph_aggregation(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.graph_agg is None:
            return embeddings
        adj = self._build_eval_batch_adj(embeddings)
        return self.graph_agg(embeddings, adj)

    def _extract_both_embeddings(self, dataset) -> tuple:
        """提取 encoder h 和 projector z 两种嵌入"""
        from data.dataset import traj_collate_fn
        loader = DataLoader(dataset, batch_size=256, shuffle=False,
                            collate_fn=traj_collate_fn, num_workers=0)
        all_h, all_z = [], []
        with torch.no_grad():
            for batch in loader:
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                h = self.encoder(features, coords, lengths)
                h = self._apply_graph_aggregation(h)
                z = self.projector(h)
                import torch.nn.functional as F
                z = F.normalize(z, dim=1)
                all_h.append(h.cpu().numpy())
                all_z.append(z.cpu().numpy())
        return np.vstack(all_h), np.vstack(all_z)

    def _predict_with_classifier(self, h: np.ndarray) -> np.ndarray:
        """用 classifier head 预测"""
        h_t = torch.from_numpy(h).to(self.device)
        with torch.no_grad():
            logits = self.classifier(h_t)
        return torch.argmax(logits, dim=1).cpu().numpy()

    def _get_eval_training_cfg(self):
        cfg = getattr(self, '_eval_trainer_config', None)
        if isinstance(cfg, dict) and cfg:
            return cfg
        return self.config.training

    def _get_last_training_pseudo_cap_rate(self):
        history = getattr(self, '_eval_history', {}) or {}
        monitors = history.get('pseudo_monitor', []) if isinstance(history, dict) else []
        for item in reversed(monitors):
            if isinstance(item, dict) and item.get('adoption_rate') is not None:
                return float(item['adoption_rate'])
        return None

    def _evaluate_pseudo_labels_correct(
            self,
            h_train: np.ndarray,
            z_train: np.ndarray,
            train_dataset,
            train_true_labels: np.ndarray,
            prototypes: torch.Tensor
    ) -> Dict:
        """伪标签质量评估：复现训练时的当前伪标签路径。"""
        from training.pseudo_label import (
            PSEUDO_SOURCE_ABSTAIN,
            PSEUDO_SOURCE_AGREE,
            PSEUDO_SOURCE_BASE_CONFLICT,
            PSEUDO_SOURCE_BASE_ONLY,
            PSEUDO_SOURCE_LP_CONFLICT,
            PSEUDO_SOURCE_LP_ONLY,
            PSEUDO_SOURCE_OBSERVED,
            apply_pseudo_label_acceptance_cap,
            class_balanced_pseudo_cap,
            fuse_pseudo_labels,
            graph_label_propagation,
        )
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        train_observed_labels = np.array([t['label'] for t in train_dataset.trajs])
        unlabeled_mask = (train_observed_labels < 0) & (train_true_labels >= 0)
        eval_cfg = self._get_eval_training_cfg()

        if unlabeled_mask.sum() == 0:
            return {'message': 'No hidden labels to evaluate'}

        # Step1: 与训练阶段一致的伪标签生成器与有效训练配置
        gen = self._build_pseudo_label_generator(eval_cfg)
        eval_epoch = getattr(self, '_best_ckpt_epoch', None)
        teacher_logits = None
        if (
            self._cfg_get(eval_cfg, 'use_teacher_clf_pseudo', False)
            and self._cfg_get(eval_cfg, 'teacher_clf_pseudo_weight', 0.0) > 0.0
        ):
            with torch.no_grad():
                teacher_logits = self.classifier(
                    torch.from_numpy(h_train).to(self.device)
                ).cpu().numpy()

        base_labels, base_conf = gen.generate(
            projected_embeddings=z_train,
            labels=train_observed_labels,
            prototypes=prototypes,
            epoch=eval_epoch,
            teacher_clf_logits=teacher_logits,
        )

        use_graph_lp = bool(self._cfg_get(eval_cfg, 'use_graph_lp', False))
        eval_mode_name = 'teacher/prototype'
        if use_graph_lp:
            glp_labels, glp_conf = graph_label_propagation(
                projected_embeddings=z_train,
                labels=train_observed_labels,
                k=self._cfg_get(eval_cfg, 'graph_k', self.config.training.graph_k),
                alpha=self._cfg_get(eval_cfg, 'lp_alpha', self.config.training.lp_alpha),
                iters=self._cfg_get(eval_cfg, 'lp_iters', self.config.training.lp_iters),
                min_support=self._cfg_get(eval_cfg, 'lp_min_support', 0.60),
                conf_power=self._cfg_get(eval_cfg, 'pseudo_lp_conf_power', 0.75),
                min_purity=self._cfg_get(eval_cfg, 'pseudo_lp_min_purity', 0.0),
            )

            thr = gen._get_dynamic_threshold(eval_epoch) if eval_epoch is not None else self._cfg_get(eval_cfg, 'pseudo_label_threshold', 0.75)
            fused_labels, fused_conf, pseudo_sources = fuse_pseudo_labels(
                base_labels, base_conf, glp_labels, glp_conf,
                observed_labels=train_observed_labels,
                thr=thr,
                lp_thr=self._cfg_get(eval_cfg, 'pseudo_lp_threshold', 0.92),
                conflict_thr=self._cfg_get(eval_cfg, 'pseudo_conflict_threshold', 0.95),
                conflict_margin=self._cfg_get(eval_cfg, 'pseudo_conflict_margin', 0.15),
                allow_lp_only=self._cfg_get(eval_cfg, 'pseudo_allow_lp_only', False),
                lp_agree_bonus=self._cfg_get(eval_cfg, 'pseudo_lp_agree_bonus', 0.0),
                lp_agree_threshold_offset=self._cfg_get(eval_cfg, 'pseudo_lp_agree_threshold_offset', 0.03),
                return_sources=True,
            )
            eval_mode_name = 'teacher/prototype+graph_lp'
        else:
            fused_labels = base_labels.copy()
            fused_conf = base_conf.copy()
            pseudo_sources = np.full(len(fused_labels), PSEUDO_SOURCE_OBSERVED, dtype=np.int64)
            eval_unlabeled_mask = train_observed_labels < 0
            pseudo_sources[eval_unlabeled_mask] = PSEUDO_SOURCE_ABSTAIN
            pseudo_sources[eval_unlabeled_mask & (fused_labels >= 0)] = PSEUDO_SOURCE_BASE_ONLY
        effective_cap_rate = self._get_last_training_pseudo_cap_rate()
        if effective_cap_rate is None:
            effective_cap_rate = self._cfg_get(eval_cfg, 'pseudo_max_adoption_rate', 1.0)
        fused_labels, fused_conf, _ = apply_pseudo_label_acceptance_cap(
            fused_labels,
            fused_conf,
            observed_labels=train_observed_labels,
            max_rate=effective_cap_rate,
            max_count=self._cfg_get(eval_cfg, 'pseudo_max_adoption_count', 0),
            class_balance=bool(self._cfg_get(eval_cfg, 'pseudo_adaptive_per_class', False)),
            class_balance_power=float(self._cfg_get(eval_cfg, 'pseudo_class_balance_power', 0.5)),
            min_per_class=int(self._cfg_get(eval_cfg, 'pseudo_class_balance_min_count', 0)),
        )
        quota = int(self._cfg_get(eval_cfg, 'pseudo_class_quota_per_update', 0))
        if quota > 0:
            fused_labels, fused_conf = class_balanced_pseudo_cap(
                fused_labels,
                fused_conf,
                train_observed_labels,
                quota_per_class=quota,
            )
        pseudo_sources[(train_observed_labels < 0) & (fused_labels < 0)] = PSEUDO_SOURCE_ABSTAIN

        # 只评估被隐藏的样本
        pseudo_preds = fused_labels[unlabeled_mask]
        true_unlabeled = train_true_labels[unlabeled_mask]

        # 过滤掉 abstain（-1）的样本
        accepted = pseudo_preds >= 0
        abstain_rate = 1.0 - accepted.mean()

        if accepted.sum() == 0:
            return {'message': 'All pseudo labels abstained', 'abstain_rate': float(abstain_rate)}

        accepted_true = true_unlabeled[accepted]
        accepted_pred = pseudo_preds[accepted]
        accepted_sources = pseudo_sources[unlabeled_mask][accepted]
        accepted_conf = fused_conf[unlabeled_mask][accepted]

        acc = float(accuracy_score(accepted_true, accepted_pred))
        macro_f1 = float(f1_score(accepted_true, accepted_pred, average='macro', zero_division=0))
        report = classification_report(
            accepted_true, accepted_pred,
            labels=list(range(self.config.experiment.num_classes)),
            target_names=[self.label_names[i] for i in range(self.config.experiment.num_classes)],
            output_dict=True,
            zero_division=0,
        )

        per_class = {}
        source_name_map = {
            PSEUDO_SOURCE_AGREE: 'agree',
            PSEUDO_SOURCE_BASE_ONLY: 'base',
            PSEUDO_SOURCE_LP_ONLY: 'lp',
            PSEUDO_SOURCE_BASE_CONFLICT: 'base_conflict',
            PSEUDO_SOURCE_LP_CONFLICT: 'lp_conflict',
        }
        for cls_idx, cls_name in self.label_names.items():
            cls_mask = accepted_pred == cls_idx
            cls_sources = accepted_sources[cls_mask]
            per_class[cls_name] = {
                'accepted': int(cls_mask.sum()),
                'avg_confidence': float(accepted_conf[cls_mask].mean()) if cls_mask.any() else None,
                'accuracy': float((accepted_true[cls_mask] == cls_idx).mean()) if cls_mask.any() else None,
                'f1': float(report.get(cls_name, {}).get('f1-score', 0.0)),
                'sources': {name: int((cls_sources == code).sum()) for code, name in source_name_map.items()},
            }

        self.logger.info(
            f"伪标签质量({eval_mode_name}): 隐藏={unlabeled_mask.sum()}, "
            f"采纳={accepted.sum()}({1-abstain_rate:.1%}), "
            f"准确率={acc:.4f}, Macro F1={macro_f1:.4f}, eval_cap={effective_cap_rate:.4f}"
        )

        return {
            'num_hidden': int(unlabeled_mask.sum()),
            'num_accepted': int(accepted.sum()),
            'abstain_rate': float(abstain_rate),
            'effective_cap_rate': float(effective_cap_rate),
            'accuracy': acc,
            'macro_f1': macro_f1,
            'report': report,
            'per_class': per_class,
        }

    def _extract_embeddings_from_dataset(self, dataset):
        """从数据集中提取embeddings"""
        loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                labels = batch['labels'].cpu().numpy()

                emb = self.encoder(features, coords, lengths)
                emb = self._apply_graph_aggregation(emb)
                all_embeddings.append(emb.cpu().numpy())
                all_labels.append(labels)

        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.hstack(all_labels)

        return all_embeddings, all_labels

    def inference(self, data_path, checkpoint_path=None):
        """步骤5: 推理模式（对新轨迹进行交通方式预测）"""
        from datetime import datetime as dt
        import pandas as pd
        import joblib
        import torch.nn.functional as F
        from collections import Counter

        self.logger.log_section("🔮 步骤5: 推理模式")

        # 1) 加载模型与原型
        if checkpoint_path is None:
            checkpoint_path = Path(self.config.checkpoint_dir) / 'best_model.pth'
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"🔄 加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projector.load_state_dict(checkpoint['projector_state_dict'])
        if 'classifier_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        if self.graph_agg is not None and 'graph_agg_state_dict' in checkpoint:
            self.graph_agg.load_state_dict(checkpoint['graph_agg_state_dict'])
        prototypes = checkpoint.get('prototypes', None)
        if prototypes is not None:
            prototypes = prototypes.to(self.device)

        self.encoder.eval()
        self.projector.eval()
        self.classifier.eval()
        if self.graph_agg is not None:
            self.graph_agg.eval()

        # 2) 加载训练阶段的特征标准化器
        scaler_path = self.exp_dir / 'data' / 'scaler.pkl'
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"✅ 推理将使用训练时的特征标准化器: {scaler_path}")
            except Exception as e:
                self.logger.warning(f"加载标准化器失败，将跳过特征标准化: {e}")
                self.scaler = None
        else:
            self.logger.warning("未找到训练时的标准化器，将跳过特征标准化。")
            self.scaler = None

        # 3) 解析输入数据（支持 .plt / .csv / .json 或包含这些文件的目录）
        def _parse_plt_file(plt_path: Path) -> pd.DataFrame:
            # GeoLife .plt 格式（跳过前6行）
            df = pd.read_csv(
                plt_path,
                skiprows=6,
                header=None,
                names=['latitude', 'longitude', 'zero', 'altitude', 'days', 'date', 'time']
            )
            if len(df) == 0:
                return df
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            df = df.dropna(subset=['datetime']).reset_index(drop=True)
            df['timestamp'] = df['datetime'].astype(np.int64) // 10 ** 9
            return df[['latitude', 'longitude', 'timestamp', 'datetime']]

        def _parse_csv_file(csv_path: Path) -> pd.DataFrame:
            # 尝试常见列名
            df = pd.read_csv(csv_path)
            lat_cols = [c for c in df.columns if c.lower() in ('lat', 'latitude')]
            lon_cols = [c for c in df.columns if c.lower() in ('lon', 'longitude', 'lng')]
            ts_cols = [c for c in df.columns if c.lower() in ('timestamp', 'time', 'datetime', 'date')]
            if not lat_cols or not lon_cols:
                raise ValueError(f"CSV缺少经纬度列: {csv_path}")
            lat_c, lon_c = lat_cols[0], lon_cols[0]
            # 生成 datetime / timestamp
            if ts_cols:
                tcol = ts_cols[0]
                # 尝试解析
                if np.issubdtype(df[tcol].dtype, np.number):
                    ts = df[tcol].astype(np.int64)
                    dtm = pd.to_datetime(ts, unit='s', errors='coerce')
                else:
                    dtm = pd.to_datetime(df[tcol], errors='coerce')
                    ts = (dtm.astype(np.int64) // 10 ** 9)
            else:
                # 无时间列，构造伪时间（等间隔1秒）
                dtm = pd.Series(pd.to_datetime([dt.utcnow()] * len(df)))
                ts = pd.Series(np.arange(len(df), dtype=np.int64))
            out = pd.DataFrame({
                'latitude': df[lat_c].astype(float),
                'longitude': df[lon_c].astype(float),
                'timestamp': ts.values,
                'datetime': dtm.values
            })
            out = out.dropna(subset=['latitude', 'longitude', 'datetime']).reset_index(drop=True)
            return out

        def _parse_json_file(json_path: Path) -> pd.DataFrame:
            # 期望格式: [{"latitude":..,"longitude":..,"timestamp":..(秒) 或 "datetime":..ISO}, ...]
            import json as _json
            with open(json_path, 'r', encoding='utf-8') as f:
                arr = _json.load(f)
            if not isinstance(arr, list) or len(arr) == 0:
                raise ValueError(f"JSON格式不正确: {json_path}")
            lat = [];
            lon = [];
            ts = [];
            dtm = []
            for item in arr:
                lat.append(float(item.get('latitude') or item.get('lat')))
                lon.append(float(item.get('longitude') or item.get('lon') or item.get('lng')))
                if 'timestamp' in item:
                    t = int(item['timestamp'])
                    ts.append(t)
                    dtm.append(pd.to_datetime(t, unit='s', errors='coerce'))
                else:
                    d = item.get('datetime') or item.get('time') or item.get('date')
                    d = pd.to_datetime(d, errors='coerce')
                    dtm.append(d)
                    ts.append(int(d.value // 10 ** 9) if pd.notna(d) else 0)
            out = pd.DataFrame({
                'latitude': lat,
                'longitude': lon,
                'timestamp': ts,
                'datetime': dtm
            })
            out = out.dropna(subset=['latitude', 'longitude', 'datetime']).reset_index(drop=True)
            return out

        def _collect_raw_trajectories(path_str: str) -> List[Dict]:
            p = Path(path_str)
            files = []
            if p.is_dir():
                files += list(p.rglob('*.plt'))
                files += list(p.rglob('*.csv'))
                files += list(p.rglob('*.json'))
                if not files:
                    self.logger.warning(f"目录中未找到可识别的文件: {p}")
            else:
                files = [p]

            raw = []
            for f in files:
                try:
                    if f.suffix.lower() == '.plt':
                        df = _parse_plt_file(f)
                    elif f.suffix.lower() == '.csv':
                        df = _parse_csv_file(f)
                    elif f.suffix.lower() == '.json':
                        df = _parse_json_file(f)
                    else:
                        self.logger.warning(f"跳过未知文件类型: {f}")
                        continue

                    if len(df) < 2:
                        self.logger.warning(f"轨迹点过少，跳过: {f}")
                        continue

                    raw.append({
                        'user_id': 'inference',
                        'trajectory_id': f.stem,
                        'data': df[['latitude', 'longitude', 'timestamp', 'datetime']],
                        'label': None,
                        'mode_name': None
                    })
                except Exception as e:
                    self.logger.warning(f"解析失败，跳过 {f}: {e}")
            return raw

        self.logger.info(f"📂 加载数据: {data_path}")
        raw_trajectories = _collect_raw_trajectories(data_path)
        if not raw_trajectories:
            self.logger.warning("没有可用轨迹，结束推理。")
            return []

        # 4) 预处理并标准化特征
        self.logger.info(f"🔧 预处理 {len(raw_trajectories)} 条轨迹...")
        preprocessor = AdvancedTrajectoryPreprocessor(max_len=self.config.data.max_len)
        proc_trajs = preprocessor.process(raw_trajectories)

        if self.scaler is not None and len(proc_trajs) > 0:
            feats = np.vstack([t['features'] for t in proc_trajs])
            feats = self.scaler.transform(feats)
            for i, t in enumerate(proc_trajs):
                t['features'] = feats[i]

        # 5) 构造数据集与 DataLoader
        dataset = TrajDataset(proc_trajs, augment=False)
        loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )

        # 6) 推理：优先用 classifier head；仅在缺失时回退到 prototype
        results = []
        id_list = []
        with torch.no_grad():
            for batch in loader:
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                emb = self.encoder(features, coords, lengths)
                emb = self._apply_graph_aggregation(emb)
                if 'classifier_state_dict' in checkpoint:
                    logits = self.classifier(emb)
                else:
                    if prototypes is None:
                        raise RuntimeError('Checkpoint contains neither classifier nor prototypes for inference.')
                    z = self.projector(emb)
                    z = F.normalize(z, dim=1)
                    logits = compute_prototype_logits(
                        z,
                        prototypes,
                        temperature=1.0,
                        aggregation=self.config.training.get('prototype_pooling', 'max'),
                        pool_temperature=self.config.training.get('prototype_pool_temperature', 1.0),
                    )

                probs = F.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)

                for i in range(pred.size(0)):
                    idx = int(batch['indices'][i].item())
                    traj_meta = proc_trajs[idx].get('metadata', {})
                    traj_id = traj_meta.get('trajectory_id') or f"traj_{idx}"
                    id_list.append(traj_id)
                    pred_id = int(pred[i].item())
                    pred_name = self.label_names.get(pred_id, f"Label {pred_id}")
                    results.append({
                        'trajectory_id': traj_id,
                        'pred_label_id': pred_id,
                        'pred_label_name': pred_name,
                        'confidence': float(conf[i].item()),
                        'probs': probs[i].detach().cpu().tolist()
                    })

        # 7) 保存结果
        out_dir = self.exp_dir / 'inference'
        ensure_dir(out_dir)
        ts = dt.now().strftime("%Y%m%d_%H%M%S")
        out_json = out_dir / f'predictions_{ts}.json'
        out_csv = out_dir / f'predictions_{ts}.csv'

        try:
            import json as _json
            with open(out_json, 'w', encoding='utf-8') as f:
                _json.dump(results, f, ensure_ascii=False, indent=2)
            # 同步保存 CSV
            import pandas as _pd
            _pd.DataFrame([{
            'trajectory_id': r['trajectory_id'],
            'pred_label_id': r['pred_label_id'],
            'pred_label_name': r['pred_label_name'],
            'confidence': r['confidence']
            } for r in results]).to_csv(out_csv, index=False)
            self.logger.info(f"✅ 推理结果已保存: {out_json}")
            self.logger.info(f"✅ 推理结果(简表)已保存: {out_csv}")
        except Exception as e:
            self.logger.warning(f"保存推理结果失败: {e}")

        # 8) 简要统计
        pred_names = [r['pred_label_name'] for r in results]
        dist = Counter(pred_names)
        self.logger.info("📊 预测分布: " + ", ".join([f"{k}:{v}" for k, v in dist.items()]))

        self.logger.info("✅ 推理完成")
        return results

    def _extract_embeddings(self):
        """提取所有轨迹的embeddings"""
        # 创建完整数据集
        all_trajectories = self.train_dataset.trajs + self.val_dataset.trajs
        all_dataset = TrajDataset(all_trajectories, augment=False)

        all_loader = DataLoader(
            all_dataset,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in all_loader:
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                labels = batch['labels'].cpu().numpy()

                emb = self.encoder(features, coords, lengths)
                all_embeddings.append(emb.cpu().numpy())
                all_labels.append(labels)

        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.hstack(all_labels)

        return all_embeddings, all_labels

    def _cfg_get(self, cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _build_pseudo_label_generator(self, cfg=None):
        """构建与训练阶段一致的伪标签生成器（用于评估）"""
        cfg = self.config.training if cfg is None else cfg
        name = self._cfg_get(cfg, 'pseudo_label_generator', 'advanced')
        thr = self._cfg_get(cfg, 'pseudo_label_threshold', 0.85)
        common_kwargs = dict(
            confidence_threshold=thr,
            progressive_threshold=self._cfg_get(cfg, 'pseudo_progressive_threshold', False),
            consistency_check=True,
            top_k_consistency=3,
            pseudo_temperature=self._cfg_get(cfg, 'pseudo_confidence_temperature', 0.30),
            min_threshold=self._cfg_get(cfg, 'pseudo_threshold_min', 0.65),
            threshold_decay=self._cfg_get(cfg, 'pseudo_threshold_decay', 0.02),
            threshold_decay_interval=self._cfg_get(cfg, 'pseudo_threshold_decay_interval', 10),
            low_margin_penalty=self._cfg_get(cfg, 'pseudo_low_margin_penalty', 0.90),
            teacher_clf_weight=self._cfg_get(cfg, 'teacher_clf_pseudo_weight', self.config.training.teacher_clf_pseudo_weight),
            proto_weight=self._cfg_get(cfg, 'proto_pseudo_weight', self.config.training.proto_pseudo_weight),
            confidence_floor=self._cfg_get(cfg, 'pseudo_confidence_floor', 0.0),
            distribution_alignment=self._cfg_get(cfg, 'pseudo_distribution_alignment', False),
            distribution_momentum=self._cfg_get(cfg, 'pseudo_distribution_momentum', 0.90),
            distribution_min_prob=self._cfg_get(cfg, 'pseudo_distribution_min_prob', 1e-3),
            teacher_temperature=self._cfg_get(cfg, 'pseudo_teacher_temperature', 1.0),
            proto_temperature=self._cfg_get(cfg, 'pseudo_proto_temperature', 1.0),
            reliability_gate=self._cfg_get(cfg, 'pseudo_reliability_gate', False),
            reliability_power=self._cfg_get(cfg, 'pseudo_reliability_power', 1.0),
            reliability_floor=self._cfg_get(cfg, 'pseudo_reliability_floor', 0.35),
            require_teacher_proto_agreement=self._cfg_get(cfg, 'pseudo_require_teacher_proto_agreement', False),
        )
        if name == 'naive':
            return PseudoLabelGenerator(
                confidence_threshold=thr,
                pseudo_temperature=self._cfg_get(cfg, 'pseudo_confidence_temperature', 0.30)
            )
        elif name == 'advanced_no_margin':
            return AdvancedPseudoLabelGenerator(
                margin_threshold=0.0,
                **common_kwargs,
            )
        else:
            return AdvancedPseudoLabelGenerator(
                margin_threshold=self._cfg_get(cfg, 'proto_margin', 0.10),
                per_class_thresholds=self._cfg_get(cfg, 'pseudo_per_class_thresholds', getattr(self.config.training, 'pseudo_per_class_thresholds', {})),
                per_class_margin=self._cfg_get(cfg, 'pseudo_per_class_margin', getattr(self.config.training, 'pseudo_per_class_margin', {})),
                **common_kwargs,
            )

    def _evaluate_knn(self, z_train, y_train, z_val, y_val, k=5):
        """kNN 在投影空间的快速评估"""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        train_mask = y_train >= 0
        val_mask = y_val >= 0
        z_train = z_train[train_mask]
        y_train = y_train[train_mask]
        z_val = z_val[val_mask]
        y_val = y_val[val_mask]

        knn = KNeighborsClassifier(n_neighbors=min(k, len(z_train)))
        knn.fit(z_train, y_train)
        preds = knn.predict(z_val)

        return {
            'k': int(k),
            'acc': float(accuracy_score(y_val, preds)),
            'macro_f1': float(f1_score(y_val, preds, average='macro')),
            'report': classification_report(y_val, preds, output_dict=True)
        }

    def _evaluate_linear_probe(self, z_train, y_train, z_val, y_val):
        """线性探测（Logistic Regression）"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        train_mask = y_train >= 0
        val_mask = y_val >= 0
        z_train = z_train[train_mask]
        y_train = y_train[train_mask]
        z_val = z_val[val_mask]
        y_val = y_val[val_mask]

        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        clf.fit(z_train, y_train)
        preds = clf.predict(z_val)

        return {
            'acc': float(accuracy_score(y_val, preds)),
            'macro_f1': float(f1_score(y_val, preds, average='macro')),
            'report': classification_report(y_val, preds, output_dict=True)
        }

    def _compute_contrastive_metrics(self, z_all, y_all, sample_pairs=20000):
        """对比学习质量指标：alignment / uniformity（Wang & Isola, 2020）"""
        import numpy as np
        import torch
        import torch.nn.functional as F

        z = torch.from_numpy(z_all).to(self.device)
        z = F.normalize(z, dim=1)
        z_np = z.cpu().numpy()
        N = z_np.shape[0]

        rng = np.random.default_rng(self.config.data.random_seed)
        n_pairs = min(sample_pairs, max(1, N * 10))
        i1 = rng.integers(0, N, size=n_pairs)
        i2 = rng.integers(0, N, size=n_pairs)

        same = (y_all[i1] == y_all[i2])
        alignment = None
        if same.any():
            d_same = ((z_np[i1[same]] - z_np[i2[same]]) ** 2).sum(axis=1)
            alignment = float(d_same.mean())

        d_all = ((z_np[i1] - z_np[i2]) ** 2).sum(axis=1)
        uniformity = float(np.log(np.exp(-2.0 * d_all).mean() + 1e-12))

        return {'alignment': alignment, 'uniformity': uniformity}

    def _evaluate_pseudo_labels(self, train_projected_embeddings, train_observed_labels, train_true_labels, prototypes, epoch=None):
        """
        在训练集“被隐藏标签”的子集上评估伪标签质量。
        输入必须是投影空间 z。
        """
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

        gen = self._build_pseudo_label_generator()

        new_labels, confidences = gen.generate(
            projected_embeddings=train_projected_embeddings,
            labels=train_observed_labels.copy(),
            prototypes=prototypes,
            epoch=epoch  # 传入
        )

        unlabeled_mask = train_observed_labels < 0
        adopted_mask = (new_labels >= 0) & unlabeled_mask
        adopted_idx = np.where(adopted_mask)[0]
        total_unlabeled = int(unlabeled_mask.sum())
        adopted = int(adopted_idx.size)

        if adopted > 0:
            adopted_true = train_true_labels[adopted_idx]
            adopted_pred = new_labels[adopted_idx]
            acc = float(accuracy_score(adopted_true, adopted_pred))
            macro_f1 = float(f1_score(adopted_true, adopted_pred, average='macro'))
            report = classification_report(adopted_true, adopted_pred, output_dict=True)
            cm = confusion_matrix(adopted_true, adopted_pred).tolist()
            avg_conf = float(np.mean(confidences[adopted_idx]))
        else:
            acc = None;
            macro_f1 = None;
            report = {};
            cm = [];
            avg_conf = 0.0

        self.logger.info(
            f"伪标签评估: 采纳 {adopted}/{total_unlabeled} "
            f"(rate={adopted / max(1, total_unlabeled):.3f}), avg_conf={avg_conf:.3f}, "
            f"acc={'{:.3f}'.format(acc) if acc is not None else 'N/A'}"
        )

        return {
            'unlabeled_total': total_unlabeled,
            'adopted': adopted,
            'adoption_rate': adopted / max(1, total_unlabeled),
            'avg_confidence': avg_conf,
            'acc': acc,
            'macro_f1': macro_f1,
            'report': report,
            'confusion_matrix': cm
        }

    def _load_all_trajectories(self):
        """加载完整轨迹数据用于分析"""
        return self.train_dataset.trajs + self.val_dataset.trajs

    def _visualize_results(self, embeddings, clusters, mapped_labels,
                           true_labels, prototypes):
        """生成所有可视化结果 - 使用 UnifiedVisualizer"""
        self.logger.info("📈 生成可视化结果...")

        # 1. 聚类可视化
        self.visualizer.plot_embeddings_2d(
            embeddings,
            labels=clusters,
            label_names=None,
            method='tsne',
            title='轨迹聚类结果',
            filename='01_trajectory_clusters.png'
        )

        # 2. 交通方式对比
        self.visualizer.plot_embeddings_comparison(
            embeddings,
            pred_labels=mapped_labels,
            true_labels=true_labels,
            label_names=self.label_names,
            filename='02_transport_modes_comparison.png'
        )

        # 3. 分布统计
        self.visualizer.plot_cluster_distribution(
            clusters,
            true_labels=true_labels,
            label_names=self.label_names,
            filename='03_cluster_distribution.png'
        )

        # 4. 混淆矩阵（如果有标签数据）
        labeled_mask = true_labels >= 0
        if labeled_mask.sum() > 0:
            label_list = [self.label_names.get(i, f'Label {i}')
                          for i in range(self.config.experiment.num_classes)]

            self.visualizer.plot_confusion_matrix(
                true_labels[labeled_mask],
                mapped_labels[labeled_mask],
                label_names=label_list,
                normalize=False,
                filename='04_confusion_matrix.png'
            )

            self.visualizer.plot_confusion_matrix(
                true_labels[labeled_mask],
                mapped_labels[labeled_mask],
                label_names=label_list,
                normalize=True,
                filename='04_confusion_matrix_normalized.png'
            )

        # 5. 簇-标签热图
        if labeled_mask.sum() > 0:
            self.visualizer.plot_cluster_label_heatmap(
                clusters,
                true_labels,
                label_names=self.label_names,
                filename='05_cluster_label_heatmap.png'
            )

        self.logger.info("✅ 可视化完成")

    def _save_processed_data(self, trajectories, scaler):
        """保存预处理后的数据"""
        data_dir = self.exp_dir / 'data'
        ensure_dir(data_dir)

        # 保存scaler
        import joblib
        joblib.dump(scaler, data_dir / 'scaler.pkl')

        self.logger.info(f"✅ 预处理数据已保存: {data_dir}")

    def _save_training_history(self, history):
        """保存训练历史"""
        history_path = Path(self.config.result_dir) / 'training_history.json'
        history_path.parent.mkdir(exist_ok=True)

        with open(history_path, 'w') as f:
            json.dump(make_json_serializable(history), f, indent=4)

        self.logger.info(f"✅ 训练历史已保存: {history_path}")

    def _save_evaluation_results(self, cluster_results, mode_results,
                                 cluster_to_label_map, **extras):
        """保存评估结果（支持附加评估项）"""
        results = {
            'config': self.config.to_dict(),
            'cluster_results': make_json_serializable(cluster_results),
            'mode_results': make_json_serializable(mode_results),
            'cluster_to_label_map': {int(k): int(v) for k, v in cluster_to_label_map.items()},
            'label_names': self.label_names,
            'timestamp': datetime.now().isoformat()
        }
        # 合并附加结果
        for k, v in extras.items():
            results[k] = make_json_serializable(v)

        results_path = Path(self.config.result_dir) / 'evaluation_results.json'

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        self.logger.info(f"✅ 评估结果已保存: {results_path}")

    def run(self):
        """运行完整流水线"""
        try:
            if self.mode == 'train':
                # 训练模式
                self.setup_data()
                self.setup_model()
                self.train()
                # 传入最佳模型路径
                best_model_path = Path(self.checkpoint_dir) / 'best_model.pth'
                if best_model_path.exists():
                    self.evaluate(checkpoint_path=str(best_model_path))
                else:
                    self.logger.warning(f"⚠️  未找到最佳模型: {best_model_path}，跳过评估")

            elif self.mode == 'eval':
                    # 仅评估模式
                    self.setup_data()
                    self.setup_model()
                    # 需要从命令行参数或配置获取checkpoint路径
                    checkpoint_path = getattr(self, 'checkpoint_path', None)
                    if checkpoint_path and Path(checkpoint_path).exists():
                        self.evaluate(checkpoint_path=checkpoint_path)
                    else:
                        raise FileNotFoundError("评估模式需要指定有效的checkpoint路径")

            elif self.mode == 'inference':
                # 推理模式
                self.setup_model()
                self.inference(self.config.experiment.get('inference_data_path'))

            self.logger.info("\n" + "=" * 70)
            self.logger.info("🎉 任务完成！")
            self.logger.info("=" * 70)


        except Exception as e:

            self.logger.error(f"❌ 发生错误: {str(e)}")

            self.logger.log_exception(e, context="Pipeline执行")  # 添加这行

            raise

        finally:

            if hasattr(self, 'logger'):
                self.logger.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='轨迹交通方式识别系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    示例:
      # 训练模式
      python main.py --mode train --config config/default.yaml
    
      # 评估模式
      python main.py --mode eval --checkpoint experiments/exp_001/checkpoints/best_model.pth
    
      # 推理模式
      python main.py --mode inference --checkpoint best_model.pth --data path/to/new/data
            """
        )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'inference'],
        default='train',
        help='运行模式'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（YAML或JSON）'
    )

    parser.add_argument(
        '--exp-name',
        type=str,
        default=None,
        help='实验名称（默认使用时间戳）'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='检查点路径（用于评估或推理）'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='数据路径（用于推理）'
    )

    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='使用的GPU编号'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式'
    )

    parser.add_argument(
        '--labeled-ratio',
        type=float,
        default=None,
        help='标签数据比例 (0.0-1.0)，例如 0.3 表示使用30%%的标签'
    )

    parser.add_argument(
        '--use-semi-supervised',
        type=lambda x: x.lower() != 'false',
        default=None,
        help='是否启用半监督学习 (True/False)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子，覆盖 config.data.random_seed'
    )

    parser.add_argument(
        '--split-mode',
        type=str,
        choices=['user_disjoint', 'sample'],
        default=None,
        help='数据划分协议，sample 表示按样本划分，user_disjoint 表示按用户划分'
    )

    parser.add_argument(
        '--label-schema',
        type=str,
        choices=['ground5', 'geolife5', 'ground4'],
        default=None,
        help='标签体系：geolife5=walk/bike/bus/car/subway（train 并入 subway）；ground5=walk/bike/bus/driving/train'
    )

    parser.add_argument(
        '--training-profile',
        type=str,
        choices=[
            'baseline',
            'sampler_only',
            'hard_only',
            'coarse_only',
            'proto_only',
            'proto_multicenter',
            'proto_warmup',
            'proto_multicenter_warmup',
            'proto_warmup_p015',
            'proto_warmup_p025',
            'proto_warmup_early5',
            'proto_warmup_late15',
            'proto_warmup_carup_subwaydown',
            'proto_warmup_teacher_agree',
            'proto_warmup_teacher_agree_late30',
            'proto_warmup_teacher_agree_w010',
            'proto_warmup_teacher_agree_w010_dynquota',
            'proto_warmup_teacher_agree_w010_proto005',
            'proto_warmup_teacher_agree_w010_dynquota_proto005',
            'proto_warmup_teacher_agree_w010_supcon',
            'proto_warmup_teacher_agree_w010_supcon_lowctr',
            'proto_warmup_teacher_agree_w010_supcon_dynquota',
            'proto_warmup_teacher_agree_late30_w010',
            'proto_warmup_teacher_agree_late30_w010_protoaware',
            'proto_warmup_teacher_agree_lp',
            'proto_warmup_teacher_agree_gnn',
            'proto_warmup_teacher_agree_gnn_lp',
            'teacher_only',
            'proto_teacher',
            'proto_coarse',
            'proto_teacher_coarse',
            'supervised_plus',
            'repr',
            'teacher_pseudo',
        ],
        default=None,
        help='训练档位：baseline/sampler/hard/coarse/proto/teacher 为筛选档；proto_multicenter/proto_warmup/proto_warmup_* 为 proto 精修档；supervised_plus/repr/teacher_pseudo 为组合档'
    )

    parser.add_argument(
        '--clean-supervised-baseline',
        action='store_true',
        help='启用干净监督基线：关闭伪标签、图传播、consistency、hard negative、coarse aux，仅保留 encoder + classifier + class-weighted CE'
    )

    return parser.parse_args()


def apply_training_profile(config: Config, profile: str):
    profile = str(profile).lower()
    tcfg = config.training

    tcfg.clean_supervised_baseline = False
    tcfg.use_contrastive = False
    tcfg.use_proto = False
    tcfg.pseudo_weight = 0.0
    tcfg.supcon_weight = 0.0
    tcfg.consistency_weight = 0.0
    tcfg.pseudo_label_interval = 0
    tcfg.use_teacher_clf_pseudo = False
    tcfg.teacher_clf_pseudo_weight = 0.0
    tcfg.proto_pseudo_weight = 0.0
    tcfg.lambda_graph_smooth = 0.0
    tcfg.lambda_graph_contrast = 0.0
    tcfg.use_ssl_pretrain = False
    tcfg.use_gnn_aggregation = False
    tcfg.use_graph_lp = False
    tcfg.use_stagewise_loss_schedule = False
    tcfg.classifier_weight_final = tcfg.classifier_weight
    tcfg.prototypes_per_class = 1
    tcfg.prototype_per_class_map = {}
    tcfg.prototype_per_class_map_low_ratio = {}
    tcfg.use_weighted_sampler = True
    tcfg.pseudo_supervision_target = 'classifier'
    tcfg.pseudo_dynamic_class_quota = False
    tcfg.pseudo_reliability_gate = False
    config.experiment.use_semi_supervised = False
    tcfg.training_profile = profile

    def apply_w010_teacher_agree_base():
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
        tcfg.pseudo_weight = 0.10
        tcfg.pseudo_label_interval = 5
        tcfg.pseudo_warmup_epochs = 20
        tcfg.pseudo_label_threshold = 0.93
        tcfg.pseudo_threshold_min = 0.93
        tcfg.pseudo_progressive_threshold = False
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        tcfg.pseudo_require_teacher_proto_agreement = True
        tcfg.pseudo_class_quota_per_update = 6
        tcfg.pseudo_dynamic_class_quota = False
        tcfg.pseudo_class_quota_bootstrap_per_class = 2
        tcfg.pseudo_class_quota_quality_bins = [0.70, 0.80, 0.90]
        tcfg.pseudo_class_quota_bin_values = [0, 2, 4, 6]
        tcfg.pseudo_adaptive_per_class = False
        tcfg.pseudo_distribution_alignment = False
        tcfg.pseudo_reliability_gate = False
        config.experiment.use_semi_supervised = True

    if profile == 'baseline':
        tcfg.clean_supervised_baseline = True
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
    elif profile == 'sampler_only':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = True
    elif profile == 'hard_only':
        tcfg.hard_negative_weight = 0.12
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
    elif profile == 'coarse_only':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.10
        tcfg.use_weighted_sampler = False
    elif profile == 'proto_only':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.20
        tcfg.proto_weight_final = 0.20
    elif profile == 'proto_multicenter':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.20
        tcfg.proto_weight_final = 0.20
        tcfg.prototypes_per_class = 1
        tcfg.prototype_per_class_map = {2: 3, 4: 3}
    elif profile == 'proto_warmup':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
    elif profile == 'proto_warmup_p015':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.15
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
    elif profile == 'proto_warmup_p025':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.25
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
    elif profile == 'proto_warmup_early5':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 5
        tcfg.classification_stage_ramp = 15
    elif profile == 'proto_warmup_late15':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 15
        tcfg.classification_stage_ramp = 15
    elif profile == 'proto_warmup_carup_subwaydown':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
        tcfg.class_weights = [1.0, 1.1, 2.0, 1.3, 2.1]
    elif profile == 'proto_warmup_teacher_agree':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
        tcfg.pseudo_weight = 0.15
        tcfg.pseudo_label_interval = 3
        tcfg.pseudo_warmup_epochs = 20
        tcfg.pseudo_label_threshold = 0.93
        tcfg.pseudo_threshold_min = 0.93
        tcfg.pseudo_progressive_threshold = False
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        tcfg.pseudo_require_teacher_proto_agreement = True
        tcfg.pseudo_class_quota_per_update = 6
        tcfg.pseudo_adaptive_per_class = False
        tcfg.pseudo_distribution_alignment = False
        tcfg.pseudo_reliability_gate = False
        config.experiment.use_semi_supervised = True
    elif profile == 'proto_warmup_teacher_agree_late30':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
        tcfg.pseudo_weight = 0.15
        tcfg.pseudo_label_interval = 3
        tcfg.pseudo_warmup_epochs = 30
        tcfg.pseudo_label_threshold = 0.93
        tcfg.pseudo_threshold_min = 0.93
        tcfg.pseudo_progressive_threshold = False
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        tcfg.pseudo_require_teacher_proto_agreement = True
        tcfg.pseudo_class_quota_per_update = 6
        tcfg.pseudo_adaptive_per_class = False
        tcfg.pseudo_distribution_alignment = False
        tcfg.pseudo_reliability_gate = False
        config.experiment.use_semi_supervised = True
    elif profile == 'proto_warmup_teacher_agree_w010':
        apply_w010_teacher_agree_base()
    elif profile == 'proto_warmup_teacher_agree_w010_dynquota':
        apply_w010_teacher_agree_base()
        tcfg.pseudo_dynamic_class_quota = True
    elif profile == 'proto_warmup_teacher_agree_w010_proto005':
        apply_w010_teacher_agree_base()
        tcfg.proto_pseudo_weight = 0.05
    elif profile == 'proto_warmup_teacher_agree_w010_dynquota_proto005':
        apply_w010_teacher_agree_base()
        tcfg.pseudo_dynamic_class_quota = True
        tcfg.proto_pseudo_weight = 0.05
    elif profile == 'proto_warmup_teacher_agree_w010_supcon':
        apply_w010_teacher_agree_base()
        tcfg.supcon_weight = 0.05
    elif profile == 'proto_warmup_teacher_agree_w010_supcon_lowctr':
        apply_w010_teacher_agree_base()
        tcfg.supcon_weight = 0.05
        tcfg.contrast_weight_final = 0.15
    elif profile == 'proto_warmup_teacher_agree_w010_supcon_dynquota':
        apply_w010_teacher_agree_base()
        tcfg.supcon_weight = 0.05
        tcfg.contrast_weight_final = 0.15
        tcfg.pseudo_dynamic_class_quota = True
    elif profile == 'proto_warmup_teacher_agree_late30_w010':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
        tcfg.pseudo_weight = 0.10
        tcfg.pseudo_label_interval = 5
        tcfg.pseudo_warmup_epochs = 30
        tcfg.pseudo_label_threshold = 0.93
        tcfg.pseudo_threshold_min = 0.93
        tcfg.pseudo_progressive_threshold = False
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        tcfg.pseudo_require_teacher_proto_agreement = True
        tcfg.pseudo_class_quota_per_update = 6
        tcfg.pseudo_adaptive_per_class = False
        tcfg.pseudo_distribution_alignment = False
        tcfg.pseudo_reliability_gate = False
        config.experiment.use_semi_supervised = True
    elif profile == 'proto_warmup_teacher_agree_late30_w010_protoaware':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
        tcfg.pseudo_weight = 0.10
        tcfg.pseudo_label_interval = 5
        tcfg.pseudo_warmup_epochs = 30
        tcfg.pseudo_label_threshold = 0.93
        tcfg.pseudo_threshold_min = 0.93
        tcfg.pseudo_progressive_threshold = False
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.25
        tcfg.pseudo_require_teacher_proto_agreement = True
        tcfg.pseudo_class_quota_per_update = 6
        tcfg.pseudo_dynamic_class_quota = True
        tcfg.pseudo_adaptive_per_class = False
        tcfg.pseudo_distribution_alignment = False
        tcfg.pseudo_reliability_gate = True
        config.experiment.use_semi_supervised = True
    elif profile == 'proto_warmup_teacher_agree_lp':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
        tcfg.pseudo_weight = 0.15
        tcfg.pseudo_label_interval = 3
        tcfg.pseudo_warmup_epochs = 20
        tcfg.pseudo_label_threshold = 0.93
        tcfg.pseudo_threshold_min = 0.93
        tcfg.pseudo_progressive_threshold = False
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        tcfg.pseudo_require_teacher_proto_agreement = True
        tcfg.pseudo_class_quota_per_update = 6
        tcfg.pseudo_adaptive_per_class = False
        tcfg.pseudo_distribution_alignment = False
        tcfg.pseudo_reliability_gate = False
        tcfg.use_graph_lp = True
        config.experiment.use_semi_supervised = True
    elif profile == 'proto_warmup_teacher_agree_gnn':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
        tcfg.pseudo_weight = 0.15
        tcfg.pseudo_label_interval = 3
        tcfg.pseudo_warmup_epochs = 20
        tcfg.pseudo_label_threshold = 0.93
        tcfg.pseudo_threshold_min = 0.93
        tcfg.pseudo_progressive_threshold = False
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        tcfg.pseudo_require_teacher_proto_agreement = True
        tcfg.pseudo_class_quota_per_update = 6
        tcfg.pseudo_adaptive_per_class = False
        tcfg.pseudo_distribution_alignment = False
        tcfg.pseudo_reliability_gate = False
        tcfg.use_gnn_aggregation = True
        config.experiment.use_semi_supervised = True
    elif profile == 'proto_warmup_teacher_agree_gnn_lp':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
        tcfg.pseudo_weight = 0.15
        tcfg.pseudo_label_interval = 3
        tcfg.pseudo_warmup_epochs = 20
        tcfg.pseudo_label_threshold = 0.93
        tcfg.pseudo_threshold_min = 0.93
        tcfg.pseudo_progressive_threshold = False
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        tcfg.pseudo_require_teacher_proto_agreement = True
        tcfg.pseudo_class_quota_per_update = 6
        tcfg.pseudo_adaptive_per_class = False
        tcfg.pseudo_distribution_alignment = False
        tcfg.pseudo_reliability_gate = False
        tcfg.use_gnn_aggregation = True
        tcfg.use_graph_lp = True
        config.experiment.use_semi_supervised = True
    elif profile == 'proto_multicenter_warmup':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.0
        tcfg.proto_weight_final = 0.20
        tcfg.use_stagewise_loss_schedule = True
        tcfg.classification_stage_start = 10
        tcfg.classification_stage_ramp = 15
        tcfg.prototypes_per_class = 1
        tcfg.prototype_per_class_map = {2: 3, 4: 3}
    elif profile == 'teacher_only':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.pseudo_weight = 0.20
        tcfg.pseudo_label_interval = 3
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        config.experiment.use_semi_supervised = True
    elif profile == 'proto_teacher':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.0
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.20
        tcfg.proto_weight_final = 0.20
        tcfg.pseudo_weight = 0.20
        tcfg.pseudo_label_interval = 3
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        config.experiment.use_semi_supervised = True
    elif profile == 'proto_coarse':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.10
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.20
        tcfg.proto_weight_final = 0.20
    elif profile == 'proto_teacher_coarse':
        tcfg.hard_negative_weight = 0.0
        tcfg.coarse_aux_weight = 0.10
        tcfg.use_weighted_sampler = False
        tcfg.use_proto = True
        tcfg.proto_weight = 0.20
        tcfg.proto_weight_final = 0.20
        tcfg.pseudo_weight = 0.20
        tcfg.pseudo_label_interval = 3
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        config.experiment.use_semi_supervised = True
    elif profile == 'supervised_plus':
        tcfg.hard_negative_weight = 0.12
        tcfg.coarse_aux_weight = 0.10
    elif profile == 'repr':
        tcfg.hard_negative_weight = 0.12
        tcfg.coarse_aux_weight = 0.10
        tcfg.use_proto = True
        tcfg.proto_weight = 0.20
        tcfg.proto_weight_final = 0.20
    elif profile == 'teacher_pseudo':
        tcfg.hard_negative_weight = 0.12
        tcfg.coarse_aux_weight = 0.10
        tcfg.use_proto = True
        tcfg.proto_weight = 0.20
        tcfg.proto_weight_final = 0.20
        tcfg.pseudo_weight = 0.20
        tcfg.pseudo_label_interval = 3
        tcfg.use_teacher_clf_pseudo = True
        tcfg.teacher_clf_pseudo_weight = 1.0
        tcfg.proto_pseudo_weight = 0.0
        config.experiment.use_semi_supervised = True
    else:
        raise ValueError(f"Unknown training profile: {profile}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 加载配置
    config = Config(config_path=args.config)

    # 覆盖标签比例配置
    if args.labeled_ratio is not None:
        if 0.0 <= args.labeled_ratio <= 1.0:
            config.data.labeled_ratio = args.labeled_ratio
        else:
            raise ValueError("❌ labeled_ratio 必须在 0.0 到 1.0 之间")

    # 覆盖半监督开关
    if args.use_semi_supervised is not None:
        config.experiment.use_semi_supervised = args.use_semi_supervised

    if args.seed is not None:
        config.data.random_seed = int(args.seed)

    if args.split_mode is not None:
        config.data.split_mode = args.split_mode

    if args.label_schema is not None:
        config.data.label_schema = args.label_schema
        config._refresh_label_schema()

    if args.training_profile is not None:
        apply_training_profile(config, args.training_profile)

    if args.clean_supervised_baseline:
        apply_training_profile(config, 'baseline')

    # 覆盖配置
    if args.exp_name:
        config.experiment.exp_name = args.exp_name
        config._setup_paths()  # 重新设置路径

    if args.data:
        # 添加推理数据路径
        config.experiment.__dict__['inference_data_path'] = args.data

    if args.debug:
        config.training.epochs = 2
        config.data.max_users = 5
        config.training.batch_size = 8

    # 保存配置到实验目录
    config_save_path = Path(config.exp_dir) / 'config.yaml'
    config.save(str(config_save_path))

    # 打印配置信息
    print("\n" + "=" * 70)
    print("⚙️  配置信息")
    print("=" * 70)
    print(f"运行模式: {args.mode}")
    print(f"实验名称: {config.experiment.exp_name}")
    print(f"GPU设备: {args.gpu}")
    print(f"训练档位: {config.training.training_profile}")
    print(f"批次大小: {config.training.batch_size}")
    print(f"训练轮数: {config.training.epochs}")
    print(f"数据根目录: {config.data.data_root}")
    print(f"实验保存目录: {config.exp_dir}")
    print("=" * 70 + "\n")

    # 创建并运行流水线
    pipeline = TransportModeRecognitionPipeline(config, mode=args.mode)
    if args.checkpoint:
        pipeline.checkpoint_path = args.checkpoint
    pipeline.run()


if __name__ == '__main__':
    main()
