# main.py
"""
轨迹交通方式识别系统 - 主程序入口
支持训练、评估、推理等多种模式
"""

import os
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime

import torch

# 导入项目模块
from geolife.graph.config.config import Config
from data.loader import ImprovedGeoLifeDataLoader
from data.preprocessor import AdvancedTrajectoryPreprocessor
from data.dataset import TrajDataset, traj_collate_fn
from data.augmentation import MultiScaleAugmenter
from models.encoders import AdaptiveTrajectoryEncoder
from models.projectors import ProjectionHead
from models.learners import SemiSupervisedPrototypicalLearner, PrototypicalContrastiveLearner
from training.trainer import SemiSupervisedTrainer
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
    ensure_dir,
    get_device,
    Timer,
    make_json_serializable  # 添加这个导入
)
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')




class TransportModeRecognitionPipeline:
    """完整的交通方式识别流水线"""

    def __init__(self, config: Config, mode='train'):
        """
        Args:
            config: Config对象
            mode: 运行模式 ('train', 'eval', 'inference')
        """
        self.config = config
        self.mode = mode

        # 设置设备
        self.device = get_device(use_cuda=(config.experiment.device == 'cuda'))

        # 使用Config中已经设置好的实验目录
        self.exp_dir = Path(config.exp_dir)

        # 修改: 使用新的日志系统
        self.logger = get_logger(
            exp_name=config.experiment.exp_name,
            log_dir=str(self.exp_dir.parent)
        )

        self.logger.log_section(f"🚀 初始化交通方式识别系统 (模式: {mode})")
        self.logger.info(f"📁 实验目录: {self.exp_dir}")
        self.logger.info(f"🖥️  设备: {self.device}")

        # 设置随机种子
        set_seed(config.data.random_seed)

        # 标签名称映射
        self.label_names = config.label_names

        # 初始化组件
        self.data_loader = None
        self.preprocessor = None
        self.train_dataset = None
        self.val_dataset = None
        self.encoder = None
        self.projector = None
        self.learner = None
        self.trainer = None
        self.scaler = None

        # 修改: 初始化可视化器
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
        ✅ 改进：数据集划分（考虑标签分布）
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

        # 执行数据划分
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
        """步骤1: 设置数据加载和预处理 - 修改版"""
        self.logger.log_section("📂 步骤1: 数据加载与预处理")

        with Timer("数据加载"):
            self.data_loader = ImprovedGeoLifeDataLoader(
                data_root=self.config.data.data_root,
                label_mapping=self.config.label_mapping,
                min_overlap=self.config.data.get('min_label_overlap', 0.35)  # 新增：可调到 0.5
            )

            # ✅ 只加载有标签的用户
            raw_trajectories = self.data_loader.load_all_data(
                max_users=self.config.data.max_users,
                min_points=self.config.data.min_points,
                only_labeled_users=True,  # 只使用有标签的用户
                require_valid_label=True  # ✅ 要求轨迹必须有有效标签
            )

            if len(raw_trajectories) == 0:
                raise ValueError("❌ 未加载到任何轨迹数据！")

            self.logger.info(f"✅ 成功加载 {len(raw_trajectories)} 条原始轨迹")

        with Timer("数据预处理"):
            self.preprocessor = AdvancedTrajectoryPreprocessor(
                max_len=self.config.data.max_len
            )
            trajectories = self.preprocessor.process(raw_trajectories)

            # ✅ 保留有标签轨迹
            trajectories = [t for t in trajectories if t['label'] is not None and t['label'] >= 0]
            self.logger.info(f"✅ 保留 {len(trajectories)} 条有标签轨迹")

        # ✅ 先分层划分
        train_trajectories, val_trajectories = self._split_dataset_stratified(trajectories)

        # ✅ 在训练集上 fit，再分别 transform
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


        # ✅ 保存完整标签（用于评估）
        self.train_full_labels = np.array([t['label'] for t in train_trajectories])
        self.val_full_labels = np.array([t['label'] for t in val_trajectories])

        self.logger.info(f"   训练集完整标签: {len(self.train_full_labels)} 条")
        self.logger.info(f"   验证集完整标签: {len(self.val_full_labels)} 条")

        # ✅ 3. 应用标签掩码(训练用,深拷贝)
        import copy
        train_trajectories_masked = self._apply_label_masking(
            copy.deepcopy(train_trajectories),  # 深拷贝!
            label_ratio=self.config.data.labeled_ratio
        )

        # 创建数据集
        if self.config.experiment.use_augmentation:
            augmenter = MultiScaleAugmenter()
            self.train_dataset = TrajDataset(train_trajectories_masked, augment=False, augmenter=augmenter)
        else:
            self.train_dataset = TrajDataset(train_trajectories_masked, augment=False)

        self.val_dataset = TrajDataset(val_trajectories, augment=False)

        # ✅ 5. 验证数据一致性
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
                z = self.projector(emb)
                all_z.append(z.cpu().numpy())

        return np.vstack(all_z)

    def _split_dataset_stratified(
            self,
            trajectories: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        ✅ 分层划分数据集（保证标签分布）
        """
        from sklearn.model_selection import train_test_split

        labels_array = np.array([t['label'] for t in trajectories])

        train_indices, val_indices = train_test_split(
            np.arange(len(trajectories)),
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_seed,
            stratify=labels_array  # 分层划分
        )

        train_trajectories = [trajectories[i] for i in train_indices]
        val_trajectories = [trajectories[i] for i in val_indices]

        return train_trajectories, val_trajectories


    def _apply_labeled_ratio(self, trajectories: List[Dict]) -> List[Dict]:
        """
        ✅ 新增：控制标签数据比例

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
                             min_samples_per_class: int = 5):
        """隐藏部分标签(深拷贝,不影响原始数据)"""
        from collections import Counter

        if label_ratio >= 1.0:
            return trajectories

        if label_ratio >= 1.0:
            return trajectories

        masked_trajectories = [t.copy() for t in trajectories]
        labels = [t['label'] for t in masked_trajectories]
        label_counts = Counter(labels)

        self.logger.info(f"   📊 原始标签分布: {dict(label_counts)}")

        # 为每个类别保留指定比例的标签
        for label_id, count in label_counts.items():
            indices = [i for i, t in enumerate(masked_trajectories) if t['label'] == label_id]
            # ✅ 关键改进: 确保每类至少保留min_samples_per_class个样本
            n_keep = max(
                min_samples_per_class,  # 至少5个
                int(count * label_ratio)  # 或按比例
            )
            n_keep = min(n_keep, count)  # 不超过总数

            np.random.seed(self.config.data.random_seed)
            keep_indices = np.random.choice(indices, size=n_keep, replace=False)

            # ✅ 隐藏未选中的标签
            for idx in indices:
                if idx not in keep_indices:
                    masked_trajectories[idx]['label'] = -1

        # ✅ 统计可见标签
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
            encoder_mode=self.config.model.encoder_mode  # <--- 添加这一行
        ).to(self.device)

        # 2.2 创建投影头
        self.projector = ProjectionHead(
            input_dim=self.config.model.hidden_dim,
            hidden_dim=self.config.model.hidden_dim,
            output_dim=self.config.model.projection_dim
        ).to(self.device)

        # 2.3 统计参数量
        from utils.helper import count_parameters

        encoder_params = count_parameters(self.encoder)
        projector_params = count_parameters(self.projector)
        total_params = encoder_params['total'] + projector_params['total']

        self.logger.info(f"   编码器参数量: {encoder_params['total']:,}")
        self.logger.info(f"   投影头参数量: {projector_params['total']:,}")
        self.logger.info(f"   总参数量: {total_params:,}")

        # 2.4 创建学习框架
        if self.config.experiment.use_semi_supervised:
            self.logger.info("   ✅ 使用半监督学习框架")
            self.learner = SemiSupervisedPrototypicalLearner(
                self.encoder,
                self.projector,
                num_classes=self.config.experiment.num_classes,
                temperature=self.config.training.temperature,
                proto_weight=self.config.training.proto_weight,
                pseudo_weight=self.config.training.pseudo_weight,
                consistency_weight=self.config.training.consistency_weight,
                lr=self.config.training.lr,
                weight_decay=self.config.training.weight_decay
            )
        else:
            self.logger.info("   ✅ 使用标准原型对比学习")
            self.learner = PrototypicalContrastiveLearner(
                self.encoder,
                self.projector,
                num_classes=self.config.experiment.num_classes,
                temperature=self.config.training.temperature,
                proto_weight=self.config.training.proto_weight,
                lr=self.config.training.lr,
                weight_decay=self.config.training.weight_decay
            )

        return self.encoder, self.projector, self.learner

    def train(self):
        """步骤3: 训练模型"""
        self.logger.log_section("🏋️  步骤3: 开始训练")

        labels = np.array([t['label'] for t in self.train_dataset.trajs])
        labeled_mask = labels >= 0

        # 给有标签更高权重，比例可按数据情况微调
        w_labeled = 1.0
        w_unlabeled = max(0.1, 0.5 * (labeled_mask.sum() / max(1, (~labeled_mask).sum())))
        weights = np.where(labeled_mask, w_labeled, w_unlabeled).astype(np.float64)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.train_dataset),
            replacement=True
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            collate_fn=traj_collate_fn,
            num_workers=self.config.experiment.get('num_workers', 0),
            pin_memory=True if self.device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=self.config.experiment.get('num_workers', 0)
        )

        # 创建伪标签生成器
        # 创建伪标签生成器
        generator_name = self.config.training.pseudo_label_generator

        if generator_name == 'naive':
            self.logger.info("Using Naive Pseudo-Label Generator (Fixed Threshold)")
            pseudo_label_gen = PseudoLabelGenerator(
                # Table 2, Ablation 1: Fixed 0.95
                confidence_threshold=0.95
            )
        elif generator_name == 'advanced_no_margin':
            self.logger.info("Using Advanced Generator (w/o Margin Filter)")
            # Table 2, Ablation 2:
            pseudo_label_gen = AdvancedPseudoLabelGenerator(
                confidence_threshold=self.config.training.pseudo_label_threshold,
                progressive_threshold=True,
                consistency_check=True,
                top_k_consistency=3,
                margin_threshold=0.0  # <--- 关闭 margin
            )
        # ... (为您表格中的每个Ablation添加一个elif) ...
        else:
            self.logger.info("Using Full Advanced Pseudo-Label Generator")
            # Table 2, Full Model:
            pseudo_label_gen = AdvancedPseudoLabelGenerator(
                confidence_threshold=self.config.training.pseudo_label_threshold,
                progressive_threshold=True,
                consistency_check=True,
                top_k_consistency=3,
                margin_threshold=0.1,
                per_class_thresholds=getattr(self.config.training, 'pseudo_per_class_thresholds', {}),
                per_class_margin=getattr(self.config.training, 'pseudo_per_class_margin', {})
            )

        # ✅ 修改：传递 encoder 和 projector，而不是 learner
        # 将 Config 对象转换为字典
        config_dict = {
            'num_classes': self.config.experiment.num_classes,
            'projection_dim': self.config.model.projection_dim,
            'temperature': self.config.training.temperature,
            'proto_weight': self.config.training.proto_weight,
            'pseudo_weight': self.config.training.pseudo_weight,
            'consistency_weight': self.config.training.consistency_weight,
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
            'pseudo_warmup_epochs': 10,
            'pseudo_ramp_epochs': 50,
            'consistency_ramp_epochs': 50,
            'proto_ema': 0.9,
            'proto_ema_conf_thr': 0.9,
        }

        config_dict.update({
            'proto_margin': self.config.training.get('proto_margin', 0.0),
            'class_weights': self.config.training.get('class_weights', None),
            'num_attention_heads': self.config.model.get('num_attention_heads', 8),
            'encoder_mode': self.config.model.encoder_mode
        })

        self.trainer = SemiSupervisedTrainer(
            encoder=self.encoder,
            projector=self.projector,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_dict,
            device=self.device,
            experiment_dir=str(self.exp_dir),
            pseudo_label_generator=pseudo_label_gen
        )

        # 开始训练
        with Timer("模型训练"):
            history = self.trainer.fit()  # ✅ 使用 fit() 方法

        # 保存训练历史
        self._save_training_history(history)

        # 修改: 使用新的可视化方法
        self.visualizer.plot_training_curves(
            train_losses=history.get('train_loss', []),
            val_losses=history.get('val_loss'),
            contrast_losses=history.get('contrast_loss'),
            proto_losses=history.get('proto_loss'),
            pseudo_losses=history.get('pseudo_loss'),
            filename='training_curves.png'
        )

        # 如果有评估指标，也绘制出来
        if 'metrics' in history:
            self.visualizer.plot_metrics(
                metrics=history['metrics'],
                filename='metrics_trends.png'
            )

        return history

    # main.py (修改 evaluate 方法)

    def evaluate(self, checkpoint_path=None):
        """统一使用投影空间z进行所有评估"""
        self.logger.log_section("📊 步骤4: 模型评估（投影空间）")

        # 加载模型...

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.projector.load_state_dict(checkpoint['projector_state_dict'])
        prototypes = checkpoint.get('prototypes')

        self.encoder.eval()
        self.projector.eval()

        # ✅ 只提取投影空间embeddings
        with Timer("提取投影空间embeddings"):
            z_train = self._extract_projected_embeddings_from_dataset(
                self.train_dataset
            )
            z_val = self._extract_projected_embeddings_from_dataset(
                self.val_dataset
            )
            z_all = np.vstack([z_train, z_val])
            all_true_labels = np.hstack([
                self.train_full_labels,
                self.val_full_labels
            ])

        # === 1. 原型最近邻评估 ===
        with Timer("原型分类评估"):
            if prototypes is not None:
                proto_preds = self._predict_with_prototypes(z_all, prototypes)

                mode_evaluator = TransportModeEvaluator(
                    label_names=self.label_names,
                    save_dir=str(self.exp_dir / 'results')
                )
                # 只输出核心指标，并保存 proto_final_summary.json
                proto_results = mode_evaluator.evaluate(
                    pred_labels=proto_preds,
                    true_labels=all_true_labels,
                    save_prefix='proto_',
                    return_detailed=False,
                    selected_metrics=['accuracy', 'macro_f1', 'balanced_accuracy']
                )
                # 额外打印一行最终指标
                self.logger.info(
                    "[FINAL] Proto Accuracy: {:.4f} | Macro F1: {:.4f} | BalAcc: {:.4f}".format(
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

        # === 4. 聚类分析（仍用z） ===
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
            mapped_labels, cluster_to_label_map = mapper.map_clusters_to_labels(
                refined_clusters, all_true_labels
            )

            # 聚类评估
            cluster_evaluator = ClusteringEvaluator(
                save_dir=str(self.exp_dir / 'results')
            )
            cluster_results = cluster_evaluator.evaluate(
                z_all, refined_clusters, all_true_labels
            )

        # === 5. 伪标签质量评估（修复逻辑） ===
        with Timer("伪标签质量评估"):
            pseudo_eval_results = self._evaluate_pseudo_labels_correct(
                z_train,
                self.train_dataset,  # 传入dataset以获取观测标签
                self.train_full_labels,
                prototypes
            )

        # 保存结果...
        self._save_evaluation_results(
            cluster_results=cluster_results,
            mode_results=proto_results,  # 注意这里叫 mode_results
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

        # 归一化
        z = F.normalize(z, dim=1)
        protos = F.normalize(protos, dim=1)

        # 余弦相似度
        logits = z @ protos.t()
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

    def _evaluate_pseudo_labels_correct(
            self,
            z_train: np.ndarray,
            train_dataset,
            train_true_labels: np.ndarray,
            prototypes: torch.Tensor
    ) -> Dict:
        """修复后的伪标签质量评估"""
        # 获取观测标签（含-1）
        train_observed_labels = np.array([
            t['label'] for t in train_dataset.trajs
        ])

        # 找出被隐藏的样本
        unlabeled_mask = train_observed_labels < 0

        if unlabeled_mask.sum() == 0:
            return {'message': 'No hidden labels to evaluate'}

        # 用原型预测被隐藏样本的标签
        z_unlabeled = z_train[unlabeled_mask]
        pseudo_preds = self._predict_with_prototypes(z_unlabeled, prototypes)

        # 获取真实标签
        true_unlabeled = train_true_labels[unlabeled_mask]

        # 计算准确率
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        acc = float(accuracy_score(true_unlabeled, pseudo_preds))
        macro_f1 = float(f1_score(true_unlabeled, pseudo_preds, average='macro'))

        report = classification_report(
            true_unlabeled, pseudo_preds,
            target_names=[self.label_names[i] for i in range(self.config.experiment.num_classes)],
            output_dict=True
        )

        self.logger.info(
            f"伪标签质量: "
            f"隐藏样本={unlabeled_mask.sum()}, "
            f"准确率={acc:.4f}, "
            f"Macro F1={macro_f1:.4f}"
        )

        return {
            'num_hidden': int(unlabeled_mask.sum()),
            'accuracy': acc,
            'macro_f1': macro_f1,
            'report': report
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
        prototypes = checkpoint.get('prototypes', None)
        if prototypes is None:
            raise RuntimeError("No prototypes found in checkpoint; cannot perform prototypical inference.")
        prototypes = prototypes.to(self.device)

        self.encoder.eval()
        self.projector.eval()

        # 2) 加载训练阶段的特征标准化器（如果存在）
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

        # 6) 推理：在投影空间用原型进行分类
        results = []
        id_list = []
        with torch.no_grad():
            protos = F.normalize(prototypes, dim=1)
            for batch in loader:
                coords = batch['coords'].to(self.device)
                features = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)

                emb = self.encoder(features, coords, lengths)
                z = self.projector(emb)
                z = F.normalize(z, dim=1)

                logits = z @ protos.t()  # 余弦相似度
                probs = F.softmax(logits, dim=1)  # 概率
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

    # main.py (TransportModeRecognitionPipeline)

    def _build_pseudo_label_generator(self):
        """构建与训练阶段一致的伪标签生成器（用于评估）"""
        name = getattr(self.config.training, 'pseudo_label_generator', 'advanced')
        thr = getattr(self.config.training, 'pseudo_label_threshold', 0.85)
        if name == 'naive':
            return PseudoLabelGenerator(confidence_threshold=thr)
        elif name == 'advanced_no_margin':
            return AdvancedPseudoLabelGenerator(
                confidence_threshold=thr,
                progressive_threshold=True,
                consistency_check=True,
                top_k_consistency=3,
                margin_threshold=0.0
            )
        else:
            return AdvancedPseudoLabelGenerator(
                confidence_threshold=thr,
                progressive_threshold=True,
                consistency_check=True,
                top_k_consistency=3,
                margin_threshold=0.1
            )

    def _evaluate_knn(self, z_train, y_train, z_val, y_val, k=5):
        """kNN 在投影空间的快速评估"""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, f1_score, classification_report

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

        # 直接调用 UnifiedVisualizer 的综合报告方法
        # 这会自动生成所有需要的可视化

        # 但我们需要先准备 cluster_results 和 mode_results
        # 这两个在 evaluate() 中已经生成，所以可以作为参数传入

        # 为了兼容性，这里手动调用各个可视化方法

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
                # ✅ 修复：传入最佳模型路径
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

            # 添加这个finally块

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


    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 加载配置
    config = Config(config_path=args.config)

    # ✅ 覆盖标签比例配置
    if args.labeled_ratio is not None:
        if 0.0 <= args.labeled_ratio <= 1.0:
            config.data.labeled_ratio = args.labeled_ratio
        else:
            raise ValueError("❌ labeled_ratio 必须在 0.0 到 1.0 之间")

    # 覆盖配置
    if args.exp_name:
        config.experiment.exp_name = args.exp_name
        config._setup_paths()  # 重新设置路径

    if args.data:
        # 添加推理数据路径（可以扩展experiment配置）
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
    print(f"批次大小: {config.training.batch_size}")
    print(f"训练轮数: {config.training.epochs}")
    print(f"数据根目录: {config.data.data_root}")
    print(f"实验保存目录: {config.exp_dir}")
    print("=" * 70 + "\n")

    # 创建并运行流水线
    pipeline = TransportModeRecognitionPipeline(config, mode=args.mode)
    pipeline.run()


if __name__ == '__main__':
    main()
