# tune_hyperparams.py
"""
超参数自动调优脚本
使用Optuna进行贝叶斯优化
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
from torch.utils.data import DataLoader

# 导入项目模块
from geolife.graph.config.config import Config
from geolife.graph.data.loader import ImprovedGeoLifeDataLoader
from geolife.graph.data.preprocessor import AdvancedTrajectoryPreprocessor
from geolife.graph.data.dataset import TrajDataset, traj_collate_fn
from geolife.graph.data.augmentation import MultiScaleAugmenter
from geolife.graph.models.encoders import AdaptiveTrajectoryEncoder
from geolife.graph.models.projectors import ProjectionHead
from geolife.graph.training.trainer import SemiSupervisedTrainer
from geolife.graph.evaluation.clustering import ClusterLabelMapper, perform_clustering
from geolife.graph.evaluation.metrics import TransportModeEvaluator
from geolife.graph.training.pseudo_label import AdvancedPseudoLabelGenerator
from geolife.graph.utils.logger import get_logger
from geolife.graph.utils.helper import get_device

warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """超参数调优器"""

    def __init__(
            self,
            base_config_path: str = None,
            n_trials: int = 50,
            n_jobs: int = 1,
            storage: str = None,
            study_name: str = None,
            gpu_id: str = '0'
    ):
        """
        Args:
            base_config_path: 基础配置文件路径
            n_trials: 试验次数
            n_jobs: 并行任务数
            storage: Optuna数据库路径
            study_name: 研究名称
            gpu_id: GPU编号
        """
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.gpu_id = gpu_id

        # 加载基础配置
        self.base_config = Config(base_config_path)

        # 设置存储
        if storage is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            storage_dir = Path('./optuna_studies')
            storage_dir.mkdir(exist_ok=True)
            storage = f'sqlite:///{storage_dir}/study_{timestamp}.db'

        if study_name is None:
            study_name = f'transport_mode_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        self.storage = storage
        self.study_name = study_name

        # 日志
        self.logger = get_logger(
            exp_name='hyperparameter_tuning',
            log_dir='./tuning_logs'
        )

        self.logger.info(f"🎯 初始化超参数调优器")
        self.logger.info(f"   试验次数: {n_trials}")
        self.logger.info(f"   并行任务: {n_jobs}")
        self.logger.info(f"   存储: {storage}")
        self.logger.info(f"   研究名称: {study_name}")

        # 缓存数据（避免重复加载）
        self.cached_data = None

    def _prepare_data_once(self):
        """只加载一次数据（所有trial共享）"""
        if self.cached_data is not None:
            return self.cached_data

        self.logger.info("📂 准备数据（仅执行一次）...")

        # 数据加载
        data_loader = ImprovedGeoLifeDataLoader(
            data_root=self.base_config.data.data_root,
            label_mapping=self.base_config.label_mapping
        )

        raw_trajectories = data_loader.load_all_data(
            max_users=self.base_config.data.max_users,
            min_points=self.base_config.data.min_points,
            only_labeled_users=True,
            require_valid_label=True
        )

        # 数据预处理
        preprocessor = AdvancedTrajectoryPreprocessor(
            max_len=self.base_config.data.max_len
        )
        trajectories = preprocessor.process(raw_trajectories)

        # 过滤有标签轨迹
        trajectories = [
            t for t in trajectories
            if t['label'] is not None and t['label'] >= 0
        ]

        # 特征标准化
        from sklearn.preprocessing import StandardScaler
        all_features = np.array([t['features'] for t in trajectories])
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)

        for i, traj in enumerate(trajectories):
            traj['features'] = all_features_scaled[i]

        # 数据集划分
        from sklearn.model_selection import train_test_split
        labels_array = np.array([t['label'] for t in trajectories])

        train_indices, val_indices = train_test_split(
            np.arange(len(trajectories)),
            test_size=self.base_config.data.test_size,
            random_state=self.base_config.data.random_seed,
            stratify=labels_array
        )

        train_trajectories = [trajectories[i] for i in train_indices]
        val_trajectories = [trajectories[i] for i in val_indices]

        # 保存完整标签
        train_full_labels = np.array([t['label'] for t in train_trajectories])
        val_full_labels = np.array([t['label'] for t in val_trajectories])

        # 应用标签隐藏
        train_trajectories_masked = self._apply_label_masking(
            train_trajectories,
            self.base_config.data.labeled_ratio
        )

        self.cached_data = {
            'train_trajectories': train_trajectories_masked,
            'val_trajectories': val_trajectories,
            'train_full_labels': train_full_labels,
            'val_full_labels': val_full_labels,
            'scaler': scaler
        }

        self.logger.info(f"✅ 数据准备完成: 训练集={len(train_trajectories_masked)}, 验证集={len(val_trajectories)}")

        return self.cached_data

    def _apply_label_masking(self, trajectories, label_ratio):
        """隐藏部分标签"""
        from collections import Counter

        if label_ratio >= 1.0:
            return trajectories

        masked_trajectories = [t.copy() for t in trajectories]
        labels = [t['label'] for t in masked_trajectories]
        label_counts = Counter(labels)

        for label_id, count in label_counts.items():
            indices = [i for i, t in enumerate(masked_trajectories)
                       if t['label'] == label_id]
            n_keep = max(1, int(count * label_ratio))

            np.random.seed(self.base_config.data.random_seed)
            keep_indices = np.random.choice(indices, size=n_keep, replace=False)

            for idx in indices:
                if idx not in keep_indices:
                    masked_trajectories[idx]['label'] = -1

        return masked_trajectories

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna目标函数

        Returns:
            目标指标值（weighted F1-score）
        """
        try:
            # 1. 建议超参数
            params = self._suggest_hyperparameters(trial)

            # 2. 创建配置
            config = self._create_config(params, trial.number)

            # 3. 训练和评估
            metrics = self._train_and_evaluate(config, trial)

            # 4. 返回目标值
            objective_value = metrics['accuracy']

            # 记录结果
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Trial {trial.number} 完成:")
            self.logger.info(f"  Weighted F1: {objective_value:.4f}")
            self.logger.info(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
            self.logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"  参数: {params}")
            self.logger.info(f"{'=' * 60}\n")

            return objective_value

        except Exception as e:
            self.logger.error(f"Trial {trial.number} 失败: {str(e)}")
            return 0.0

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        定义超参数搜索空间
        """
        # 先选择 hidden_dim
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])

        # 根据 hidden_dim 选择合法的 num_attention_heads
        # valid_heads = [h for h in [2, 4, 8] if hidden_dim % h == 0]
        # num_attention_heads = trial.suggest_categorical('num_attention_heads', valid_heads)

        params = {
            # 🔹 损失权重
            'proto_weight': trial.suggest_float('proto_weight', 0.5, 1.2, step=0.1),
            'pseudo_weight': trial.suggest_float('pseudo_weight', 0.1, 0.5, step=0.1),
            'consistency_weight': trial.suggest_float('consistency_weight', 0.1, 0.3, step=0.1),

            # 🔹 伪标签阈值
            'pseudo_label_threshold': trial.suggest_float('pseudo_label_threshold', 0.7, 0.95, step=0.05),

            # 🔹 模型架构
            'hidden_dim': hidden_dim,
            'projection_dim': trial.suggest_categorical('projection_dim', [32, 64, 128]),
            'dropout': trial.suggest_float('dropout', 0.2, 0.5, step=0.1),
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [4, 8]),

            # 🔹 训练参数
            'lr': trial.suggest_loguniform('lr', 1e-5, 1e-3),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-3),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        }

        return params

    def _create_config(self, params: Dict[str, Any], trial_number: int) -> Config:
        """根据超参数创建配置"""
        config = Config()

        # 复制基础配置
        config.data = self.base_config.data
        config.experiment = self.base_config.experiment

        # 更新模型参数
        config.model.hidden_dim = params['hidden_dim']
        config.model.projection_dim = params['projection_dim']
        config.model.dropout = params['dropout']
        config.model.num_attention_heads = params['num_attention_heads']

        # 更新训练参数
        config.training.proto_weight = params['proto_weight']
        config.training.pseudo_weight = params['pseudo_weight']
        config.training.consistency_weight = params['consistency_weight']
        config.training.pseudo_label_threshold = params['pseudo_label_threshold']
        config.training.lr = params['lr']
        config.training.weight_decay = params['weight_decay']
        config.training.batch_size = params['batch_size']

        # 减少训练轮数（加速调参）
        config.training.epochs = 100
        config.training.patience = 15

        # 设置实验名称
        config.experiment.exp_name = f'tune_trial_{trial_number:03d}'
        config._setup_paths()

        return config

    def _train_and_evaluate(
            self,
            config: Config,
            trial: optuna.Trial
    ) -> Dict[str, float]:
        """训练并评估模型"""
        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        device = get_device(use_cuda=True)

        # 加载缓存数据
        data = self._prepare_data_once()

        # 创建数据集
        if config.experiment.use_augmentation:
            augmenter = MultiScaleAugmenter()
            train_dataset = TrajDataset(data['train_trajectories'], augment=True, augmenter=augmenter)
        else:
            train_dataset = TrajDataset(data['train_trajectories'], augment=False)

        val_dataset = TrajDataset(data['val_trajectories'], augment=False)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=traj_collate_fn,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size * 2,
            shuffle=False,
            collate_fn=traj_collate_fn,
            num_workers=0
        )

        # 创建模型
        encoder = AdaptiveTrajectoryEncoder(
            feat_dim=config.model.feat_dim,
            coord_dim=config.model.coord_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            num_heads=config.model.num_attention_heads
        ).to(device)

        projector = ProjectionHead(
            input_dim=config.model.hidden_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.projection_dim
        ).to(device)

        # 配置字典
        config_dict = {
            'num_classes': config.experiment.num_classes,
            'projection_dim': config.model.projection_dim,
            'temperature': config.training.temperature,
            'proto_weight': config.training.proto_weight,
            'pseudo_weight': config.training.pseudo_weight,
            'consistency_weight': config.training.consistency_weight,
            'lr': config.training.lr,
            'weight_decay': config.training.weight_decay,
            'epochs': config.training.epochs,
            'patience': config.training.patience,
            'feat_dim': config.model.feat_dim,
            'coord_dim': config.model.coord_dim,
            'hidden_dim': config.model.hidden_dim,
            'dropout': config.model.dropout,
            'ema_decay': 0.999,
            'use_amp': True,
            'max_grad_norm': 1.0,
            'mask_ratio': 0.2,  # ✅ 添加这个
            'optimizer': 'adamw',  # ✅ 添加这个
            'scheduler': 'cosine',  # ✅ 添加这个
            'warmup_epochs': 5,  # ✅ 添加这个
            'save_interval': 10  # ✅ 添加这个
        }

        # 创建Trainer
        trainer = SemiSupervisedTrainer(
            encoder=encoder,
            projector=projector,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_dict,
            device=device,
            experiment_dir=str(config.exp_dir)
        )

        # 创建伪标签生成器
        pseudo_label_gen = AdvancedPseudoLabelGenerator(
            confidence_threshold=config.training.pseudo_label_threshold,
            progressive_threshold=True,
            consistency_check=True
        )

        # 训练（带pruning）
        history = self._train_with_pruning(trainer, pseudo_label_gen, trial, config)

        # 评估
        metrics = self._evaluate_model(
            encoder, projector, train_dataset, val_dataset,
            data['train_full_labels'], data['val_full_labels'],
            config, device
        )

        # 清理显存
        del encoder, projector, trainer
        torch.cuda.empty_cache()

        return metrics

    def _train_with_pruning(
            self,
            trainer: SemiSupervisedTrainer,
            pseudo_label_gen: AdvancedPseudoLabelGenerator,
            trial: optuna.Trial,
            config: Config
    ) -> Dict:
        """带剪枝的训练"""
        history = {'train_loss': [], 'val_loss': [], 'best_val_loss': float('inf')}

        # ✅ 在第一轮训练前初始化伪标签字典
        trainer.pseudo_labels_dict = None

        for epoch in range(config.training.epochs):
            trainer.current_epoch = epoch

            # 训练
            train_metrics = trainer.train_epoch()
            history['train_loss'].append(train_metrics['total_loss'])

            # 验证
            val_metrics = trainer.validate()
            history['val_loss'].append(val_metrics['total_loss'])

            if val_metrics['total_loss'] < history['best_val_loss']:
                history['best_val_loss'] = val_metrics['total_loss']
                history['best_epoch'] = epoch

            # ✅ 确保在合适的时机更新伪标签
            if epoch > 0 and epoch % config.training.pseudo_label_interval == 0:
                try:
                    trainer.update_pseudo_labels(pseudo_label_gen)
                except Exception as e:
                    self.logger.warning(f"伪标签更新失败: {e}")

            # Optuna剪枝
            trial.report(val_metrics['total_loss'], epoch)
            if trial.should_prune():
                self.logger.info(f"  ✂️ Trial {trial.number} 被剪枝 (Epoch {epoch})")
                raise optuna.TrialPruned()

            # Early stopping
            if epoch - history.get('best_epoch', 0) > config.training.patience:
                break

        return history

    def _evaluate_model(
            self, encoder, projector, train_dataset, val_dataset,
            train_full_labels, val_full_labels, config, device
    ) -> Dict[str, float]:
        """评估模型性能"""
        encoder.eval()
        projector.eval()

        # 提取embeddings
        def extract_embeddings(dataset):
            loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=traj_collate_fn)
            embeddings, labels = [], []

            with torch.no_grad():
                for batch in loader:
                    coords = batch['coords'].to(device)
                    features = batch['features'].to(device)
                    lengths = batch['lengths'].to(device)

                    emb = encoder(features, coords, lengths)
                    embeddings.append(emb.cpu().numpy())
                    labels.append(batch['labels'].cpu().numpy())

            return np.vstack(embeddings), np.hstack(labels)

        train_emb, _ = extract_embeddings(train_dataset)
        val_emb, _ = extract_embeddings(val_dataset)

        all_embeddings = np.vstack([train_emb, val_emb])
        all_true_labels = np.hstack([train_full_labels, val_full_labels])

        # 聚类
        clusters, refined_clusters = perform_clustering(
            all_embeddings,
            n_clusters=config.experiment.num_clusters,
            method='kmeans',
            refine=True,
            logger=self.logger
        )

        # 映射到标签
        mapper = ClusterLabelMapper(num_classes=config.experiment.num_classes)
        mapped_labels, _ = mapper.map_clusters_to_labels(refined_clusters, all_true_labels)

        # ✅ 修复：确保 save_dir 是 Path 对象
        from pathlib import Path
        results_dir = Path(config.exp_dir) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        # 计算指标
        evaluator = TransportModeEvaluator(
            label_names=config.label_names,
            save_dir=str(results_dir)  # ✅ 传入字符串路径
        )
        results = evaluator.evaluate(mapped_labels, all_true_labels)

        return {
            'weighted_f1': results['weighted_f1'],
            'macro_f1': results['macro_f1'],
            'accuracy': results['accuracy']
        }

    def run(self):
        """运行超参数优化"""
        self.logger.log_section("🚀 开始超参数优化")

        # 创建study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # 运行优化
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )

        # 保存结果
        self._save_results(study)
        self._print_best_results(study)

        return study

    def _save_results(self, study: optuna.Study):
        """保存优化结果"""
        results_dir = Path('./tuning_results')
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存最佳参数
        with open(results_dir / f'best_params_{timestamp}.json', 'w') as f:
            json.dump(study.best_params, f, indent=4)

        # 保存所有试验
        trials_df = study.trials_dataframe()
        trials_df.to_csv(results_dir / f'all_trials_{timestamp}.csv', index=False)

        # 生成可视化
        self._generate_visualization(study, results_dir, timestamp)

        self.logger.info(f"\n✅ 结果已保存到: {results_dir}")

    def _generate_visualization(self, study, save_dir, timestamp):
        """生成可视化报告"""
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate
            )

            # 优化历史
            fig = plot_optimization_history(study)
            fig.write_html(str(save_dir / f'optimization_history_{timestamp}.html'))

            # 参数重要性
            fig = plot_param_importances(study)
            fig.write_html(str(save_dir / f'param_importances_{timestamp}.html'))

            # 平行坐标图
            fig = plot_parallel_coordinate(study)
            fig.write_html(str(save_dir / f'parallel_coordinate_{timestamp}.html'))

        except Exception as e:
            self.logger.warning(f"生成可视化失败: {str(e)}")

    def _print_best_results(self, study: optuna.Study):
        """打印最佳结果"""
        self.logger.log_section("🏆 最佳超参数")
        self.logger.info(f"最佳 Weighted F1: {study.best_value:.4f}")
        self.logger.info(f"最佳试验编号: {study.best_trial.number}")
        self.logger.info(f"\n最佳参数:")

        for key, value in study.best_params.items():
            self.logger.info(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description='超参数自动调优')
    parser.add_argument('--config', type=str, default=None, help='基础配置文件')
    parser.add_argument('--n-trials', type=int, default=50, help='试验次数')
    parser.add_argument('--n-jobs', type=int, default=1, help='并行任务数')
    parser.add_argument('--gpu', type=str, default='0', help='GPU编号')
    parser.add_argument('--study-name', type=str, default=None, help='Study名称')
    parser.add_argument('--storage', type=str, default=None, help='数据库路径')
    return parser.parse_args()


def main():
    args = parse_args()

    tuner = HyperparameterTuner(
        base_config_path=args.config,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=args.study_name,
        gpu_id=args.gpu
    )

    study = tuner.run()

    print("\n" + "=" * 70)
    print("✅ 超参数优化完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
