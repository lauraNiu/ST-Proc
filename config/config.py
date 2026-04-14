# config/config.py

"""
配置管理模块
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml
import json
@dataclass
class DataConfig:
    """数据配置"""
    data_root: str = '/autodl-fs/data/251021-mode-recognition/geolife/graph/Dataset/Geolife Trajectories 1.3/Geolife Trajectories 1.3/Data'
    max_users: int = None
    min_points: int = 20
    max_len: int = 200
    test_size: float = 0.2
    random_seed: int = 42
    # 标签控制参数
    labeled_ratio: float = 0.2  # 训练时可见标签的比例 (0.0-1.0)
    only_labeled_users: bool = False
    include_real_unlabeled_in_train: bool = False  # benchmark 默认关闭；仅显式开启 real-world SSL 时并入真实无标签用户
    require_valid_label: bool = False
    min_samples_per_class: int = 10  # 每类至少保留的样本数
    min_label_overlap: float = 0.35



    def get(self, key: str, default=None):
        """字典式访问"""
        return getattr(self, key, default)

@dataclass
class ModelConfig:
    """模型配置"""
    hidden_dim: int = 256
    projection_dim: int = 128
    feat_dim: int = 54
    coord_dim: int = 4
    dropout: float = 0.2
    num_encoder_layers: int = 3
    num_attention_heads: int = 8
    encoder_mode: str = 'adaptive_gate'  # 'adaptive_gate', 'st_only', 'stats_only', 'simple_fusion'

    def get(self, key: str, default=None):
        """字典式访问"""
        return getattr(self, key, default)


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64
    epochs: int = 200
    lr: float = 0.0003
    weight_decay: float = 1e-4
    temperature: float = 0.07
    contrast_weight: float = 1.0
    proto_weight: float = 1.0
    pseudo_weight: float = 1.0          # 提高伪标签损失权重
    consistency_weight: float = 0.1
    patience: int = 40
    min_delta: float = 1e-4
    pseudo_label_interval: int = 2
    pseudo_label_threshold: float = 0.75
    pseudo_progressive_threshold: bool = False  # 默认关闭后期放松阈值
    pseudo_threshold_min: float = 0.75     # 关闭 progressive 时保持与主阈值一致
    pseudo_threshold_decay: float = 0.00   # 保守模式下默认不衰减
    pseudo_threshold_decay_interval: int = 10
    pseudo_confidence_temperature: float = 0.10   # 收尖原型概率，使高置信样本能通过阈值
    pseudo_low_margin_penalty: float = 0.95       # 轻微放松 low-margin 惩罚，但仍保留边界样本抑制
    pseudo_max_adoption_rate: float = 0.25        # 每次更新最多采纳 25% 的无标签样本
    pseudo_max_adoption_count: int = 0            # 0 表示仅按比例上限裁剪
    pseudo_lp_threshold: float = 0.92             # LP-only 分支显著更严格
    pseudo_conflict_threshold: float = 0.95       # 冲突时只允许极高置信的一方胜出
    pseudo_conflict_margin: float = 0.15          # 冲突时需要更大的置信度差
    pseudo_allow_lp_only: bool = False            # 默认禁止仅靠 LP 放行伪标签
    pseudo_lp_only_warmup_epochs: int = 15        # 训练稳定后再允许少量 LP-only
    pseudo_lp_max_adoption_rate: float = 0.05     # LP-only 最多占无标签池 5%
    pseudo_lp_agree_bonus: float = 0.03           # base 与 LP 一致时给 LP 轻微置信度加成
    lp_min_support: float = 0.55                  # 小幅放松 LP 支撑门槛，仅帮助 agreement 样本形成
    use_contrastive: bool = True
    use_proto: bool = True
    pseudo_label_generator: str = 'advanced'
    pseudo_warmup_epochs: int = 3          # 更短的 warmup，尽早开始利用伪标签
    pseudo_ramp_epochs: int = 30
    consistency_ramp_epochs: int = 30
    graph_k: int = 15
    lambda_graph_smooth: float = 0.10
    lambda_graph_contrast: float = 0.20
    graph_build_interval: int = 1
    lp_alpha: float = 0.95
    lp_iters: int = 20
    proto_ema: float = 0.95
    proto_ema_conf_thr: float = 0.90       # 提高门槛，避免大量伪标签反向污染原型
    use_pseudo_for_proto_ema: bool = False
    pseudo_proto_ema_agreement_only: bool = True
    pseudo_class_balance_power: float = 0.6       # 伪标签预算更多向稀有类倾斜
    pseudo_class_balance_min_count: int = 4       # 每次更新每类至少保留的最小伪标签候选数
    pseudo_adaptive_per_class: bool = True        # 根据当前可见类频动态调整 per-class threshold
    pseudo_rare_class_gamma: float = 0.08         # 稀有类阈值/边界放松幅度
    pseudo_margin_relax_gamma: float = 0.03       # 稀有类 margin 放松幅度
    pseudo_confidence_floor: float = 0.55         # 融合后最低有效置信度
    pseudo_quality_momentum: float = 0.70         # per-class pseudo precision 的 EMA 系数
    pseudo_quality_relax_threshold: float = 0.93  # 历史 precision 足够高时才允许放松 hard class 阈值
    pseudo_quality_strict_threshold: float = 0.85 # 历史 precision 偏低时主动收紧 hard class 阈值
    pseudo_quality_strict_scale: float = 0.20     # 低质量 hard class 的额外阈值增量缩放
    pseudo_hard_class_extra_strictness: Dict[int, float] = field(default_factory=lambda: {2: 0.03, 4: 0.05})
    pseudo_distribution_alignment: bool = True
    pseudo_distribution_momentum: float = 0.90
    pseudo_distribution_min_prob: float = 1e-3
    pseudo_teacher_temperature: float = 1.10
    pseudo_proto_temperature: float = 1.00
    pseudo_reliability_gate: bool = True
    pseudo_reliability_power: float = 1.0
    pseudo_reliability_floor: float = 0.35
    pseudo_reliability_warmup_epochs: int = 10
    pseudo_lp_agree_threshold_offset: float = 0.03
    pseudo_lp_conf_power: float = 0.75
    pseudo_lp_min_purity: float = 0.65
    lp_graph_temperature: float = 0.20
    lp_mutual_knn: bool = True
    lp_seed_weight: float = 0.35
    lp_entropy_weight: float = 0.20
    lp_neighbor_agreement_weight: float = 0.25

    low_ratio_cutoff: float = 0.10
    mid_ratio_cutoff: float = 0.30
    low_ratio_pseudo_warmup_epochs: int = 18
    low_ratio_pseudo_label_interval: int = 5
    low_ratio_initial_pseudo_max_adoption_rate: float = 0.05
    low_ratio_target_pseudo_max_adoption_rate: float = 0.12
    low_ratio_cap_ramp_epochs: int = 30
    low_ratio_cap_ramp_min_quality: float = 0.88
    low_ratio_cap_ramp_max_quality: float = 0.94
    low_ratio_pseudo_class_balance_min_count: int = 0
    low_ratio_hard_negative_weight_scale: float = 0.35
    low_ratio_coarse_aux_weight_scale: float = 0.50
    low_ratio_aux_ramp_epochs: int = 35
    mid_ratio_initial_pseudo_max_adoption_rate: float = 0.15
    mid_ratio_target_pseudo_max_adoption_rate: float = 0.20
    mid_ratio_pseudo_warmup_epochs: int = 10
    mid_ratio_pseudo_label_interval: int = 3

    prototype_stage_low_ratio_cutoff: float = 0.10
    prototype_per_class_map_low_ratio: Dict[int, int] = field(default_factory=lambda: {2: 3, 4: 3})
    prototype_expand_epoch: int = 28
    prototype_expand_quality_thr: float = 0.92

    use_ssl_pretrain: bool = True
    pretrain_epochs: int = 20
    low_ratio_pretrain_epochs: int = 30
    pretrain_lr: float = 3e-4
    pretrain_weight_decay: float = 1e-4
    pretrain_graph_smooth_weight: float = 0.05
    pretrain_graph_contrast_weight: float = 0.10

    proto_margin: float = 0.10
    # walk(0)/bike(1)/bus(2)/car(3)/subway(4)
    class_weights: Optional[List[float]] = field(default_factory=lambda: [1.0, 1.2, 2.5, 1.0, 3.5])

    # 默认不再主动下调某些类别门槛，避免 bus/car/subway 被过量放行
    pseudo_per_class_thresholds: Dict[int, float] = field(default_factory=dict)
    pseudo_per_class_margin: Dict[int, float] = field(default_factory=dict)

    # 分类头权重（相对于 prototype loss）
    classifier_weight: float = 1.0
    classifier_weight_final: float = 1.6
    classifier_label_smoothing: float = 0.03
    classifier_hidden_dim: int = 256
    use_hierarchical_classifier: bool = True
    hierarchical_prior_scale: float = 0.70
    use_effective_class_balancing: bool = True
    class_balance_beta: float = 0.999
    class_balance_weight_power: float = 1.0
    class_balance_max_weight: float = 6.0
    use_logit_adjustment: bool = True
    logit_adjust_tau: float = 1.0
    prototypes_per_class: int = 3
    prototype_per_class_map: Dict[int, int] = field(default_factory=lambda: {2: 5, 4: 6})
    prototype_pooling: str = 'logsumexp'
    prototype_pool_temperature: float = 0.35
    proto_weight_final: float = 0.70
    contrast_weight_final: float = 0.25
    graph_weight_final: float = 0.35
    use_stagewise_loss_schedule: bool = True
    classification_stage_start: int = 20
    classification_stage_ramp: int = 40
    selection_metric: str = 'macro_f1'
    disable_ssl_for_full_label: bool = True
    full_label_pseudo_weight: float = 0.0
    full_label_consistency_weight: float = 0.0
    full_label_lambda_graph_smooth: float = 0.0
    full_label_lambda_graph_contrast: float = 0.0

    # Teacher classifier 主导伪标签生成
    use_teacher_clf_pseudo: bool = True           # 用 teacher classifier 主导伪标签生成
    teacher_clf_pseudo_weight: float = 0.7        # teacher clf 置信度权重
    proto_pseudo_weight: float = 0.3              # prototype 置信度权重（辅助）
    pseudo_class_quota_per_update: int = 0        # 0=不限，>0 则每类最多采纳 N 个
    patience_after_pseudo: int = 20               # 伪标签激活后额外保护 epoch 数

    # GNN 聚合层
    use_gnn_aggregation: bool = True              # 在 encoder 输出后应用 GAT 图聚合

    # 采样器
    sampler_unlabeled_weight: float = 0.20
    sampler_class_balance_power: float = 0.60
    sampler_min_class_weight: float = 0.05
    sampler_max_class_weight: float = 4.0
    sampler_hard_class_boost: Dict[int, float] = field(default_factory=lambda: {2: 1.35, 4: 1.60})

    # 难类判别增强
    hard_negative_pairs: List[List[int]] = field(default_factory=lambda: [[2, 4], [2, 3]])
    hard_negative_margin: float = 0.25
    hard_negative_weight: float = 0.20
    coarse_groups: List[List[int]] = field(default_factory=lambda: [[0, 1], [2, 3], [4]])
    coarse_aux_weight: float = 0.15

    def get(self, key: str, default=None):
        """字典式访问"""
        return getattr(self, key, default)


@dataclass
class ExperimentConfig:
    """实验配置"""
    num_clusters: int = 5
    num_classes: int = 5
    use_smote: bool = False
    use_augmentation: bool = True
    use_semi_supervised: bool = True
    exp_name: str = 'exp_graph_ssl_0.2'
    save_dir: str = './ablation_experiments'
    device: str = 'cuda'
    num_workers: int = 8

    def get(self, key: str, default=None):
        """字典式访问"""
        return getattr(self, key, default)


class Config:
    """统一配置管理器"""

    def __init__(self, config_path: str = None):
        #  1. 先初始化内部路径变量
        self._exp_dir = None
        self._checkpoint_dir = None
        self._log_dir = None
        self._result_dir = None

        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.experiment = ExperimentConfig()

        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)

        self._setup_paths()
        # GeoLife 数据集的标准标签映射
        # 原始标签映射
        self.original_label_mapping = {
            'walk': 0,
            'bike': 1,
            'bus': 2,
            'car': 3,
            'subway': 4,
            'train': 5,
            'airplane': 6,
            'boat': 7,
            'run': 8,
            'motorcycle': 9,
            'taxi': 10
        }

        # 标签合并映射（原始标签ID -> 新标签ID）
        self.label_mapping = {
            0: 0,  # walk -> walk
            1: 1,  # bike -> bike
            2: 2,  # bus -> bus
            3: 3,  # car -> car
            4: 4,  # subway -> subway
            5: -1,  # train -> 排除
            6: -1,  # airplane -> 排除
            7: -1,  # boat -> 排除
            8: -1,  # run -> 排除
            9: -1,  # motorcycle -> 排除
            10: 3  # taxi -> car (合并)
        }

    def _setup_paths(self):
        """设置实验路径"""
        # 赋值给内部变量（带下划线）
        self._exp_dir = os.path.join(
            self.experiment.save_dir,
            self.experiment.exp_name
        )
        self._checkpoint_dir = os.path.join(self._exp_dir, 'checkpoints')
        self._log_dir = os.path.join(self._exp_dir, 'logs')
        self._result_dir = os.path.join(self._exp_dir, 'results')

        # 创建目录
        for dir_path in [self._checkpoint_dir, self._log_dir, self._result_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def load_from_file(self, config_path: str):
        """从文件加载配置"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)

        # 更新配置
        for key, value in config_dict.get('data', {}).items():
            setattr(self.data, key, value)
        for key, value in config_dict.get('model', {}).items():
            setattr(self.model, key, value)
        for key, value in config_dict.get('training', {}).items():
            setattr(self.training, key, value)
        for key, value in config_dict.get('experiment', {}).items():
            setattr(self.experiment, key, value)

    def save(self, save_path: str):
        """保存配置到文件"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment': self.experiment.__dict__
        }

        with open(save_path, 'w') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=4)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment': self.experiment.__dict__
        }

    def get(self, key: str, default=None):
        """
        兼容性方法：支持 config.get('key', default)
        """
        # 按优先级搜索配置项
        for section in [self.experiment, self.training, self.model, self.data]:
            if hasattr(section, key):
                return getattr(section, key)
        return default

    @property
    def label_names(self) -> Dict[int, str]:
        """交通方式标签名称 - 合并后的版本"""
        return {
            0: 'walk',
            1: 'bike',
            2: 'bus',
            3: 'car',  # 合并了car和taxi
            4: 'subway'
        }

    @property
    def original_to_merged_labels(self) -> Dict[int, int]:
        """原始标签到合并标签的映射"""
        return {
            0: 0,  # walk -> walk
            1: 1,  # bike -> bike
            2: 2,  # bus -> bus
            3: 3,  # car -> car
            4: 4,  # subway -> subway
            5: -1,  # train -> 忽略
            6: -1,  # airplane -> 忽略
            7: -1,  # boat -> 忽略
            8: -1,  # run -> 忽略
            9: -1,  # motorcycle -> 忽略
            10: 3  # taxi -> car (合并)
        }

    @property
    def exp_dir(self):
        return self._exp_dir

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def result_dir(self):
        return self._result_dir


