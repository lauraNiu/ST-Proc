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
    data_root: str = '/autodl-fs/data/251021-mode-recognition/geolife/Geolife Trajectories 1.3/Geolife Trajectories 1.3/Data'
    max_users: int = None
    min_points: int = 20
    max_len: int = 100
    test_size: float = 0.1
    random_seed: int = 42
    # 标签控制参数
    labeled_ratio: float = 0.2  # 训练时可见标签的比例 (0.0-1.0)
    only_labeled_users: bool = True  # 只使用有标签的用户
    min_samples_per_class: int = 15  # 每类至少保留的样本数
    min_label_overlap: float = 0.35



    def get(self, key: str, default=None):
        """字典式访问"""
        return getattr(self, key, default)

@dataclass
class ModelConfig:
    """模型配置"""
    hidden_dim: int = 128
    projection_dim: int = 64
    feat_dim: int = 48
    coord_dim: int = 4
    dropout: float = 0.3
    num_encoder_layers: int = 2
    num_attention_heads: int = 4
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
    temperature: float = 0.1
    proto_weight: float = 1.0
    pseudo_weight: float = 0.5
    consistency_weight: float = 0.2
    patience: int = 50
    min_delta: float = 1e-4
    pseudo_label_interval: int = 3
    pseudo_label_threshold: float = 0.8
    use_contrastive: bool = True
    use_proto: bool = True
    pseudo_label_generator: str = 'advanced'
    pseudo_warmup_epochs: int = 5
    pseudo_ramp_epochs: int = 50
    consistency_ramp_epochs: int = 50
    graph_k: int = 10                     # kNN 图的 k
    lambda_graph_smooth: float = 0.05     # 图拉普拉斯平滑正则权重
    lambda_graph_contrast: float = 0.10   # 邻居对比损失权重
    graph_build_interval: int = 1         # 每多少个 epoch 重建一次全局 kNN 图

    proto_margin: float = 0.15  # CosFace 风格 margin
    class_weights: Optional[List[float]] = None  # 新增：按类别顺序的权重，或 None

    # 伪标签 per-class 配置（bus(2)/car(3)  0.90 / 0.12）
    pseudo_per_class_thresholds: Dict[int, float] = field(default_factory=dict)
    pseudo_per_class_margin: Dict[int, float] = field(default_factory=dict)

    def get(self, key: str, default=None):
        """字典式访问"""
        return getattr(self, key, default)


@dataclass
class ExperimentConfig:
    """实验配置"""
    num_clusters: int = 5
    num_classes: int = 5
    use_smote: bool = False
    use_augmentation: bool = False
    use_semi_supervised: bool = False
    exp_name: str = 'exp_test_0.2_wosemi'
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


