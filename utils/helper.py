"""
辅助工具函数模块
提供常用的工具函数和辅助类
"""

import os
import json
import pickle
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from collections import defaultdict
import hashlib
from datetime import datetime


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善量
            mode: 'min' 或 'max'
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        检查是否应该早停

        Returns:
            bool: True 表示应该停止训练
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False

        # 检查是否有改善
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"   ✅ 性能提升! 最佳epoch: {epoch}")
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"   ⏳ EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"   🛑 早停触发! 最佳epoch: {self.best_epoch}")
                return True

            return False

    def reset(self):
        """重置计数器"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class AverageMeter:
    """平均值计算器"""

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self):
        """重置所有统计量"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """更新统计量"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricsTracker:
    """指标追踪器"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def get_metric(self, key: str) -> List[float]:
        """获取指定指标的历史记录"""
        return self.metrics.get(key, [])

    def get_best(self, key: str, mode: str = 'min') -> tuple:
        """
        获取最佳指标值和对应的索引

        Returns:
            (best_value, best_index)
        """
        values = self.get_metric(key)
        if not values:
            return None, None

        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        return values[best_idx], best_idx

    def get_latest(self, key: str, n: int = 1) -> List[float]:
        """获取最近n个值"""
        values = self.get_metric(key)
        return values[-n:] if values else []

    def summary(self) -> Dict[str, Dict[str, float]]:
        """生成摘要统计"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1]
                }
        return summary

    def to_dict(self) -> Dict[str, List[float]]:
        """转换为字典"""
        return dict(self.metrics)


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    设置随机种子以确保结果可复现

    Args:
        seed: 随机种子
        deterministic: 是否启用确定性算法
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"⚠️ deterministic algorithms enable failed: {e}")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"🌱 随机种子已设置: {seed} | deterministic={deterministic}")


def seed_worker(worker_id: int):
    """为 DataLoader worker 设置确定性随机种子。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def create_torch_generator(seed: int) -> torch.Generator:
    """创建带固定种子的 torch.Generator。"""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def make_json_serializable(obj: Any) -> Any:
    """
    递归转换对象为 JSON 可序列化类型

    Args:
        obj: 要转换的对象

    Returns:
        JSON可序列化的对象
    """
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 4):
    """
    保存数据为JSON格式

    Args:
        data: 要保存的数据
        filepath: 保存路径
        indent: 缩进空格数
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 转换为可序列化格式
    serializable_data = make_json_serializable(data)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=indent, ensure_ascii=False)

    print(f"   ✅ JSON已保存: {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    从JSON文件加载数据

    Args:
        filepath: 文件路径

    Returns:
        加载的数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_pickle(data: Any, filepath: Union[str, Path]):
    """
    保存数据为Pickle格式

    Args:
        data: 要保存的数据
        filepath: 保存路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"   ✅ Pickle已保存: {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    从Pickle文件加载数据

    Args:
        filepath: 文件路径

    Returns:
        加载的数据
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def ensure_dir(dirpath: Union[str, Path]) -> Path:
    """
    确保目录存在，不存在则创建

    Args:
        dirpath: 目录路径

    Returns:
        Path对象
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        参数统计信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    打印模型摘要信息

    Args:
        model: PyTorch模型
        model_name: 模型名称
    """
    params = count_parameters(model)

    print(f"\n{'=' * 60}")
    print(f"📊 {model_name} 摘要")
    print(f"{'=' * 60}")
    print(f"   总参数量: {params['total']:,}")
    print(f"   可训练参数: {params['trainable']:,}")
    print(f"   不可训练参数: {params['non_trainable']:,}")
    print(f"{'=' * 60}\n")


def get_device(use_cuda: bool = True) -> torch.device:
    """
    获取可用的计算设备

    Args:
        use_cuda: 是否使用CUDA

    Returns:
        torch.device对象
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("💻 使用CPU")

    return device


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    获取优化器当前学习率

    Args:
        optimizer: PyTorch优化器

    Returns:
        当前学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def compute_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    计算文件哈希值

    Args:
        filepath: 文件路径
        algorithm: 哈希算法 ('md5', 'sha256')

    Returns:
        哈希值字符串
    """
    hash_func = hashlib.new(algorithm)

    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def format_time(seconds: float) -> str:
    """
    格式化时间显示

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_timestamp(fmt: str = '%Y%m%d_%H%M%S') -> str:
    """
    获取当前时间戳

    Args:
        fmt: 时间格式

    Returns:
        时间戳字符串
    """
    return datetime.now().strftime(fmt)


class Timer:
    """计时器"""

    def __init__(self, name: str = ''):
        self.name = name
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        print(f"⏱️  {self.name}: {format_time(self.elapsed)}")

    def start(self):
        """开始计时"""
        self.start_time = datetime.now()

    def stop(self):
        """停止计时"""
        if self.start_time:
            self.elapsed = (datetime.now() - self.start_time).total_seconds()

    def reset(self):
        """重置计时器"""
        self.start_time = None
        self.elapsed = 0


def print_section(title: str, width: int = 70, char: str = '='):
    """
    打印格式化的章节标题

    Args:
        title: 标题文本
        width: 总宽度
        char: 分隔符字符
    """
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_dict(data: Dict, indent: int = 0, max_length: int = 100):
    """
    递归打印字典内容

    Args:
        data: 要打印的字典
        indent: 缩进级别
        max_length: 最大显示长度
    """
    prefix = '  ' * indent

    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict(value, indent + 1, max_length)
        else:
            value_str = str(value)
            if len(value_str) > max_length:
                value_str = value_str[:max_length] + '...'
            print(f"{prefix}{key}: {value_str}")


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    合并配置字典（覆盖式）

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        合并后的配置
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_config(config: Dict, required_keys: List[str]) -> bool:
    """
    验证配置是否包含必需的键

    Args:
        config: 配置字典
        required_keys: 必需的键列表

    Returns:
        是否通过验证
    """
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        print(f"❌ 配置验证失败! 缺少以下键: {missing_keys}")
        return False

    print("✅ 配置验证通过")
    return True


class ProgressTracker:
    """训练进度追踪器"""

    def __init__(self, total_epochs: int, metrics: List[str] = None):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.metrics = metrics or ['loss']
        self.history = defaultdict(list)

    def update(self, epoch: int, **metrics):
        """更新进度"""
        self.current_epoch = epoch
        for key, value in metrics.items():
            self.history[key].append(value)

    def print_progress(self, **current_metrics):
        """打印当前进度"""
        progress = (self.current_epoch / self.total_epochs) * 100
        print(f"\n{'=' * 60}")
        print(f"📈 Epoch {self.current_epoch}/{self.total_epochs} ({progress:.1f}%)")
        print(f"{'=' * 60}")

        for key, value in current_metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

    def get_summary(self) -> Dict:
        """获取训练摘要"""
        return {
            'total_epochs': self.total_epochs,
            'completed_epochs': self.current_epoch,
            'history': dict(self.history)
        }


if __name__ == '__main__':
    print("测试辅助工具函数")

    # 测试计时器
    with Timer("测试任务"):
        import time

        time.sleep(1)

    # 测试平均值计算器
    meter = AverageMeter("Loss")
    meter.update(0.5)
    meter.update(0.3)
    print(meter)

    # 测试指标追踪器
    tracker = MetricsTracker()
    tracker.update(loss=0.5, accuracy=0.8)
    tracker.update(loss=0.3, accuracy=0.85)
    print("\n指标摘要:")
    print_dict(tracker.summary())
