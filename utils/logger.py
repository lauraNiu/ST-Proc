import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
from collections import defaultdict
import torch


class ExperimentLogger:
    """实验日志管理器
    
    功能:
    1. 统一的日志输出格式
    2. 多级别日志记录 (DEBUG, INFO, WARNING, ERROR)
    3. 文件和控制台双输出
    4. 实验指标追踪
    5. 训练过程可视化
    """
    
    def __init__(
        self,
        exp_name: str,
        log_dir: str = "./experiments",
        level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True
    ):
        """初始化日志器
        
        Args:
            exp_name: 实验名称
            log_dir: 日志目录
            level: 日志级别
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
        """
        self.exp_name = exp_name
        self.log_dir = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化日志器
        self.logger = self._setup_logger(level, console_output, file_output)
        
        # 指标追踪
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
        self.info(f"🚀 实验开始: {exp_name}")
        self.info(f"📁 日志目录: {self.log_dir}")
    
    def _setup_logger(
        self,
        level: int,
        console_output: bool,
        file_output: bool
    ) -> logging.Logger:
        """配置日志器"""
        logger = logging.getLogger(self.exp_name)
        logger.setLevel(level)
        logger.handlers = []  # 清空已有handlers
        
        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 文件输出
        if file_output:
            log_file = self.log_dir / f"log_{self.timestamp}.txt"
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    # ==================== 基础日志方法 ====================
    
    def debug(self, message: str):
        """DEBUG级别日志"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """ERROR级别日志"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """CRITICAL级别日志"""
        self.logger.critical(message)
    
    # ==================== 格式化日志方法 ====================
    
    def log_section(self, title: str, width: int = 70):
        """打印分隔线标题"""
        self.info("=" * width)
        self.info(f"{title:^{width}}")
        self.info("=" * width)
    
    def log_subsection(self, title: str, width: int = 70):
        """打印子标题"""
        self.info("-" * width)
        self.info(title)
        self.info("-" * width)
    
    def log_dict(self, data: dict, title: Optional[str] = None, indent: int = 3):
        """格式化打印字典"""
        if title:
            self.info(title)
        
        for key, value in data.items():
            if isinstance(value, float):
                self.info(f"{' ' * indent}{key}: {value:.6f}")
            elif isinstance(value, dict):
                self.info(f"{' ' * indent}{key}:")
                self.log_dict(value, indent=indent + 3)
            else:
                self.info(f"{' ' * indent}{key}: {value}")
    
    def log_config(self, config: dict):
        """记录配置信息"""
        self.log_section("实验配置")
        self.log_dict(config)
        
        # 保存配置到文件
        config_file = self.log_dir / f"config_{self.timestamp}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        self.info(f"📄 配置已保存: {config_file}")
    
    # ==================== 训练过程日志 ====================
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.current_epoch = epoch
        self.info("")
        self.info("=" * 70)
        self.info(f"📅 Epoch {epoch}/{total_epochs}")
        self.info("=" * 70)
    
    def log_epoch_end(self, metrics: dict):
        """记录epoch结束"""
        self.info("")
        self.info(f"📊 Epoch {self.current_epoch} 结果:")
        self.log_dict(metrics)
        
        # 保存指标
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        # 更新最佳指标
        self._update_best_metrics(metrics)
    
    def log_step(self, step: int, metrics: dict, prefix: str = ""):
        """记录训练步骤"""
        self.global_step = step
        
        metrics_str = " | ".join([
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        ])
        
        message = f"{prefix}Step {step} | {metrics_str}"
        self.info(message)
    
    def log_loss(
        self,
        loss: float,
        loss_type: str = "total",
        phase: str = "train"
    ):
        """记录损失值"""
        key = f"{phase}_{loss_type}_loss"
        self.metrics[key].append(loss)
        self.info(f"{phase.capitalize()} {loss_type} Loss: {loss:.6f}")
    
    # ==================== 评估日志 ====================
    
    def log_evaluation(self, results: dict, title: str = "评估结果"):
        """记录评估结果"""
        self.log_section(f"📊 {title}")
        self.log_dict(results)
    
    def log_confusion_matrix(self, cm, label_names: Optional[list] = None):
        """记录混淆矩阵"""
        self.info("\n混淆矩阵:")
        
        if label_names:
            header = "      " + " ".join([f"{name:>6}" for name in label_names])
            self.info(header)
        
        for i, row in enumerate(cm):
            if label_names:
                row_str = f"{label_names[i]:>6}" + " ".join([f"{val:>6}" for val in row])
            else:
                row_str = " ".join([f"{val:>6}" for val in row])
            self.info(row_str)
    
    def log_best_metrics(self):
        """记录最佳指标"""
        self.log_section("🏆 最佳指标")
        self.log_dict(self.best_metrics)
    
    # ==================== 数据统计日志 ====================
    
    def log_data_statistics(self, data_stats: dict):
        """记录数据集统计信息"""
        self.log_section("📊 数据统计")
        self.log_dict(data_stats)
    
    def log_class_distribution(self, class_counts: dict, title: str = "类别分布"):
        """记录类别分布"""
        self.info(f"\n{title}:")
        total = sum(class_counts.values())
        
        for class_name, count in sorted(class_counts.items()):
            percentage = count / total * 100
            self.info(f"   {class_name:>15}: {count:>6} ({percentage:>5.2f}%)")
        
        self.info(f"   {'Total':>15}: {total:>6}")
    
    # ==================== 模型信息日志 ====================
    
    def log_model_summary(self, model: torch.nn.Module):
        """记录模型结构摘要"""
        self.log_section("🤖 模型信息")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"   总参数量: {total_params:,}")
        self.info(f"   可训练参数: {trainable_params:,}")
        self.info(f"   冻结参数: {total_params - trainable_params:,}")
        
        # 模型结构
        self.info("\n模型结构:")
        for name, module in model.named_children():
            self.info(f"   {name}: {module.__class__.__name__}")
    
    def log_checkpoint_saved(self, path: str, metrics: Optional[dict] = None):
        """记录模型保存"""
        self.info(f"💾 模型已保存: {path}")
        if metrics:
            self.info(f"   指标: {metrics}")
    
    # ==================== 时间和进度日志 ====================
    
    def log_time(self, elapsed_time: float, stage: str = ""):
        """记录耗时"""
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
        
        if stage:
            self.info(f"⏱️  {stage} 耗时: {time_str}")
        else:
            self.info(f"⏱️  耗时: {time_str}")
    
    def log_progress(self, current: int, total: int, prefix: str = ""):
        """记录进度"""
        percentage = current / total * 100
        bar_length = 50
        filled_length = int(bar_length * current / total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        self.info(f"{prefix}[{bar}] {percentage:.1f}% ({current}/{total})")
    
    # ==================== 指标追踪 ====================
    
    def _update_best_metrics(self, metrics: dict):
        """更新最佳指标"""
        for key, value in metrics.items():
            if 'loss' in key.lower():
                # 损失越小越好
                if key not in self.best_metrics or value < self.best_metrics[key]:
                    self.best_metrics[key] = value
                    self.info(f"   🎯 新的最佳 {key}: {value:.6f}")
            else:
                # 其他指标越大越好
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = value
                    self.info(f"   🎯 新的最佳 {key}: {value:.6f}")
    
    def get_metric_history(self, metric_name: str) -> list:
        """获取指标历史"""
        return self.metrics.get(metric_name, [])
    
    def save_metrics(self):
        """保存所有指标到文件"""
        metrics_file = self.log_dir / f"metrics_{self.timestamp}.json"
        
        metrics_data = {
            'history': dict(self.metrics),
            'best': self.best_metrics,
            'final_epoch': self.current_epoch
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=4, ensure_ascii=False)
        
        self.info(f"📊 指标已保存: {metrics_file}")
    
    # ==================== 异常处理 ====================
    
    def log_exception(self, exception: Exception, context: str = ""):
        """记录异常"""
        import traceback
        
        self.error("=" * 70)
        self.error(f"❌ 异常发生{f' ({context})' if context else ''}")
        self.error(f"异常类型: {type(exception).__name__}")
        self.error(f"异常信息: {str(exception)}")
        self.error("\n堆栈跟踪:")
        self.error(traceback.format_exc())
        self.error("=" * 70)
    
    # ==================== 实验结束 ====================
    
    def close(self):
        """关闭日志器"""
        self.log_section("✅ 实验结束")
        self.log_best_metrics()
        self.save_metrics()
        
        # 关闭所有handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        if exc_type is not None:
            self.log_exception(exc_val, "实验执行")
        self.close()
        return False


class MetricsTracker:
    """轻量级指标追踪器
    
    用于追踪训练过程中的关键指标
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.best_metrics = {}
    
    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
            
            # 自动追踪最佳值
            if 'loss' in key.lower():
                if key not in self.best_metrics or value < self.best_metrics[key]:
                    self.best_metrics[key] = value
            else:
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = value
    
    def get_latest(self, key: str) -> Optional[float]:
        """获取最新值"""
        return self.metrics[key][-1] if key in self.metrics else None
    
    def get_best(self, key: str) -> Optional[float]:
        """获取最佳值"""
        return self.best_metrics.get(key)
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> Optional[float]:
        """计算平均值"""
        if key not in self.metrics:
            return None
        
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        
        return sum(values) / len(values) if values else None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'history': dict(self.metrics),
            'best': self.best_metrics
        }



def get_logger(
    exp_name: str,
    log_dir: str = "./experiments",
    level: int = logging.INFO
) -> ExperimentLogger:
    """获取实验日志器（便捷函数）"""
    return ExperimentLogger(exp_name, log_dir, level)


def setup_basic_logger(name: str = "trajectory") -> logging.Logger:
    """设置基础日志器（用于简单场景）"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# ==================== 使用示例 ====================

if __name__ == "__main__":

    with get_logger("test_experiment") as logger:
        logger.log_config({
            'model': 'ResNet50',
            'batch_size': 32,
            'lr': 0.001
        })
        
        logger.log_section("训练开始")
        
        for epoch in range(1, 4):
            logger.log_epoch_start(epoch, 3)
            
            # 训练
            train_loss = 0.5 - epoch * 0.1
            logger.log_loss(train_loss, loss_type="total", phase="train")
            
            # 验证
            val_loss = 0.6 - epoch * 0.1
            logger.log_epoch_end({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': 0.7 + epoch * 0.05
            })

    tracker = MetricsTracker()
    
    for i in range(10):
        tracker.update(
            loss=1.0 - i * 0.05,
            accuracy=0.5 + i * 0.03
        )
    
    print(f"最佳准确率: {tracker.get_best('accuracy'):.4f}")
    print(f"平均损失(最后5轮): {tracker.get_average('loss', last_n=5):.4f}")
