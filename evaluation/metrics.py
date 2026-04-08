"""
评估指标模块
提供完整的评估指标和可解释性分析
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score
)
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple  # 确保有 Optional

logger = logging.getLogger(__name__)


class TransportModeEvaluator:
    """交通方式识别评估器"""

    def __init__(self, label_names: Dict[int, str], save_dir: str = './results'):
        """
        初始化评估器

        Args:
            label_names: 标签ID到名称的映射
            save_dir: 结果保存目录
        """
        self.label_names = label_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
            self,
            pred_labels: np.ndarray,
            true_labels: np.ndarray,
            save_prefix: str = '',
            return_detailed: bool = True,
            selected_metrics: Optional[List[str]] = None  # 新增参数
    ) -> Dict:
        """
        全面评估交通方式识别性能

        Args:
            pred_labels: 预测标签 [N]
            true_labels: 真实标签 [N]，-1表示无标签
            save_prefix: 保存文件的前缀
            return_detailed: 是否返回详细结果

        Returns:
            评估指标字典
        """
        # 过滤出有标签的样本
        labeled_mask = true_labels >= 0

        if labeled_mask.sum() == 0:
            logger.warning("⚠️  没有标签数据进行评估")
            return {}

        y_true = true_labels[labeled_mask]
        y_pred = pred_labels[labeled_mask]

        # 过滤出有效预测
        valid_mask = y_pred >= 0
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            logger.warning("⚠️  没有有效的预测结果")
            return {}

        # ==================== 计算基本指标 ====================
        results = {}

        # 1. 准确率
        results['accuracy'] = accuracy_score(y_true, y_pred)

        # 2. 平衡准确率（对不平衡数据集更有意义）
        results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        # 3. Cohen's Kappa（考虑随机一致性）
        results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # 4. 各种平均的F1分数
        results['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        results['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # 5. 各种平均的精确率和召回率
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        results['macro_precision'] = np.mean(precision)
        results['macro_recall'] = np.mean(recall)
        results['weighted_precision'] = np.average(precision, weights=support)
        results['weighted_recall'] = np.average(recall, weights=support)

        # ==================== 按类别统计 ====================
        present_labels = sorted(np.unique(y_true))
        results['per_class'] = {}
        results['per_class_detailed'] = {}

        for label_id in present_labels:
            label_name = self.label_names.get(label_id, f'class_{label_id}')

            # 找到该类别在precision等数组中的索引
            idx = np.where(np.unique(y_true) == label_id)[0]
            if len(idx) == 0:
                continue
            idx = idx[0]

            # 基本指标
            results['per_class'][label_name] = {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1': float(f1[idx]),
                'support': int(support[idx])
            }

            # 详细统计
            true_positive = ((y_true == label_id) & (y_pred == label_id)).sum()
            false_positive = ((y_true != label_id) & (y_pred == label_id)).sum()
            false_negative = ((y_true == label_id) & (y_pred != label_id)).sum()
            true_negative = ((y_true != label_id) & (y_pred != label_id)).sum()

            results['per_class_detailed'][label_name] = {
                'true_positive': int(true_positive),
                'false_positive': int(false_positive),
                'false_negative': int(false_negative),
                'true_negative': int(true_negative),
                'specificity': float(true_negative / (true_negative + false_positive))
                               if (true_negative + false_positive) > 0 else 0.0
            }
        # 计算完 results 后，新增核心指标 summary
        if selected_metrics:
            results['summary'] = {k: results[k] for k in selected_metrics if k in results}

        # 打印时高亮核心指标
        self._print_results(results, y_true, y_pred, present_labels, selected_metrics=selected_metrics)

        # 保存核心指标（独立小文件）
        if selected_metrics:
            summary_path = self.save_dir / f'{save_prefix}final_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(results['summary'], f, indent=4, ensure_ascii=False)
            logger.info(f"   ✅ 核心指标已保存: {summary_path.name}")

        # ==================== 保存可视化 ====================
        target_names = [self.label_names.get(i, f'class_{i}') for i in present_labels]

        # 1. 混淆矩阵
        self._plot_confusion_matrix(y_true, y_pred, target_names, save_prefix)

        # 2. 性能对比图
        self._plot_performance_comparison(results['per_class'], save_prefix)

        # 3. 详细报告
        if return_detailed:
            self._save_detailed_report(results, y_true, y_pred, target_names, save_prefix)

        return results

    def _print_results(self, results: Dict, y_true: np.ndarray,
                      y_pred: np.ndarray, present_labels: List,
                      selected_metrics: Optional[List[str]] = None):  # 新增参数
        logger.info(f"\n{'=' * 80}")
        logger.info("📊 交通方式识别评估结果")
        logger.info(f"{'=' * 80}")

        # 若有核心指标，先高亮打印
        if selected_metrics and 'summary' in results:
            logger.info("\n【核心指标】")
            for k in selected_metrics:
                if k in results['summary']:
                    v = results['summary'][k]
                    if isinstance(v, float):
                        logger.info(f"  {k}: {v:.4f} ({v*100:.2f}%)" if k in ('accuracy','balanced_accuracy') else f"  {k}: {v:.4f}")
            logger.info('-' * 80)

        # 总体指标
        logger.info(f"\n【总体性能指标】")
        logger.info(f"  准确率 (Accuracy):              {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        logger.info(f"  平衡准确率 (Balanced Accuracy): {results['balanced_accuracy']:.4f} ({results['balanced_accuracy']*100:.2f}%)")
        logger.info(f"  Cohen's Kappa:                  {results['cohen_kappa']:.4f}")

        logger.info(f"\n【F1分数】")
        logger.info(f"  宏平均 F1 (Macro F1):           {results['macro_f1']:.4f}")
        logger.info(f"  加权平均 F1 (Weighted F1):      {results['weighted_f1']:.4f}")
        logger.info(f"  微平均 F1 (Micro F1):           {results['micro_f1']:.4f}")

        logger.info(f"\n【精确率和召回率】")
        logger.info(f"  宏平均精确率:                   {results['macro_precision']:.4f}")
        logger.info(f"  宏平均召回率:                   {results['macro_recall']:.4f}")
        logger.info(f"  加权平均精确率:                 {results['weighted_precision']:.4f}")
        logger.info(f"  加权平均召回率:                 {results['weighted_recall']:.4f}")

        # 按类别详细报告
        logger.info(f"\n{'=' * 80}")
        logger.info("【各类别详细性能】")
        logger.info(f"{'=' * 80}")
        logger.info(f"{'类别':<12} {'支持数':<8} {'精确率':<10} {'召回率':<10} {'F1':<10} {'特异性':<10}")
        logger.info(f"{'-' * 80}")

        for label_name in sorted(results['per_class'].keys()):
            metrics = results['per_class'][label_name]
            detailed = results['per_class_detailed'][label_name]

            logger.info(
                f"{label_name:<12} "
                f"{metrics['support']:<8} "
                f"{metrics['precision']:<10.4f} "
                f"{metrics['recall']:<10.4f} "
                f"{metrics['f1']:<10.4f} "
                f"{detailed['specificity']:<10.4f}"
            )

        logger.info(f"{'=' * 80}\n")

    def _plot_confusion_matrix(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            target_names: List[str],
            save_prefix: str
    ):
        """绘制并保存混淆矩阵（包含归一化和非归一化版本）"""
        cm = confusion_matrix(y_true, y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # 非归一化混淆矩阵
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': 'Count'},
            ax=axes[0]
        )
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)

        # 归一化混淆矩阵（按行归一化）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': 'Percentage'},
            ax=axes[1]
        )
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)

        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

        plt.tight_layout()

        filename = f'{save_prefix}confusion_matrix_complete.png'
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"   ✅ 混淆矩阵已保存: {save_path.name}")

    def _plot_performance_comparison(self, per_class_metrics: Dict, save_prefix: str):
        """绘制各类别性能对比图"""
        labels = list(per_class_metrics.keys())
        precision = [per_class_metrics[l]['precision'] for l in labels]
        recall = [per_class_metrics[l]['recall'] for l in labels]
        f1 = [per_class_metrics[l]['f1'] for l in labels]
        support = [per_class_metrics[l]['support'] for l in labels]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        x = np.arange(len(labels))
        width = 0.25

        # 1. Precision, Recall, F1对比
        ax = axes[0, 0]
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        ax.set_xlabel('Transport Mode', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Precision, Recall, F1-Score Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])

        # 2. 样本支持数
        ax = axes[0, 1]
        bars = ax.bar(x, support, alpha=0.8, color='skyblue', edgecolor='black')
        ax.set_xlabel('Transport Mode', fontsize=11)
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title('Sample Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # 在柱子上添加数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # 3. F1分数排名
        ax = axes[1, 0]
        sorted_indices = np.argsort(f1)
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_f1 = [f1[i] for i in sorted_indices]

        colors = ['red' if s < 0.5 else 'orange' if s < 0.7 else 'green' for s in sorted_f1]
        ax.barh(range(len(sorted_labels)), sorted_f1, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels)
        ax.set_xlabel('F1-Score', fontsize=11)
        ax.set_title('F1-Score Ranking (Low to High)', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim([0, 1.05])

        # 添加分数标签
        for i, v in enumerate(sorted_f1):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)

        # 4. Precision vs Recall 散点图
        ax = axes[1, 1]
        scatter = ax.scatter(recall, precision, s=[s*2 for s in support],
                           alpha=0.6, c=f1, cmap='RdYlGn', edgecolors='black', linewidth=1)

        # 添加标签
        for i, label in enumerate(labels):
            ax.annotate(label, (recall[i], precision[i]),
                       fontsize=8, ha='center', va='bottom')

        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('Precision vs Recall (Size=Support, Color=F1)',
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])

        # 添加对角线
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='F1-Score')

        plt.tight_layout()

        filename = f'{save_prefix}performance_comparison.png'
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"   ✅ 性能对比图已保存: {save_path.name}")

    def _save_detailed_report(self, results: Dict, y_true: np.ndarray,
                            y_pred: np.ndarray, target_names: List[str],
                            save_prefix: str):
        """保存详细的评估报告为CSV和JSON"""
        # 1. CSV格式的分类报告
        report_data = []
        for label_name in sorted(results['per_class'].keys()):
            metrics = results['per_class'][label_name]
            detailed = results['per_class_detailed'][label_name]

            report_data.append({
                'Transport Mode': label_name,
                'Support': metrics['support'],
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'Specificity': f"{detailed['specificity']:.4f}",
                'True Positive': detailed['true_positive'],
                'False Positive': detailed['false_positive'],
                'False Negative': detailed['false_negative'],
                'True Negative': detailed['true_negative']
            })

        df = pd.DataFrame(report_data)
        csv_path = self.save_dir / f'{save_prefix}classification_report.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"   ✅ 分类报告已保存: {csv_path.name}")

        # 2. JSON格式的完整结果
        import json
        json_path = self.save_dir / f'{save_prefix}evaluation_metrics.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"   ✅ 评估指标已保存: {json_path.name}")


class MetricsCalculator:
    """通用指标计算器"""

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        计算所有常用指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            指标字典
        """
        metrics = {}

        # 基本指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        # F1分数
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # Precision和Recall
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['macro_precision'] = precision
        metrics['macro_recall'] = recall

        return metrics

    @staticmethod
    def calculate_purity(clusters: np.ndarray, labels: np.ndarray) -> float:
        """计算聚类纯度"""
        total_correct = 0
        total_samples = 0

        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_labels = labels[cluster_mask]
            cluster_labels = cluster_labels[cluster_labels >= 0]

            if len(cluster_labels) > 0:
                label_counts = np.bincount(cluster_labels)
                most_common_count = label_counts.max()
                total_correct += most_common_count
                total_samples += len(cluster_labels)

        purity = total_correct / total_samples if total_samples > 0 else 0
        return purity
