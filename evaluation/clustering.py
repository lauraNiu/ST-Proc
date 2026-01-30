# evaluation/clustering.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """聚类质量评估器"""

    def __init__(self, save_dir: str = './results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
            self,
            embeddings: np.ndarray,
            pred_clusters: np.ndarray,
            true_labels: np.ndarray,
            save_confusion: bool = True
    ) -> Dict[str, float]:
        """全面评估聚类结果"""
        results = {}

        # 无监督指标
        try:
            results['silhouette'] = silhouette_score(embeddings, pred_clusters)
            results['davies_bouldin'] = davies_bouldin_score(embeddings, pred_clusters)
            logger.info(f"Silhouette Score: {results['silhouette']:.4f}")
            logger.info(f"Davies-Bouldin Index: {results['davies_bouldin']:.4f}")
        except Exception as e:
            logger.warning(f"无监督指标计算失败: {e}")
            results['silhouette'] = 0.0
            results['davies_bouldin'] = 0.0

        # 有监督指标
        labeled_mask = true_labels >= 0

        if labeled_mask.sum() > 0:
            labeled_pred = pred_clusters[labeled_mask]
            labeled_true = true_labels[labeled_mask]

            try:
                results['ari'] = adjusted_rand_score(labeled_true, labeled_pred)
                results['nmi'] = normalized_mutual_info_score(labeled_true, labeled_pred)
                results['best_accuracy'] = self._compute_best_accuracy(
                    labeled_true, labeled_pred
                )

                logger.info(f"ARI: {results['ari']:.4f}")
                logger.info(f"NMI: {results['nmi']:.4f}")
                logger.info(f"Best Accuracy: {results['best_accuracy']:.4f}")

                if save_confusion:
                    self._plot_confusion_matrix(labeled_true, labeled_pred)

            except Exception as e:
                logger.warning(f"有监督指标计算失败: {e}")
                results['ari'] = 0.0
                results['nmi'] = 0.0
                results['best_accuracy'] = 0.0

        results['n_clusters'] = len(np.unique(pred_clusters))
        results['n_labeled'] = int(labeled_mask.sum())
        results['n_samples'] = len(pred_clusters)

        return results

    def _compute_best_accuracy(
            self,
            true_labels: np.ndarray,
            pred_clusters: np.ndarray
    ) -> float:
        """使用匈牙利算法计算最佳聚类准确率"""
        cm = confusion_matrix(true_labels, pred_clusters)
        row_ind, col_ind = linear_sum_assignment(-cm)
        accuracy = cm[row_ind, col_ind].sum() / cm.sum()
        return accuracy

    def _plot_confusion_matrix(
            self,
            true_labels: np.ndarray,
            pred_clusters: np.ndarray
    ):
        """绘制并保存混淆矩阵"""
        cm = confusion_matrix(true_labels, pred_clusters)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Count'}
        )
        plt.title('Clustering Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Cluster', fontsize=12)
        plt.tight_layout()

        save_path = self.save_dir / 'confusion_matrix_clustering.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"混淆矩阵已保存: {save_path}")


class ClusterLabelMapper:
    """簇到标签的映射器"""

    def __init__(self, num_classes: int = 11):
        self.num_classes = num_classes

    def map_clusters_to_labels(
            self,
            clusters: np.ndarray,
            true_labels: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """使用匈牙利算法将簇映射到真实标签"""
        labeled_mask = true_labels >= 0

        if labeled_mask.sum() == 0:
            logger.warning("没有标签数据，无法进行簇-标签映射")
            return clusters, {}

        labeled_clusters = clusters[labeled_mask]
        labeled_true = true_labels[labeled_mask]

        unique_clusters = np.unique(clusters)
        cost_matrix = np.zeros((len(unique_clusters), self.num_classes))

        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = labeled_clusters == cluster_id

            if cluster_mask.sum() == 0:
                continue

            cluster_labels = labeled_true[cluster_mask]

            for label_id in range(self.num_classes):
                cost_matrix[i, label_id] = -np.sum(cluster_labels == label_id)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        cluster_to_label = {}
        for cluster_idx, label_id in zip(row_ind, col_ind):
            cluster_id = unique_clusters[cluster_idx]
            cluster_to_label[cluster_id] = label_id

        mapped_labels = np.array([
            cluster_to_label.get(c, -1) for c in clusters
        ])

        correctly_mapped = 0
        total_labeled = labeled_mask.sum()

        for cluster_id, label_id in cluster_to_label.items():
            mask = (clusters == cluster_id) & (true_labels == label_id)
            correctly_mapped += mask.sum()

        mapping_accuracy = correctly_mapped / total_labeled if total_labeled > 0 else 0

        logger.info(f"簇-标签映射完成:")
        logger.info(f"  映射准确率: {mapping_accuracy:.2%}")
        logger.info(f"  映射数量: {len(cluster_to_label)}")

        return mapped_labels, cluster_to_label


class HierarchicalClusterRefiner:
    """✅ 修复：层次聚类优化器 - 防止过度分裂"""

    def __init__(
            self,
            n_sub_clusters: int = 3,  # ✅ 从6降低到3
            min_cluster_size: int = 50,  # ✅ 从20提高到50
            max_refinement_depth: int = 1  # ✅ 新增：限制细化深度
    ):
        """
        Args:
            n_sub_clusters: 每个簇内的子簇数量（降低以减少过度分裂）
            min_cluster_size: 进行细化的最小簇大小（提高以避免小簇分裂）
            max_refinement_depth: 最大细化深度（防止递归过深）
        """
        self.n_sub_clusters = n_sub_clusters
        self.min_cluster_size = min_cluster_size
        self.max_refinement_depth = max_refinement_depth

    def refine(
            self,
            embeddings: np.ndarray,
            initial_clusters: np.ndarray,
            depth: int = 0  # ✅ 新增：跟踪递归深度
    ) -> np.ndarray:
        """
        对初始聚类结果进行层次化优化

        Args:
            embeddings: 样本嵌入
            initial_clusters: 初始簇标签
            depth: 当前递归深度

        Returns:
            优化后的簇标签
        """
        # ✅ 检查递归深度
        if depth >= self.max_refinement_depth:
            logger.info(f"   达到最大细化深度 {depth}，停止细化")
            return initial_clusters

        refined_clusters = initial_clusters.copy()
        unique_clusters = np.unique(initial_clusters)

        refined_count = 0

        for cluster_id in unique_clusters:
            cluster_mask = initial_clusters == cluster_id
            cluster_embeddings = embeddings[cluster_mask]

            # ✅ 跳过小簇
            if len(cluster_embeddings) < self.min_cluster_size:
                logger.debug(f"   跳过小簇 {cluster_id} (size={len(cluster_embeddings)})")
                continue

            # ✅ 动态确定子簇数量（避免过度分裂）
            n_subs = min(
                self.n_sub_clusters,
                max(2, len(cluster_embeddings) // 30)  # ✅ 从10改为30
            )

            if n_subs < 2:
                continue

            # ✅ 使用轮廓系数评估是否需要细化
            try:
                from sklearn.metrics import silhouette_score

                # 尝试细化
                sub_clustering = AgglomerativeClustering(
                    n_clusters=n_subs,
                    linkage='ward'
                )
                sub_labels = sub_clustering.fit_predict(cluster_embeddings)

                # 计算细化后的轮廓系数
                if len(np.unique(sub_labels)) > 1:
                    sub_silhouette = silhouette_score(cluster_embeddings, sub_labels)

                    # ✅ 只有轮廓系数提升才进行细化
                    if sub_silhouette > 0.1:  # ✅ 设置阈值
                        unique_subs = np.unique(sub_labels)
                        for i, sub_id in enumerate(unique_subs):
                            sub_mask = sub_labels == sub_id
                            global_indices = np.where(cluster_mask)[0][sub_mask]

                            # ✅ 使用更简单的编号方案
                            new_cluster_id = len(np.unique(refined_clusters)) + i
                            refined_clusters[global_indices] = new_cluster_id

                        refined_count += 1
                        logger.debug(f"   细化簇 {cluster_id} → {n_subs}个子簇 (silhouette={sub_silhouette:.3f})")
                    else:
                        logger.debug(f"   簇 {cluster_id} 轮廓系数未提升，跳过细化")

            except Exception as e:
                logger.warning(f"   簇 {cluster_id} 细化失败: {e}")
                continue

        logger.info(f"层次优化完成 (深度{depth}):")
        logger.info(f"  原始簇数: {len(unique_clusters)}")
        logger.info(f"  优化后簇数: {len(np.unique(refined_clusters))}")
        logger.info(f"  细化簇数: {refined_count}")

        return refined_clusters


def perform_clustering(
        embeddings: np.ndarray,
        n_clusters: int = 11,
        method: str = 'kmeans',  # ✅ 新增：支持多种聚类方法
        refine: bool = True,  # ✅ 新增：是否进行层次优化
        logger=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ✅ 改进：执行聚类分析

    Args:
        embeddings: 嵌入向量
        n_clusters: 簇数量
        method: 聚类方法 ('kmeans', 'hierarchical', 'dbscan')
        refine: 是否进行层次优化
        logger: 日志记录器

    Returns:
        (初始聚类, 优化后的聚类)
    """
    if logger:
        logger.info(f"🔍 开始聚类 (method={method}, n_clusters={n_clusters})")

    # ✅ 支持多种聚类方法
    if method == 'kmeans':
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,  # ✅ 增加初始化次数
            max_iter=500  # ✅ 增加最大迭代次数
        )
        initial_clusters = clusterer.fit_predict(embeddings)

    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        initial_clusters = clusterer.fit_predict(embeddings)

    elif method == 'dbscan':
        # ✅ DBSCAN自动确定簇数量
        from sklearn.neighbors import NearestNeighbors

        # 使用k-distance图确定eps
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        distances = np.sort(distances[:, -1])
        eps = np.percentile(distances, 90)

        clusterer = DBSCAN(eps=eps, min_samples=k)
        initial_clusters = clusterer.fit_predict(embeddings)

        if logger:
            logger.info(f"   DBSCAN自动检测到 {len(np.unique(initial_clusters))} 个簇")

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # ✅ 可选的层次优化
    if refine and method != 'dbscan':
        refiner = HierarchicalClusterRefiner(
            n_sub_clusters=3,
            min_cluster_size=50,
            max_refinement_depth=1
        )
        refined_clusters = refiner.refine(embeddings, initial_clusters)
    else:
        refined_clusters = initial_clusters

    if logger:
        logger.info(f"✅ 聚类完成: {len(np.unique(refined_clusters))} 个簇")

    return initial_clusters, refined_clusters
