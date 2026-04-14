"""
Unified Visualization Module
Integrates all visualization functionalities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class UnifiedVisualizer:
    """Unified Visualizer - Integrates all visualization functionalities"""

    def __init__(self, save_dir: str = './results', style='seaborn-v0_8-darkgrid',
                 figsize=(12, 8), dpi=300):
        """
        Args:
            save_dir: Results save directory
            style: matplotlib style
            figsize: Default figure size
            dpi: Figure resolution
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.style = style
        self.figsize = figsize
        self.dpi = dpi

        self._setup_style()

    def _setup_style(self):
        """Setup plotting style"""
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')

        # Set font for better English support
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = self.dpi

    def _save_and_close(self, fig, filename: str):
        """Save and close figure"""
        save_path = self.save_dir / filename
        fig.tight_layout()
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {save_path.name}")
        return save_path

    # ==================== 1. Trajectory Visualization ====================

    def plot_single_trajectory(self, coords: np.ndarray, label: Optional[int] = None,
                               label_name: Optional[str] = None,
                               filename: str = 'single_trajectory.png'):
        """Plot a single trajectory"""
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(coords[:, 1], coords[:, 0], 'b-', linewidth=2, alpha=0.6)
        ax.scatter(coords[0, 1], coords[0, 0], c='green', s=100,
                   marker='o', label='Start', zorder=5)
        ax.scatter(coords[-1, 1], coords[-1, 0], c='red', s=100,
                   marker='s', label='End', zorder=5)

        title = 'Trajectory Visualization'
        if label_name:
            title += f' - {label_name}'
        elif label is not None:
            title += f' - Label {label}'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

        return self._save_and_close(fig, filename)

    def plot_multiple_trajectories(self, trajectories: List[Dict],
                                   max_trajectories: int = 9,
                                   filename: str = 'multiple_trajectories.png'):
        """Plot multiple trajectories (grid layout)"""
        n_trajs = min(len(trajectories), max_trajectories)
        n_cols = 3
        n_rows = (n_trajs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for i, (ax, traj) in enumerate(zip(axes[:n_trajs], trajectories[:n_trajs])):
            coords = traj['raw_coords'][:traj['original_length']]

            ax.plot(coords[:, 1], coords[:, 0], 'b-', linewidth=1.5, alpha=0.6)
            ax.scatter(coords[0, 1], coords[0, 0], c='green', s=50, marker='o')
            ax.scatter(coords[-1, 1], coords[-1, 0], c='red', s=50, marker='s')

            title = f"Trajectory {i + 1}"
            if traj.get('mode_name'):
                title += f" - {traj['mode_name']}"

            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Longitude', fontsize=9)
            ax.set_ylabel('Latitude', fontsize=9)
            ax.grid(alpha=0.3)

        for ax in axes[n_trajs:]:
            ax.axis('off')

        return self._save_and_close(fig, filename)

    # ==================== 2. Embedding Visualization ====================

    def plot_embeddings_2d(self, embeddings: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          label_names: Optional[Dict] = None,
                          method: str = 'tsne',
                          title: str = 'Embedding Visualization',
                          filename: str = 'embeddings_2d.png'):
        """2D embedding visualization (t-SNE or PCA)"""
        print(f"   🔄 Performing {method.upper()} dimensionality reduction...")

        # Dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")

        fig, ax = plt.subplots(figsize=self.figsize)

        if labels is not None:
            unique_labels = np.unique(labels[labels >= 0])
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.6,
                s=50
            )

            # Legend
            if label_names:
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=plt.cm.tab10(i / 10),
                               markersize=8,
                               label=label_names.get(label, f'Label {label}'))
                    for i, label in enumerate(unique_labels)
                ]
                ax.legend(handles=legend_elements, loc='best', fontsize=9)
            else:
                plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)

        ax.set_title(f'{title} ({method.upper()})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.grid(alpha=0.3)

        return self._save_and_close(fig, filename)

    def plot_two_embeddings(self,
                            embeddings_1: np.ndarray,
                            embeddings_2: np.ndarray,
                            labels: Optional[np.ndarray] = None,
                            label_names: Optional[Dict] = None,
                            titles: Optional[List[str]] = None,
                            method: str = 'tsne',
                            filename: str = 'space_comparison.png'):
        """Side-by-side comparison of two embedding spaces."""

        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        if titles is None:
            titles = ["Space 1", "Space 2"]

        # 降维
        if method.lower() == 'tsne':
            reducer1 = TSNE(n_components=2, random_state=42, perplexity=30)
            reducer2 = TSNE(n_components=2, random_state=42, perplexity=30)
            emb2d_1 = reducer1.fit_transform(embeddings_1)
            emb2d_2 = reducer2.fit_transform(embeddings_2)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            emb2d_1 = reducer.fit_transform(embeddings_1)
            emb2d_2 = reducer.fit_transform(embeddings_2)
        else:
            raise ValueError(f"Unknown method: {method}")

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        if labels is not None:
            unique_l = np.unique(labels[labels >= 0]) if (labels >= 0).any() else np.unique(labels)
            sc1 = axes[0].scatter(emb2d_1[:, 0], emb2d_1[:, 1], c=labels, cmap='tab10', alpha=0.6, s=50)
            if label_names is not None and (labels >= 0).any():
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=plt.cm.tab10(i / 10),
                               markersize=8,
                               label=label_names.get(l, f'Label {l}'))
                    for i, l in enumerate(unique_l)
                ]
                axes[0].legend(handles=legend_elements, loc='best', fontsize=9)
        else:
            sc1 = axes[0].scatter(emb2d_1[:, 0], emb2d_1[:, 1], alpha=0.6, s=50)
        axes[0].set_title(titles[0], fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Dimension 1');
        axes[0].set_ylabel('Dimension 2');
        axes[0].grid(alpha=0.3)

        if labels is not None:
            unique_l = np.unique(labels[labels >= 0]) if (labels >= 0).any() else np.unique(labels)
            sc2 = axes[1].scatter(emb2d_2[:, 0], emb2d_2[:, 1], c=labels, cmap='tab10', alpha=0.6, s=50)
            if label_names is not None and (labels >= 0).any():
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=plt.cm.tab10(i / 10),
                               markersize=8,
                               label=label_names.get(l, f'Label {l}'))
                    for i, l in enumerate(unique_l)
                ]
                axes[1].legend(handles=legend_elements, loc='best', fontsize=9)
        else:
            sc2 = axes[1].scatter(emb2d_2[:, 0], emb2d_2[:, 1], alpha=0.6, s=50)
        axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Dimension 1');
        axes[1].set_ylabel('Dimension 2');
        axes[1].grid(alpha=0.3)

        return self._save_and_close(fig, filename)

    def plot_embeddings_comparison(self, embeddings: np.ndarray,
                                   pred_labels: np.ndarray,
                                   true_labels: Optional[np.ndarray] = None,
                                   label_names: Optional[Dict] = None,
                                   filename: str = 'embeddings_comparison.png'):
        """Compare embeddings visualization of predicted and true labels"""
        print("   🔄 Generating comparison visualization...")

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)

        if true_labels is not None and (true_labels >= 0).sum() > 0:
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(12, 8))
            axes = [axes]

        # Left plot: Predicted labels
        valid_mask = pred_labels >= 0
        unique_pred = np.unique(pred_labels[valid_mask])

        scatter1 = axes[0].scatter(
            embeddings_2d[valid_mask, 0],
            embeddings_2d[valid_mask, 1],
            c=pred_labels[valid_mask],
            cmap='tab10',
            alpha=0.6,
            s=50
        )
        axes[0].set_title('Predicted Labels', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Dimension 1', fontsize=12)
        axes[0].set_ylabel('Dimension 2', fontsize=12)
        axes[0].grid(alpha=0.3)

        if label_names:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=plt.cm.tab10(i / 10),
                           markersize=8,
                           label=label_names.get(label, f'Label {label}'))
                for i, label in enumerate(unique_pred)
            ]
            axes[0].legend(handles=legend_elements, loc='best', fontsize=9)

        # Right plot: True labels
        if true_labels is not None and len(axes) == 2:
            labeled_mask = true_labels >= 0
            unique_true = np.unique(true_labels[labeled_mask])

            scatter2 = axes[1].scatter(
                embeddings_2d[labeled_mask, 0],
                embeddings_2d[labeled_mask, 1],
                c=true_labels[labeled_mask],
                cmap='tab10',
                alpha=0.6,
                s=50
            )
            axes[1].set_title('True Labels', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Dimension 1', fontsize=12)
            axes[1].set_ylabel('Dimension 2', fontsize=12)
            axes[1].grid(alpha=0.3)

            if label_names:
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=plt.cm.tab10(i / 10),
                               markersize=8,
                               label=label_names.get(label, f'Label {label}'))
                    for i, label in enumerate(unique_true)
                ]
                axes[1].legend(handles=legend_elements, loc='best', fontsize=9)

        return self._save_and_close(fig, filename)

    # ==================== 3. Training Curves Visualization ====================

    def plot_training_curves(self, train_losses: List[float],
                            val_losses: Optional[List[float]] = None,
                            contrast_losses: Optional[List[float]] = None,
                            proto_losses: Optional[List[float]] = None,
                            pseudo_losses: Optional[List[float]] = None,
                            filename: str = 'training_curves.png'):
        """Plot training curves"""
        # Calculate number of subplots needed
        plot_items = [
            ('Total Loss', train_losses, val_losses),
            ('Contrastive Loss', contrast_losses, None),
            ('Prototypical Loss', proto_losses, None),
            ('Pseudo-label Loss', pseudo_losses, None)
        ]

        # Filter out None items
        plot_items = [(name, train, val) for name, train, val in plot_items
                      if train is not None]

        n_plots = len(plot_items)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        colors = ['blue', 'green', 'red', 'purple']

        for idx, ((name, train_loss, val_loss), color) in enumerate(zip(plot_items, colors)):
            ax = axes[idx]

            ax.plot(train_loss, label=f'Train {name}', linewidth=2, color=color)
            if val_loss is not None:
                ax.plot(val_loss, label=f'Val {name}',
                       linewidth=2, color=color, linestyle='--', alpha=0.7)

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        # Hide extra subplots
        for ax in axes[n_plots:]:
            ax.axis('off')

        return self._save_and_close(fig, filename)

    def plot_metrics(self, metrics: Dict[str, List[float]],
                    filename: str = 'metrics_trends.png'):
        """Plot evaluation metrics trends.

        Supports either:
        1. dict[str, list[float]]  -> already grouped by metric name
        2. list[dict[str, float]]  -> one dict per epoch
        """
        if isinstance(metrics, list):
            if len(metrics) == 0:
                return None

            metric_dict = {}
            for epoch_metrics in metrics:
                if not isinstance(epoch_metrics, dict):
                    continue
                for metric_name, value in epoch_metrics.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        metric_dict.setdefault(metric_name, []).append(float(value))
            metrics = metric_dict

        if not isinstance(metrics, dict) or len(metrics) == 0:
            return None

        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for idx, (metric_name, values) in enumerate(metrics.items()):
            axes[idx].plot(values, linewidth=2, marker='o')
            axes[idx].set_xlabel('Epoch', fontsize=12)
            axes[idx].set_ylabel(metric_name, fontsize=12)
            axes[idx].set_title(metric_name, fontsize=14, fontweight='bold')
            axes[idx].grid(alpha=0.3)

        for ax in axes[n_metrics:]:
            ax.axis('off')

        return self._save_and_close(fig, filename)

    # ==================== 4. Clustering Visualization ====================

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             label_names: Optional[List[str]] = None,
                             normalize: bool = False,
                             filename: str = 'confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            cm_display = cm
        else:
            fmt = 'd'
            cm_display = cm

        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=label_names if label_names else 'auto',
                    yticklabels=label_names if label_names else 'auto',
                    cbar_kws={'label': 'Ratio' if normalize else 'Count'},
                    ax=ax)

        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''),
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)

        if label_names:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

        return self._save_and_close(fig, filename)

    def plot_cluster_distribution(self, clusters: np.ndarray,
                                  true_labels: Optional[np.ndarray] = None,
                                  label_names: Optional[Dict] = None,
                                  filename: str = 'cluster_distribution.png'):
        """Plot cluster and label distributions"""
        has_labels = true_labels is not None and (true_labels >= 0).sum() > 0

        fig, axes = plt.subplots(1, 2 if has_labels else 1,
                                figsize=(8 * (2 if has_labels else 1), 6))

        if not has_labels:
            axes = [axes]

        # Left plot: Cluster size distribution
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        axes[0].bar(range(len(unique_clusters)), cluster_counts,
                   color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Cluster ID', fontsize=12)
        axes[0].set_ylabel('Number of Samples', fontsize=12)
        axes[0].set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(len(unique_clusters)))
        axes[0].set_xticklabels(unique_clusters)
        axes[0].grid(axis='y', alpha=0.3)

        # Right plot: Label distribution
        if has_labels:
            labeled_mask = true_labels >= 0
            unique_labels, label_counts = np.unique(true_labels[labeled_mask],
                                                    return_counts=True)

            if label_names:
                labels_text = [label_names.get(l, f'Label {l}') for l in unique_labels]
            else:
                labels_text = [f'Label {l}' for l in unique_labels]

            axes[1].bar(range(len(unique_labels)), label_counts,
                       color='lightcoral', edgecolor='black')
            axes[1].set_xlabel('Transport Mode', fontsize=12)
            axes[1].set_ylabel('Number of Samples', fontsize=12)
            axes[1].set_title('Label Distribution', fontsize=14, fontweight='bold')
            axes[1].set_xticks(range(len(unique_labels)))
            axes[1].set_xticklabels(labels_text, rotation=45, ha='right')
            axes[1].grid(axis='y', alpha=0.3)

        return self._save_and_close(fig, filename)

    def plot_cluster_label_heatmap(self, clusters: np.ndarray,
                                   true_labels: np.ndarray,
                                   label_names: Optional[Dict] = None,
                                   filename: str = 'cluster_label_heatmap.png'):
        """Plot cluster-label distribution heatmap"""
        labeled_mask = true_labels >= 0

        if labeled_mask.sum() == 0:
            print("   ⚠️  No labeled data, skipping heatmap generation")
            return None

        labeled_clusters = clusters[labeled_mask]
        labeled_true = true_labels[labeled_mask]

        label_ids = np.unique(labeled_true)
        cluster_ids = np.unique(clusters)

        # Build distribution matrix
        distribution_matrix = np.zeros((len(label_ids), len(cluster_ids)))

        for i, label_id in enumerate(label_ids):
            label_mask = labeled_true == label_id
            label_clusters = labeled_clusters[label_mask]

            cluster_counts = np.bincount(label_clusters, minlength=clusters.max() + 1)
            cluster_counts_subset = cluster_counts[cluster_ids]

            total = cluster_counts_subset.sum()
            if total > 0:
                distribution_matrix[i] = cluster_counts_subset / total

        fig, ax = plt.subplots(figsize=self.figsize)

        if label_names:
            ylabels = [label_names.get(lid, f'Label {lid}') for lid in label_ids]
        else:
            ylabels = [f'Label {lid}' for lid in label_ids]

        xlabels = [f'Cluster {cid}' for cid in cluster_ids]

        sns.heatmap(
            distribution_matrix,
            annot=True,
            fmt='.2%',
            cmap='YlOrRd',
            xticklabels=xlabels,
            yticklabels=ylabels,
            cbar_kws={'label': 'Ratio'},
            ax=ax
        )

        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Transport Mode', fontsize=12)
        ax.set_title('Label Distribution in Clusters', fontsize=14, fontweight='bold')

        return self._save_and_close(fig, filename)

    # ==================== 5. Comprehensive Evaluation Report ====================

    def generate_evaluation_report(self,
                                   embeddings: np.ndarray,
                                   clusters: np.ndarray,
                                   mapped_labels: np.ndarray,
                                   true_labels: np.ndarray,
                                   label_names: Dict,
                                   cluster_results: Dict,
                                   mode_results: Dict):
        """Generate complete evaluation visualization report"""
        print("\n" + "=" * 70)
        print("📊 Generating Complete Visualization Report")
        print("=" * 70)

        # 1. Clustering visualization
        print("\n1️⃣  Clustering Results Visualization")
        self.plot_embeddings_2d(
            embeddings, clusters, None, method='tsne',
            title='Trajectory Clustering Results',
            filename='01_trajectory_clusters.png'
        )

        # 2. Transport mode comparison
        print("\n2️⃣  Transport Mode Comparison")
        self.plot_embeddings_comparison(
            embeddings, mapped_labels, true_labels, label_names,
            filename='02_transport_modes_comparison.png'
        )

        # 3. Distribution statistics
        print("\n3️⃣  Distribution Statistics")
        self.plot_cluster_distribution(
            clusters, true_labels, label_names,
            filename='03_cluster_distribution.png'
        )

        # 4. Confusion matrix (with labeled data)
        labeled_mask = true_labels >= 0
        if labeled_mask.sum() > 0:
            print("\n4️⃣  Confusion Matrix")
            label_list = [label_names.get(i, f'Label {i}')
                         for i in range(len(label_names))]

            self.plot_confusion_matrix(
                true_labels[labeled_mask],
                mapped_labels[labeled_mask],
                label_names=label_list,
                filename='04_confusion_matrix.png'
            )

            self.plot_confusion_matrix(
                true_labels[labeled_mask],
                mapped_labels[labeled_mask],
                label_names=label_list,
                normalize=True,
                filename='04_confusion_matrix_normalized.png'
            )

        # 5. Cluster-label heatmap
        print("\n5️⃣  Cluster-Label Heatmap")
        self.plot_cluster_label_heatmap(
            clusters, true_labels, label_names,
            filename='05_cluster_label_heatmap.png'
        )

        print("\n✅ Evaluation report generation complete!")
        print(f"📁 Save location: {self.save_dir}")


# Compatibility aliases
EmbeddingVisualizer = UnifiedVisualizer
TrainingVisualizer = UnifiedVisualizer
ClusterVisualizer = UnifiedVisualizer
TrajectoryVisualizer = UnifiedVisualizer
