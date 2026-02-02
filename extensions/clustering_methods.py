import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class ClusteringConfig:
    eps: float = 3.0
    min_samples: int = 5
    n_clusters: Optional[int] = 2
    freq_scale: float = 100.0
    time_scale: float = 0.05

class BaseClustering:
    def __init__(self, config: ClusteringConfig):
        self.cfg = config
    def normalize_points(self, points, freqs = None, times = None):
        if freqs is not None and times is not None:
            real_points = np.zeros_like(points, dtype=float)
            freq_indices = np.clip(points[:, 0].astype(int), 0, len(freqs) - 1)
            time_indices = np.clip(points[:, 1].astype(int), 0, len(times) - 1)
            real_points[:, 0] = freqs[freq_indices] / self.cfg.freq_scale
            real_points[:, 1] = times[time_indices] / self.cfg.time_scale
            return real_points
        return points.astype(float)

    def reindex_labels(self, labels: np.ndarray) -> np.ndarray:
        unique_labels = sorted([l for l in np.unique(labels) if l != -1])
        new_labels = np.full_like(labels, -1)
        for i, old_label in enumerate(unique_labels):
            new_labels[labels == old_label] = i
        return new_labels

class AgglomerativeClustering(BaseClustering):
    def fit(self, points, freqs = None, times = None):
        if len(points) == 0: return np.array([])
        pts = self.normalize_points(points, freqs, times)
        n_points = len(pts)
        labels = np.arange(n_points)
        n_clusters = n_points
        target_n = self.cfg.n_clusters if self.cfg.n_clusters else 1
        while n_clusters > target_n:
            min_dist = np.inf
            pair = (0, 0)
            for i in range(n_points):
                dists = np.sqrt(np.sum((pts[i + 1:] - pts[i]) ** 2, axis=1))
                if dists.size == 0: continue
                curr_min_idx = np.argmin(dists) + i + 1
                if dists[curr_min_idx - i - 1] < min_dist:
                    if labels[i] != labels[curr_min_idx]:
                        min_dist = dists[curr_min_idx - i - 1]
                        pair = (labels[i], labels[curr_min_idx])
            if min_dist == np.inf: break
            labels[labels == pair[1]] = pair[0]
            n_clusters = len(np.unique(labels))
        return self.reindex_labels(labels)

class HDBSCAN(BaseClustering):
    def fit(self, points, freqs = None, times = None):
        if len(points) < self.cfg.min_samples:
            return np.full(len(points), -1)
        pts = self.normalize_points(points, freqs, times)
        n_points = len(pts)
        core_distances = np.zeros(n_points)
        for i in range(n_points):
            d = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
            core_distances[i] = np.partition(d, self.cfg.min_samples - 1)[self.cfg.min_samples - 1]
        mst_edges = self.compute_mst_efficient(pts, core_distances)
        mst_edges = mst_edges[mst_edges[:, 2].argsort()]
        raw_labels = self.extract_clusters(mst_edges, n_points)
        return self.reindex_labels(raw_labels)

    def compute_mst_efficient(self, pts, core_dists):
        n = len(pts)
        visited = np.zeros(n, dtype=bool)
        min_reach_dist = np.full(n, np.inf)
        parent = np.full(n, -1)
        min_reach_dist[0] = 0
        edges = []
        for _ in range(n):
            u = np.where(~visited)[0][np.argmin(min_reach_dist[~visited])]
            visited[u] = True
            if parent[u] != -1:
                edges.append([u, parent[u], min_reach_dist[u]])
            unvisited = ~visited
            if not np.any(unvisited): break
            dists_u_v = np.sqrt(np.sum((pts[unvisited] - pts[u]) ** 2, axis=1))
            mrd_values = np.maximum(np.maximum(core_dists[u], core_dists[unvisited]), dists_u_v)
            mask = mrd_values < min_reach_dist[unvisited]
            indices = np.where(unvisited)[0][mask]
            min_reach_dist[indices] = mrd_values[mask]
            parent[indices] = u
        return np.array(edges)

    def extract_clusters(self, mst_edges, n_points):
        threshold = self.cfg.eps
        labels = np.full(n_points, -1, dtype=int)
        adj = {i: [] for i in range(n_points)}
        for u, v, w in mst_edges:
            if w <= threshold:
                u, v = int(u), int(v)
                adj[u].append(v)
                adj[v].append(u)
        cluster_id = 0
        for i in range(n_points):
            if labels[i] == -1 and len(adj[i]) > 0:
                stack = [i]
                current_cluster = []
                labels[i] = cluster_id
                while stack:
                    curr = stack.pop()
                    current_cluster.append(curr)
                    for neighbor in adj[curr]:
                        if labels[neighbor] == -1:
                            labels[neighbor] = cluster_id
                            stack.append(neighbor)
                if len(current_cluster) < self.cfg.min_samples:
                    for idx in current_cluster:
                        labels[idx] = -1
                else:
                    cluster_id += 1
        return labels
