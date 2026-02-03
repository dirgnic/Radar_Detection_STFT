import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class ClusteringConfig:
    eps: float = 3.0                    # distanta maxima pentru conectarea punctelor
    min_samples: int = 5                    # nr minim de puncte pentru un cluster valid
    n_clusters: Optional[int] = 2           # numerul tinta de clustere
    freq_scale: float = 100.0           # pentru distanta euclidiana
    time_scale: float = 0.05

class BaseClustering:
    def __init__(self, config: ClusteringConfig):
        self.cfg = config
    def normalize_points(self, points, freqs = None, times = None):
        if freqs is not None and times is not None:         # transforma indicii in coordonate scalate
            real_points = np.zeros_like(points, dtype=float)
            freq_indices = np.clip(points[:, 0].astype(int), 0, len(freqs) - 1)
            time_indices = np.clip(points[:, 1].astype(int), 0, len(times) - 1)
            real_points[:, 0] = freqs[freq_indices] / self.cfg.freq_scale       # scaleaza frecventa si timpul
            real_points[:, 1] = times[time_indices] / self.cfg.time_scale
            return real_points
        return points.astype(float)

    def reindex_labels(self, labels):
        unique_labels = sorted([l for l in np.unique(labels) if l != -1])       # identific etichetele unice si ignora zgomotul
        new_labels = np.full_like(labels, -1)
        for i, old_label in enumerate(unique_labels):
            new_labels[labels == old_label] = i
        return new_labels

class AgglomerativeClustering(BaseClustering):              # clustering bottom-up
    def fit(self, points, freqs = None, times = None):
        if len(points) == 0: return np.array([])
        pts = self.normalize_points(points, freqs, times)       # normalizeaza datele conform scalei
        n_points = len(pts)
        labels = np.arange(n_points)
        n_clusters = n_points                               # initial fiecare punct e propriul lui cluster
        target_n = self.cfg.n_clusters if self.cfg.n_clusters else 1        # cate clustere vrem
        while n_clusters > target_n:                # pana ajungem la nr dorit de clustere
            min_dist = np.inf
            pair = (0, 0)                           # pereche de clustere ce urmeaza a fi unite
            for i in range(n_points):
                dists = np.sqrt(np.sum((pts[i + 1:] - pts[i]) ** 2, axis=1))        # distanta de la un punct la restul
                if dists.size == 0: continue        # trece peste daca e ultimul punct
                curr_min_idx = np.argmin(dists) + i + 1         # indexul celui mai apropiat vecin
                if dists[curr_min_idx - i - 1] < min_dist:      # daca am gasit o distanta mai mica decat minimul actual si punctele sunt in clustere diferite
                    if labels[i] != labels[curr_min_idx]:
                        min_dist = dists[curr_min_idx - i - 1]
                        pair = (labels[i], labels[curr_min_idx])        # retine perechea si actualizraza minimul
            if min_dist == np.inf: break
            labels[labels == pair[1]] = pair[0]                     # uneste clusterele
            n_clusters = len(np.unique(labels))             # actualizeaza numarul de clustere ramase
        return self.reindex_labels(labels)

class HDBSCAN(BaseClustering):                  # hierarchical
    def fit(self, points, freqs = None, times = None):
        if len(points) < self.cfg.min_samples:
            return np.full(len(points), -1)     # daca avem prea putine puncte, sunt marcate ca zgomot
        pts = self.normalize_points(points, freqs, times)
        n_points = len(pts)
        core_distances = np.zeros(n_points)             # pentru distantele la toate celelalte puncte
        for i in range(n_points):
            d = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
            core_distances[i] = np.partition(d, self.cfg.min_samples - 1)[self.cfg.min_samples - 1]
        mst_edges = self.compute_mst_efficient(pts, core_distances)     # construieste apm cu mrd
        mst_edges = mst_edges[mst_edges[:, 2].argsort()]                # sorteaza muchiile crescator dupa cost
        raw_labels = self.extract_clusters(mst_edges, n_points)         # extrage clusterele prin taierea muchiilor peste pragul eps
        return self.reindex_labels(raw_labels)

    def compute_mst_efficient(self, pts, core_dists):                   # algoritmul lui Prim adaptat
        n = len(pts)
        visited = np.zeros(n, dtype=bool)
        min_reach_dist = np.full(n, np.inf)
        parent = np.full(n, -1)                         # tatal fiecarui nod
        min_reach_dist[0] = 0
        edges = []
        for _ in range(n):
            u = np.where(~visited)[0][np.argmin(min_reach_dist[~visited])]      # nodul nevizitat cu mrd minim
            visited[u] = True                       # viziteaza nodul
            if parent[u] != -1:                         # daca nu e radacina
                edges.append([u, parent[u], min_reach_dist[u]])     # adauga muchia
            unvisited = ~visited
            if not np.any(unvisited): break
            dists_u_v = np.sqrt(np.sum((pts[unvisited] - pts[u]) ** 2, axis=1))
            mrd_values = np.maximum(np.maximum(core_dists[u], core_dists[unvisited]), dists_u_v)# MRD = max(core_dist[u], core_dist[v], dist_eucl(u,v))
            mask = mrd_values < min_reach_dist[unvisited]       # daca am gasit o cale mai scurta
            indices = np.where(unvisited)[0][mask]              # identifica indicii ce trbuie actualizati si actualizeaza parintele
            min_reach_dist[indices] = mrd_values[mask]
            parent[indices] = u
        return np.array(edges)                      # returneaza muchiile

    def extract_clusters(self, mst_edges, n_points):
        threshold = self.cfg.eps            # pragul de distanta pentru separare
        labels = np.full(n_points, -1, dtype=int)
        adj = {i: [] for i in range(n_points)}
        for u, v, w in mst_edges:               # parcurge muchiile arborelui
            if w <= threshold:
                u, v = int(u), int(v)
                adj[u].append(v)                # adauga conexiunea in ambele sensuri (graf neorientat)
                adj[v].append(u)
        cluster_id = 0
        for i in range(n_points):               # parcurge punctele pentru a gasi componente conexe
            if labels[i] == -1 and len(adj[i]) > 0:
                stack = [i]
                current_cluster = []
                labels[i] = cluster_id
                while stack:                # DFS
                    curr = stack.pop()
                    current_cluster.append(curr)
                    for neighbor in adj[curr]:
                        if labels[neighbor] == -1:
                            labels[neighbor] = cluster_id
                            stack.append(neighbor)
                if len(current_cluster) < self.cfg.min_samples:         # daca este cluster valid conform min_samples treci mai departe
                    for idx in current_cluster:
                        labels[idx] = -1
                else:
                    cluster_id += 1
        return labels
