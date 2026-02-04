from __future__ import annotations

from typing import List, Optional

import numpy as np


class DBSCAN:
    """
    Implementare DBSCAN pentru clustering punctelor detectate.

    IMPORTANT: In acest proiect punctele sunt clusterizate in coordonate normalizate
    la rezolutia STFT (unitati de bin). Asta face `eps` stabil la schimbarea setarilor
    STFT (window/hop).
    """

    def __init__(
        self,
        eps: float = 3.0,
        min_samples: int = 5,
        freq_scale: float = 100.0,
        time_scale: float = 0.05,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.freq_scale = freq_scale
        self.time_scale = time_scale

    def fit(
        self,
        points: np.ndarray,
        freqs: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Aplica DBSCAN pe punctele 2D.

        Args:
            points: (N,2) cu coordonatele (freq_idx, time_idx)
            freqs/times: optional, pentru conversie/normalizare

        Returns:
            labels (-1 = zgomot)
        """
        if len(points) == 0:
            return np.array([])

        if freqs is not None and times is not None:
            df = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
            dt = abs(times[1] - times[0]) if len(times) > 1 else 1.0

            real_points = np.zeros_like(points, dtype=float)
            freq_indices = np.clip(points[:, 0].astype(int), 0, len(freqs) - 1)
            time_indices = np.clip(points[:, 1].astype(int), 0, len(times) - 1)
            real_points[:, 0] = freqs[freq_indices] / df
            real_points[:, 1] = times[time_indices] / dt
            points_to_use = real_points
        else:
            points_to_use = points.astype(float)

        n_points = len(points_to_use)
        labels = np.full(n_points, -1, dtype=int)
        cluster_id = 0

        for i in range(n_points):
            if labels[i] != -1:
                continue

            neighbors = self._region_query(points_to_use, i)
            if len(neighbors) < self.min_samples:
                continue

            labels[i] = cluster_id
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                if labels[q] != -1:
                    j += 1
                    continue

                labels[q] = cluster_id
                q_neighbors = self._region_query(points_to_use, q)
                if len(q_neighbors) >= self.min_samples:
                    for neighbor in q_neighbors:
                        if neighbor not in seed_set:
                            seed_set.append(neighbor)
                j += 1

            cluster_id += 1

        return labels

    def _region_query(self, points: np.ndarray, idx: int) -> List[int]:
        """
        Gaseste vecinii in raza eps, cu distanta anizotropica.
        """
        diff = points - points[idx]

        weighted_diff = diff.copy()
        weighted_diff[:, 0] = diff[:, 0] / 3.0  # freq tolerant
        weighted_diff[:, 1] = diff[:, 1] * 1.5  # time strict

        distances = np.sqrt(np.sum(weighted_diff**2, axis=1))
        return list(np.where(distances <= self.eps)[0])

