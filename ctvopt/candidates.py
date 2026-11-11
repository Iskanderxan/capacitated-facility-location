"""Candidate (server, DMA) pairs: the variable-reduction step.

The full model has one assignment variable per (server, DMA) pair: N x M of them
(200 x 800 = 160,000). Most are useless - a DMA is almost never served by a
server on the far side of the map. Keeping only the k nearest *eligible*
servers per DMA (eligible = capacity >= that DMA's demand) shrinks the model to
roughly k x M variables.

This is a heuristic restriction: the optimum of the restricted model can be
worse than the true optimum if some DMA's best server in the optimal plan is
not among its k nearest. Use ``benchmark`` to measure how much that costs on
your data instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .data import Instance, InfeasibleInstanceError


@dataclass
class CandidatePairs:
    """Flat arrays describing the allowed (server, DMA) pairs, grouped by DMA."""

    server: np.ndarray  # server row position for each pair
    dma: np.ndarray     # DMA row position for each pair
    dist: np.ndarray    # Euclidean distance for each pair
    k: int | None       # None = all eligible servers

    def __len__(self) -> int:
        return len(self.server)

    def per_dma(self) -> list[np.ndarray]:
        """Server positions allowed for each DMA, nearest first."""
        bounds = np.searchsorted(self.dma, np.arange(len(np.unique(self.dma)) + 1))
        return [self.server[bounds[j]:bounds[j + 1]] for j in range(len(bounds) - 1)]


def build_candidates(instance: Instance, k: int | None) -> CandidatePairs:
    """k nearest eligible servers per DMA (k=None or k>=N keeps every eligible one)."""
    n, m = instance.n_servers, instance.n_dmas
    demand, capacity = instance.demand, instance.capacity

    # How many servers are big enough for each DMA at all?
    n_eligible = n - np.searchsorted(np.sort(capacity), demand, side="left")
    if (n_eligible == 0).any():
        bad = instance.dmas.loc[n_eligible == 0, "dma_id"].tolist()[:5]
        raise InfeasibleInstanceError(f"No server is large enough for DMA(s) {bad}.")

    if k is None or k >= n:
        return _all_eligible(instance, k=None)
    if k < 1:
        raise ValueError("k must be >= 1 (or None for all servers).")

    # Ask the KD-tree for a few more than k neighbours because some of them may be
    # too small for the DMA. Rows that still come up short are recomputed in full.
    k_query = min(n, max(2 * k, k + 5))
    dist, idx = cKDTree(instance.server_xy).query(instance.dma_xy, k=k_query)
    dist, idx = dist.reshape(m, k_query), idx.reshape(m, k_query)
    eligible = capacity[idx] >= demand[:, None]
    keep = eligible & (np.cumsum(eligible, axis=1) <= k)

    want = np.minimum(k, n_eligible)
    short_rows = np.flatnonzero(keep.sum(axis=1) < want)

    rows, cols = np.nonzero(keep)  # row-major order => grouped by DMA, nearest first
    server, dma, d = idx[rows, cols], rows, dist[rows, cols]

    if len(short_rows):
        drop = np.isin(dma, short_rows)
        server, dma, d = server[~drop], dma[~drop], d[~drop]
        extra_s, extra_j, extra_d = [], [], []
        for j in short_rows:
            dj = np.hypot(*(instance.server_xy - instance.dma_xy[j]).T)
            ok = np.flatnonzero(capacity >= demand[j])
            best = ok[np.argsort(dj[ok], kind="stable")[: want[j]]]
            extra_s.append(best)
            extra_j.append(np.full(len(best), j))
            extra_d.append(dj[best])
        server = np.concatenate([server, *extra_s])
        dma = np.concatenate([dma, *extra_j])
        d = np.concatenate([d, *extra_d])
        order = np.lexsort((d, dma))
        server, dma, d = server[order], dma[order], d[order]

    return CandidatePairs(server=server.astype(np.int64), dma=dma.astype(np.int64), dist=d, k=k)


def _all_eligible(instance: Instance, k: None) -> CandidatePairs:
    n, m = instance.n_servers, instance.n_dmas
    j_grid, i_grid = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    j_flat, i_flat = j_grid.ravel(), i_grid.ravel()
    ok = instance.capacity[i_flat] >= instance.demand[j_flat]
    j_flat, i_flat = j_flat[ok], i_flat[ok]
    d = instance.distances_to(j_flat, i_flat)
    order = np.lexsort((d, j_flat))
    return CandidatePairs(server=i_flat[order], dma=j_flat[order], dist=d[order], k=k)
