"""Greedy construction heuristic - the fast baseline.

Idea: walk through DMAs from the largest demand to the smallest (big DMAs are the
hardest to fit, so place them while there is still room). Give each DMA the
server with the smallest *marginal* cost:

    marginal cost = distance                 if the server is already open
                  = distance + setup_cost    if it would have to be opened

Every decision is final - nothing is ever revisited. That is what makes it
"greedy": fast (one pass, O(M log M + M*k)) but blind to how today's choice
constrains tomorrow's DMAs. It typically lands well above the optimum; the MIP
exists to close that gap.
"""

from __future__ import annotations

import time

import numpy as np

from .candidates import CandidatePairs, build_candidates
from .data import Instance, InfeasibleInstanceError
from .solution import Solution


def greedy(instance: Instance, k: int | None = 10, candidates: CandidatePairs | None = None) -> Solution:
    start = time.perf_counter()
    if candidates is None:
        candidates = build_candidates(instance, k)
    allowed = candidates.per_dma()

    remaining = instance.capacity.copy()
    is_open = np.zeros(instance.n_servers, dtype=bool)
    assignment = np.full(instance.n_dmas, -1, dtype=np.int64)
    n_fallbacks = 0

    for j in np.argsort(-instance.demand, kind="stable"):
        need = instance.demand[j]
        servers = allowed[j]
        fits = remaining[servers] >= need
        if not fits.any():
            # None of the k candidates has room left: widen to every server.
            n_fallbacks += 1
            servers = np.flatnonzero(remaining >= need)
            if len(servers) == 0:
                raise InfeasibleInstanceError(
                    "Greedy could not place DMA "
                    f"{instance.dmas.at[j, 'dma_id']} (the instance may still be feasible; "
                    "greedy packing is not exhaustive)."
                )
            fits = np.ones(len(servers), dtype=bool)
        servers = servers[fits]
        dist = instance.distances_to(np.full(len(servers), j), servers)
        marginal = dist + np.where(is_open[servers], 0.0, instance.setup_cost[servers])
        best = servers[np.argmin(marginal)]
        assignment[j] = best
        remaining[best] -= need
        is_open[best] = True

    return Solution(
        assignment=assignment,
        method="greedy",
        solve_seconds=time.perf_counter() - start,
        status="heuristic",
        info={"k": candidates.k, "dmas_needing_fallback": n_fallbacks},
    )
