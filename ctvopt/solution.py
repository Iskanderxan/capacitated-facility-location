"""Solution container, cost evaluation, constraint checking and export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Instance


@dataclass
class Costs:
    setup: float
    delivery: float
    total: float
    n_open: int


@dataclass
class Solution:
    """An assignment of every DMA to one server.

    ``assignment[j]`` is the row position of the server that serves DMA row ``j``.
    The set of activated servers is derived from it: a server is open exactly when
    at least one DMA is assigned to it (opening an unused server only adds cost).
    """

    assignment: np.ndarray
    method: str
    solve_seconds: float
    status: str = "feasible"
    # Proven lower bound on the cost of the model that was actually solved.
    # For a k-restricted MIP this bounds the *restricted* problem, not the full one.
    lower_bound: float | None = None
    info: dict = field(default_factory=dict)

    def open_servers(self) -> np.ndarray:
        return np.unique(self.assignment)


def evaluate(instance: Instance, assignment: np.ndarray) -> Costs:
    """Recompute costs from scratch, independently of any solver output."""
    assignment = np.asarray(assignment, dtype=int)
    open_idx = np.unique(assignment)
    setup = float(instance.setup_cost[open_idx].sum())
    delivery = float(instance.distances_to(np.arange(instance.n_dmas), assignment).sum())
    return Costs(setup=setup, delivery=delivery, total=setup + delivery, n_open=len(open_idx))


def violations(instance: Instance, assignment: np.ndarray) -> list[str]:
    """List every broken constraint (empty list = the solution is valid)."""
    problems: list[str] = []
    assignment = np.asarray(assignment)
    if assignment.shape != (instance.n_dmas,):
        return [f"assignment has shape {assignment.shape}, expected ({instance.n_dmas},)"]
    if not np.issubdtype(assignment.dtype, np.integer):
        return ["assignment must contain integer server positions"]
    if (assignment < 0).any() or (assignment >= instance.n_servers).any():
        problems.append("some DMAs are unassigned or point to a non-existent server")
        return problems
    load = np.bincount(assignment, weights=instance.demand, minlength=instance.n_servers)
    over = np.flatnonzero(load > instance.capacity + 1e-6)
    for i in over[:10]:
        problems.append(
            f"server {instance.servers.at[i, 'server_id']} load {load[i]:,.0f} "
            f"> capacity {instance.capacity[i]:,.0f}"
        )
    if len(over) > 10:
        problems.append(f"... and {len(over) - 10} more overloaded servers")
    return problems


def assert_valid(instance: Instance, solution: Solution) -> None:
    problems = violations(instance, solution.assignment)
    if problems:
        raise AssertionError("Invalid solution:\n  " + "\n  ".join(problems))


def to_tables(instance: Instance, solution: Solution) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (assignments, active_servers) tables keyed by the original ids."""
    a = solution.assignment
    dist = instance.distances_to(np.arange(instance.n_dmas), a)
    assignments = pd.DataFrame({
        "dma_id": instance.dmas["dma_id"].to_numpy(),
        "server_id": instance.servers["server_id"].to_numpy()[a],
        "demand_streams": instance.demand.astype(int),
        "distance": np.round(dist, 3),
    })
    open_idx = solution.open_servers()
    load = np.bincount(a, weights=instance.demand, minlength=instance.n_servers)
    n_dmas = np.bincount(a, minlength=instance.n_servers)
    active = pd.DataFrame({
        "server_id": instance.servers["server_id"].to_numpy()[open_idx],
        "location_x": instance.server_xy[open_idx, 0],
        "location_y": instance.server_xy[open_idx, 1],
        "setup_cost": instance.setup_cost[open_idx],
        "capacity_streams": instance.capacity[open_idx].astype(int),
        "assigned_streams": load[open_idx].astype(int),
        "utilisation": np.round(load[open_idx] / instance.capacity[open_idx], 4),
        "n_dmas": n_dmas[open_idx],
    })
    return assignments, active


def write_outputs(
    instance: Instance,
    solution: Solution,
    out_dir: str | Path,
    extra_summary: dict | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assignments, active = to_tables(instance, solution)
    paths = {
        "assignments": out / "assignments.csv",
        "active_servers": out / "active_servers.csv",
        "summary": out / "summary.json",
    }
    assignments.to_csv(paths["assignments"], index=False)
    active.to_csv(paths["active_servers"], index=False)
    costs = evaluate(instance, solution.assignment)
    summary = {
        "method": solution.method,
        "status": solution.status,
        "solve_seconds": round(solution.solve_seconds, 3),
        "n_servers_available": instance.n_servers,
        "n_dmas": instance.n_dmas,
        "costs": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(costs).items()},
        "lower_bound_of_solved_model": solution.lower_bound,
        "info": solution.info,
    }
    if extra_summary:
        summary.update(extra_summary)
    paths["summary"].write_text(json.dumps(summary, indent=2, default=_json_default))
    return paths


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")
