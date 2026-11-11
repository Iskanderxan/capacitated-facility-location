"""Loading, validating, sampling and generating problem instances.

An *instance* is two tables:

servers.csv : server_id, location_x, location_y, setup_cost, capacity_streams
dmas.csv    : dma_id, location_x, location_y, demand_streams

Internally every server and DMA is addressed by its row position (0..N-1 and
0..M-1) so that all heavy lifting can be done with NumPy arrays; the original
ids are only used when results are written out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SERVER_COLUMNS = ["server_id", "location_x", "location_y", "setup_cost", "capacity_streams"]
DMA_COLUMNS = ["dma_id", "location_x", "location_y", "demand_streams"]


class InfeasibleInstanceError(ValueError):
    """Raised when no assignment can satisfy the constraints."""


@dataclass
class Instance:
    """A problem instance with NumPy views of the columns the solvers need."""

    servers: pd.DataFrame
    dmas: pd.DataFrame

    server_xy: np.ndarray = field(init=False, repr=False)
    setup_cost: np.ndarray = field(init=False, repr=False)
    capacity: np.ndarray = field(init=False, repr=False)
    dma_xy: np.ndarray = field(init=False, repr=False)
    demand: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.servers = self.servers.reset_index(drop=True)
        self.dmas = self.dmas.reset_index(drop=True)
        self.server_xy = self.servers[["location_x", "location_y"]].to_numpy(dtype=float)
        self.setup_cost = self.servers["setup_cost"].to_numpy(dtype=float)
        self.capacity = self.servers["capacity_streams"].to_numpy(dtype=float)
        self.dma_xy = self.dmas[["location_x", "location_y"]].to_numpy(dtype=float)
        self.demand = self.dmas["demand_streams"].to_numpy(dtype=float)

    @property
    def n_servers(self) -> int:
        return len(self.servers)

    @property
    def n_dmas(self) -> int:
        return len(self.dmas)

    def distances_to(self, dma_rows: np.ndarray, server_rows: np.ndarray) -> np.ndarray:
        """Euclidean distance for paired arrays of DMA and server positions."""
        diff = self.dma_xy[dma_rows] - self.server_xy[server_rows]
        return np.hypot(diff[:, 0], diff[:, 1])

    def check_obvious_infeasibility(self) -> None:
        """Cheap necessary conditions for feasibility.

        Passing these checks does NOT prove the instance is feasible (packing DMAs
        into servers is a bin-packing problem), but failing them proves it is not.
        """
        if self.demand.sum() > self.capacity.sum():
            raise InfeasibleInstanceError(
                f"Total demand ({self.demand.sum():,.0f}) exceeds total server "
                f"capacity ({self.capacity.sum():,.0f})."
            )
        too_big = self.demand > self.capacity.max()
        if too_big.any():
            ids = self.dmas.loc[too_big, "dma_id"].tolist()[:5]
            raise InfeasibleInstanceError(
                f"{int(too_big.sum())} DMA(s) demand more than the largest server can "
                f"hold, e.g. {ids}."
            )


def _validate_table(df: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing column(s) {missing}. Expected {columns}.")
    df = df[columns].copy()
    if df.empty:
        raise ValueError(f"{name}: the file has no rows.")
    numeric = columns[1:]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[numeric].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} row(s) have missing or non-numeric values.")
    id_col = columns[0]
    if df[id_col].duplicated().any():
        dups = df.loc[df[id_col].duplicated(), id_col].tolist()[:5]
        raise ValueError(f"{name}: duplicate {id_col} values, e.g. {dups}.")
    non_negative = [c for c in numeric if not c.startswith("location")]
    if (df[non_negative] < 0).any().any():
        raise ValueError(f"{name}: {non_negative} must be non-negative.")
    return df


def make_instance(servers: pd.DataFrame, dmas: pd.DataFrame) -> Instance:
    """Validate two DataFrames and wrap them in an Instance."""
    servers = _validate_table(servers, SERVER_COLUMNS, "servers")
    dmas = _validate_table(dmas, DMA_COLUMNS, "dmas")
    if (servers["capacity_streams"] <= 0).any():
        raise ValueError("servers: capacity_streams must be positive.")
    return Instance(servers=servers, dmas=dmas)


def load_instance(servers_path: str | Path, dmas_path: str | Path) -> Instance:
    """Read and validate servers.csv and dmas.csv."""
    return make_instance(pd.read_csv(servers_path), pd.read_csv(dmas_path))


def sample_instance(
    instance: Instance,
    n_servers: int | None = None,
    n_dmas: int | None = None,
    seed: int = 42,
) -> Instance:
    """Random subset of servers and/or DMAs (useful to try a large file quickly)."""
    servers, dmas = instance.servers, instance.dmas
    if n_servers is not None and n_servers < len(servers):
        servers = servers.sample(n=n_servers, random_state=seed)
    if n_dmas is not None and n_dmas < len(dmas):
        dmas = dmas.sample(n=n_dmas, random_state=seed)
    return Instance(servers=servers, dmas=dmas)


def generate_instance(
    n_servers: int = 200,
    n_dmas: int = 800,
    seed: int = 42,
    capacity_ratio: float = 2.5,
    width: float = 100_000.0,
    height: float = 60_000.0,
) -> Instance:
    """Create a synthetic but realistic-looking instance.

    * DMAs: ~80% grouped around metro centres, ~20% scattered (rural).
      Demand is log-normal, so a few markets are much larger than the rest.
    * Servers: ~60% near metro centres, ~40% anywhere.
    * Capacity: three size tiers, scaled so total capacity = capacity_ratio x
      total demand (so there is real choice about which servers to open).
    * Setup cost: grows sub-linearly with capacity (economies of scale) plus noise.
    """
    if capacity_ratio < 1.05:
        raise ValueError("capacity_ratio must be at least 1.05 to leave room for packing.")
    rng = np.random.default_rng(seed)

    n_metros = max(5, min(30, n_dmas // 30))
    metros = np.column_stack([rng.uniform(0.05, 0.95, n_metros) * width,
                              rng.uniform(0.05, 0.95, n_metros) * height])
    metro_weight = rng.dirichlet(np.full(n_metros, 1.5))
    metro_spread = rng.uniform(0.02, 0.06, n_metros) * width

    def clustered_points(n: int, rural_share: float, spread_scale: float) -> np.ndarray:
        n_rural = int(round(n * rural_share))
        n_urban = n - n_rural
        which = rng.choice(n_metros, size=n_urban, p=metro_weight)
        urban = metros[which] + rng.normal(size=(n_urban, 2)) * (metro_spread[which, None] * spread_scale)
        rural = np.column_stack([rng.uniform(0, width, n_rural), rng.uniform(0, height, n_rural)])
        pts = np.vstack([urban, rural])
        pts[:, 0] = np.clip(pts[:, 0], 0, width)
        pts[:, 1] = np.clip(pts[:, 1], 0, height)
        return pts[rng.permutation(n)]

    dma_xy = clustered_points(n_dmas, rural_share=0.2, spread_scale=1.0)
    demand = np.clip(np.round(rng.lognormal(mean=6.3, sigma=0.6, size=n_dmas)), 20, None)

    server_xy = clustered_points(n_servers, rural_share=0.4, spread_scale=2.0)
    tier = rng.choice([1.0, 2.5, 6.0], size=n_servers, p=[0.5, 0.35, 0.15])
    raw_capacity = tier * rng.uniform(0.8, 1.2, n_servers)
    capacity = raw_capacity / raw_capacity.sum() * capacity_ratio * demand.sum()
    capacity = np.maximum(np.round(capacity, -1), np.ceil(demand.max() * 1.2 / 10) * 10)
    setup_cost = np.round((1_500 + 12 * capacity ** 0.75) * rng.uniform(0.8, 1.2, n_servers), 2)

    servers = pd.DataFrame({
        "server_id": [f"S{i:04d}" for i in range(1, n_servers + 1)],
        "location_x": np.round(server_xy[:, 0], 2),
        "location_y": np.round(server_xy[:, 1], 2),
        "setup_cost": setup_cost,
        "capacity_streams": capacity.astype(int),
    })
    dmas = pd.DataFrame({
        "dma_id": [f"D{j:04d}" for j in range(1, n_dmas + 1)],
        "location_x": np.round(dma_xy[:, 0], 2),
        "location_y": np.round(dma_xy[:, 1], 2),
        "demand_streams": demand.astype(int),
    })
    return Instance(servers=servers, dmas=dmas)


def save_instance(instance: Instance, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s_path, d_path = out / "servers.csv", out / "dmas.csv"
    instance.servers.to_csv(s_path, index=False)
    instance.dmas.to_csv(d_path, index=False)
    return s_path, d_path
