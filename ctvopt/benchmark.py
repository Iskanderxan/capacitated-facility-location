"""Measure the speed / quality trade-off instead of guessing it.

For each configuration (greedy, MIP with k = 3, 5, 10, ..., the full MIP, and
the weak formulation) we record solve time and cost, and compare every cost
against the best *proven lower bound* on the full problem:

    lower bound = max( LP relaxation of the full model,
                       dual bound reported by the full MIP, if it was run )

Because the bound is <= the true optimum, ``(cost - bound) / bound`` is an
upper bound on each method's real suboptimality - a number you can defend.
"""

from __future__ import annotations

import time

import pandas as pd

from .data import Instance
from .greedy import greedy
from .mip import SolverError, solve_lp_relaxation, solve_mip
from .solution import assert_valid, evaluate


def run_benchmark(
    instance: Instance,
    ks: list[int | None],
    time_limit: float = 120.0,
    mip_gap: float = 0.01,
    include_weak: bool = True,
    weak_k: int = 10,
) -> tuple[pd.DataFrame, float]:
    rows = []

    t0 = time.perf_counter()
    lp_bound = solve_lp_relaxation(instance, k=None)
    lp_seconds = time.perf_counter() - t0
    best_bound = lp_bound

    sol = greedy(instance, k=10)
    assert_valid(instance, sol)
    rows.append(_row("greedy", "k=10", sol, instance))

    configs = [(k, "strong") for k in ks]
    if include_weak:
        configs.append((weak_k, "weak"))
    for k, formulation in configs:
        label = "all" if k is None else str(k)
        try:
            sol = solve_mip(instance, k=k, time_limit=time_limit, mip_gap=mip_gap,
                            formulation=formulation)
        except SolverError as exc:
            rows.append({"method": f"mip ({formulation})", "candidates": f"k={label}",
                         "status": f"no solution: {exc}"})
            continue
        assert_valid(instance, sol)
        rows.append(_row(f"mip ({formulation})", f"k={label}", sol, instance))
        if k is None and formulation == "strong" and sol.lower_bound is not None:
            best_bound = max(best_bound, sol.lower_bound)

    df = pd.DataFrame(rows)
    df["gap_vs_bound_%"] = (df["total_cost"] - best_bound) / best_bound * 100
    df.attrs["lp_bound"] = lp_bound
    df.attrs["lp_seconds"] = lp_seconds
    return df, best_bound


def _row(method: str, candidates: str, sol, instance: Instance) -> dict:
    costs = evaluate(instance, sol.assignment)
    return {
        "method": method,
        "candidates": candidates,
        "status": sol.status,
        "seconds": round(sol.solve_seconds, 2),
        "variables": sol.info.get("n_variables", ""),
        "constraints": sol.info.get("n_constraints", ""),
        "open_servers": costs.n_open,
        "setup_cost": costs.setup,
        "delivery_cost": costs.delivery,
        "total_cost": costs.total,
    }


def to_markdown(df: pd.DataFrame, best_bound: float) -> str:
    cols = ["method", "candidates", "status", "seconds", "variables", "constraints",
            "open_servers", "total_cost", "gap_vs_bound_%"]
    header = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    lines = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c == "total_cost" and pd.notna(v) and v != "":
                v = f"{v:,.0f}"
            elif c == "gap_vs_bound_%" and pd.notna(v) and v != "":
                v = f"{v:.2f}"
            elif isinstance(v, float) and pd.isna(v):
                v = ""
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    footer = (f"\nLower bound used for the gap column: {best_bound:,.0f} "
              f"(full-model LP relaxation: {df.attrs.get('lp_bound', float('nan')):,.0f}).\n")
    return header + "\n".join(lines) + "\n" + footer
