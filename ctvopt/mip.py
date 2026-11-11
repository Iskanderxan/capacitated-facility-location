"""Mixed Integer Programming model for the capacitated facility location problem.

Decision variables (all binary)
    x_i  = 1 if server i is activated
    y_ij = 1 if DMA j is served by server i   (only for candidate pairs)

Objective
    minimise  sum_i setup_i * x_i  +  sum_(i,j) dist_ij * y_ij

Constraints
    (assign)    sum_i y_ij = 1                      for every DMA j
    (capacity)  sum_j demand_j * y_ij <= cap_i * x_i for every server i
    (link)      y_ij <= x_i                          for every candidate pair   [strong only]

About the "link" rows: they are logically redundant - the capacity row already
forbids using a closed server. What they change is the LP relaxation: without
them the LP may open 3% of a server and route 3% of its capacity through it,
so the root lower bound is much weaker (on the bundled data: 2.89M weak vs
3.26M strong, with the optimum near 3.30M). The textbook advice is therefore
"strong formulation = faster proof". Modern solvers, however, derive many of
these cuts on their own, and on the bundled data HiGHS was NOT slower with the
weak model. Measure it on your data: ``benchmark`` includes a weak row.

Two interchangeable back-ends build exactly this model:
    * "highs" - SciPy's ``scipy.optimize.milp`` (HiGHS solver, ships with SciPy)
    * "pulp"  - PuLP + CBC (``pip install pulp``), kept for readers who know PuLP
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .candidates import CandidatePairs, build_candidates
from .data import Instance, InfeasibleInstanceError
from .solution import Solution


class SolverError(RuntimeError):
    """The solver stopped without producing a usable solution."""


@dataclass
class Model:
    """Matrix form of the model: min c'z  s.t.  lb <= A z <= ub, z binary."""

    c: np.ndarray
    A: sparse.csr_matrix
    lb: np.ndarray
    ub: np.ndarray
    integrality: np.ndarray
    servers: np.ndarray        # server row positions that have an x variable
    pair_x_index: np.ndarray   # for each pair, index of its server's x variable
    candidates: CandidatePairs
    formulation: str

    @property
    def n_x(self) -> int:
        return len(self.servers)

    @property
    def n_vars(self) -> int:
        return len(self.c)

    @property
    def n_rows(self) -> int:
        return self.A.shape[0]


def build_model(instance: Instance, candidates: CandidatePairs, formulation: str = "strong") -> Model:
    if formulation not in {"strong", "weak"}:
        raise ValueError("formulation must be 'strong' or 'weak'")
    servers, pair_x = np.unique(candidates.server, return_inverse=True)
    n_x, n_p, m = len(servers), len(candidates), instance.n_dmas
    y_col = n_x + np.arange(n_p)  # column of each y variable (x variables come first)

    c = np.concatenate([instance.setup_cost[servers], candidates.dist])

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0

    # (assign) each DMA served exactly once
    rows.append(r + candidates.dma); cols.append(y_col); vals.append(np.ones(n_p))
    lb.append(np.ones(m)); ub.append(np.ones(m)); r += m

    # (capacity) sum_j d_j y_ij - cap_i x_i <= 0
    rows.append(r + pair_x); cols.append(y_col); vals.append(instance.demand[candidates.dma])
    rows.append(r + np.arange(n_x)); cols.append(np.arange(n_x)); vals.append(-instance.capacity[servers])
    lb.append(np.full(n_x, -np.inf)); ub.append(np.zeros(n_x)); r += n_x

    # (link) y_ij - x_i <= 0
    if formulation == "strong":
        rows.append(r + np.arange(n_p)); cols.append(y_col); vals.append(np.ones(n_p))
        rows.append(r + np.arange(n_p)); cols.append(pair_x); vals.append(-np.ones(n_p))
        lb.append(np.full(n_p, -np.inf)); ub.append(np.zeros(n_p)); r += n_p

    A = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(r, n_x + n_p),
    )
    return Model(
        c=c, A=A, lb=np.concatenate(lb), ub=np.concatenate(ub),
        integrality=np.ones(n_x + n_p), servers=servers, pair_x_index=pair_x,
        candidates=candidates, formulation=formulation,
    )


def solve_mip(
    instance: Instance,
    k: int | None = 10,
    time_limit: float = 60.0,
    mip_gap: float = 0.01,
    solver: str = "highs",
    formulation: str = "strong",
    auto_expand: bool = True,
    warm_start: np.ndarray | None = None,
    verbose: bool = False,
) -> Solution:
    """Solve the k-restricted MIP.

    If the restriction makes the model infeasible (some DMAs cannot all fit into
    their k nearest servers) and ``auto_expand`` is True, k is doubled and the
    model re-solved, up to the full model.

    ``warm_start`` (an assignment array, e.g. from greedy) is used by the PuLP/CBC
    back-end only; SciPy's HiGHS interface does not accept a starting solution.
    """
    instance.check_obvious_infeasibility()
    start = time.perf_counter()
    k_try = k
    attempts = []
    while True:
        candidates = build_candidates(instance, k_try)
        model = build_model(instance, candidates, formulation)
        remaining = max(1.0, time_limit - (time.perf_counter() - start))
        if solver == "highs":
            result = _solve_highs(model, remaining, mip_gap, verbose)
        elif solver == "pulp":
            result = _solve_pulp(instance, model, remaining, mip_gap, verbose, warm_start)
        else:
            raise ValueError("solver must be 'highs' or 'pulp'")
        attempts.append({"k": k_try, "status": result["status"]})

        if result["status"] != "infeasible":
            break
        full = k_try is None or k_try >= instance.n_servers
        if full:
            raise InfeasibleInstanceError("The model is infeasible even with all servers allowed.")
        if not auto_expand:
            raise InfeasibleInstanceError(
                f"Infeasible with k={k_try}; increase k or enable auto_expand."
            )
        k_try = min(2 * k_try, instance.n_servers)

    if result["z"] is None:
        raise SolverError(f"Solver stopped with status '{result['status']}' and no solution.")

    assignment = _assignment_from_y(instance, model, result["z"])
    k_label = "all" if candidates.k is None else candidates.k
    return Solution(
        assignment=assignment,
        method=f"mip-{solver}",
        solve_seconds=time.perf_counter() - start,
        status=result["status"],
        lower_bound=result["bound"],
        info={
            "k_used": k_label,
            "formulation": formulation,
            "n_variables": model.n_vars,
            "n_constraints": model.n_rows,
            "solver_gap": result["gap"],
            "attempts": attempts,
        },
    )


def solve_lp_relaxation(instance: Instance, k: int | None = None, formulation: str = "strong") -> float:
    """Objective of the LP relaxation (binaries relaxed to [0, 1]).

    With k=None this is a valid LOWER BOUND on the true optimum of the full
    problem, so ``(cost - bound) / bound`` is an honest upper bound on how far
    any solution can be from optimal.
    """
    model = build_model(instance, build_candidates(instance, k), formulation)
    res = milp(
        c=model.c, integrality=np.zeros(model.n_vars), bounds=Bounds(0, 1),
        constraints=LinearConstraint(model.A, model.lb, model.ub),
    )
    if res.status != 0:
        raise SolverError(f"LP relaxation failed: {res.message}")
    return float(res.fun)


def _solve_highs(model: Model, time_limit: float, mip_gap: float, verbose: bool) -> dict:
    res = milp(
        c=model.c,
        integrality=model.integrality,
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(model.A, model.lb, model.ub),
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": verbose},
    )
    # 0 = optimal within mip_gap, 1 time/iteration limit, 2 infeasible, 3 unbounded, 4 other
    status = {0: "within_gap", 1: "time_limit", 2: "infeasible", 3: "unbounded"}.get(res.status, "error")
    bound = getattr(res, "mip_dual_bound", None)
    gap = getattr(res, "mip_gap", None)
    return {
        "status": status,
        "z": res.x,
        "bound": None if bound is None or not np.isfinite(bound) else float(bound),
        "gap": None if gap is None or not np.isfinite(gap) else float(gap),
    }


def _solve_pulp(
    instance: Instance,
    model: Model,
    time_limit: float,
    mip_gap: float,
    verbose: bool,
    warm_start: np.ndarray | None,
) -> dict:
    try:
        import pulp
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("The 'pulp' back-end needs PuLP: pip install pulp") from exc

    cand = model.candidates
    prob = pulp.LpProblem("ctv_facility_location", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{t}", cat="Binary") for t in range(model.n_x)]
    y = [pulp.LpVariable(f"y_{p}", cat="Binary") for p in range(len(cand))]

    prob += (
        pulp.lpSum(float(instance.setup_cost[s]) * x[t] for t, s in enumerate(model.servers))
        + pulp.lpSum(float(cand.dist[p]) * y[p] for p in range(len(cand)))
    )

    pairs_of_dma: dict[int, list[int]] = {}
    pairs_of_server: dict[int, list[int]] = {}
    for p in range(len(cand)):
        pairs_of_dma.setdefault(int(cand.dma[p]), []).append(p)
        pairs_of_server.setdefault(int(model.pair_x_index[p]), []).append(p)

    for j, ps in pairs_of_dma.items():
        prob += pulp.lpSum(y[p] for p in ps) == 1, f"assign_{j}"
    for t, ps in pairs_of_server.items():
        prob += (
            pulp.lpSum(float(instance.demand[cand.dma[p]]) * y[p] for p in ps)
            <= float(instance.capacity[model.servers[t]]) * x[t]
        ), f"capacity_{t}"
    if model.formulation == "strong":
        for p in range(len(cand)):
            prob += y[p] <= x[int(model.pair_x_index[p])], f"link_{p}"

    use_warm = warm_start is not None
    if use_warm:
        chosen = set(zip(np.arange(instance.n_dmas).tolist(), np.asarray(warm_start).tolist()))
        open_set = set(np.asarray(warm_start).tolist())
        for t, s in enumerate(model.servers):
            x[t].setInitialValue(1 if int(s) in open_set else 0)
        for p in range(len(cand)):
            y[p].setInitialValue(1 if (int(cand.dma[p]), int(cand.server[p])) in chosen else 0)

    solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit, gapRel=mip_gap, warmStart=use_warm)
    prob.solve(solver)

    # PuLP reports "Optimal" even when CBC hit the time limit with an unproven
    # solution; sol_status distinguishes 1 = proven optimal, 2 = feasible only.
    sol_status = getattr(prob, "sol_status", None)
    if prob.status == pulp.LpStatusInfeasible:
        return {"status": "infeasible", "z": None, "bound": None, "gap": None}
    if prob.status != pulp.LpStatusOptimal or sol_status not in (1, 2, None):
        return {"status": pulp.LpStatus[prob.status].lower(), "z": None, "bound": None, "gap": None}
    z = np.array([v.varValue or 0.0 for v in x] + [v.varValue or 0.0 for v in y])
    status = "time_limit" if sol_status == 2 else "within_gap"
    return {"status": status, "z": z, "bound": None, "gap": None}


def _assignment_from_y(instance: Instance, model: Model, z: np.ndarray) -> np.ndarray:
    y = z[model.n_x:]
    chosen = y > 0.5
    assignment = np.full(instance.n_dmas, -1, dtype=np.int64)
    assignment[model.candidates.dma[chosen]] = model.candidates.server[chosen]
    counts = np.bincount(model.candidates.dma[chosen], minlength=instance.n_dmas)
    if (counts != 1).any():
        raise SolverError("Solver returned a solution where some DMA is not assigned exactly once.")
    return assignment
