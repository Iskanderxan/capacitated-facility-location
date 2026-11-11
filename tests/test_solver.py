import itertools

import numpy as np
import pandas as pd
import pytest

from ctvopt import (
    InfeasibleInstanceError,
    build_candidates,
    evaluate,
    generate_instance,
    greedy,
    make_instance,
    solve_lp_relaxation,
    solve_mip,
    violations,
)
from ctvopt.cli import main


def tiny_instance():
    servers = pd.DataFrame({
        "server_id": ["A", "B", "C"],
        "location_x": [0.0, 10.0, 5.0],
        "location_y": [0.0, 0.0, 8.0],
        "setup_cost": [5.0, 6.0, 4.0],
        "capacity_streams": [10, 12, 7],
    })
    dmas = pd.DataFrame({
        "dma_id": [f"d{j}" for j in range(6)],
        "location_x": [1.0, 2.0, 9.0, 8.0, 5.0, 4.0],
        "location_y": [1.0, 0.0, 1.0, 2.0, 7.0, 6.0],
        "demand_streams": [4, 3, 5, 4, 3, 3],
    })
    return make_instance(servers, dmas)


def brute_force_optimum(inst):
    best = np.inf
    for combo in itertools.product(range(inst.n_servers), repeat=inst.n_dmas):
        a = np.array(combo)
        if not violations(inst, a):
            best = min(best, evaluate(inst, a).total)
    return best


@pytest.mark.parametrize("formulation", ["strong", "weak"])
def test_full_mip_matches_brute_force(formulation):
    inst = tiny_instance()
    sol = solve_mip(inst, k=None, mip_gap=0.0, formulation=formulation)
    assert violations(inst, sol.assignment) == []
    assert evaluate(inst, sol.assignment).total == pytest.approx(brute_force_optimum(inst))


def test_lp_relaxation_is_a_lower_bound():
    inst = tiny_instance()
    assert solve_lp_relaxation(inst) <= brute_force_optimum(inst) + 1e-9


def test_mip_valid_and_not_worse_than_greedy():
    inst = generate_instance(n_servers=30, n_dmas=120, seed=7)
    g = greedy(inst, k=10)
    m = solve_mip(inst, k=10, mip_gap=0.0, time_limit=60)
    assert violations(inst, g.assignment) == []
    assert violations(inst, m.assignment) == []
    assert evaluate(inst, m.assignment).total <= evaluate(inst, g.assignment).total + 1e-6


def test_restricted_model_never_beats_full_model():
    inst = generate_instance(n_servers=20, n_dmas=60, seed=3)
    full = evaluate(inst, solve_mip(inst, k=None, mip_gap=0.0).assignment).total
    restricted = evaluate(inst, solve_mip(inst, k=3, mip_gap=0.0).assignment).total
    assert restricted >= full - 1e-6


def test_candidates_are_nearest_eligible_servers():
    inst = generate_instance(n_servers=40, n_dmas=100, seed=1)
    cand = build_candidates(inst, k=5)
    full = build_candidates(inst, k=None)
    for j in range(inst.n_dmas):
        mine = cand.server[cand.dma == j]
        reference = full.server[full.dma == j][:5]
        assert set(mine) == set(reference)
        assert (inst.capacity[mine] >= inst.demand[j]).all()


def test_auto_expand_when_k_is_too_small():
    # Both DMAs have server A as their only nearest server, but A fits only one.
    servers = pd.DataFrame({
        "server_id": ["A", "B"], "location_x": [0.0, 100.0], "location_y": [0.0, 0.0],
        "setup_cost": [1.0, 1.0], "capacity_streams": [10, 10],
    })
    dmas = pd.DataFrame({
        "dma_id": ["x", "y"], "location_x": [1.0, 2.0], "location_y": [0.0, 0.0],
        "demand_streams": [6, 6],
    })
    inst = make_instance(servers, dmas)
    with pytest.raises(InfeasibleInstanceError):
        solve_mip(inst, k=1, auto_expand=False)
    sol = solve_mip(inst, k=1, auto_expand=True)
    assert violations(inst, sol.assignment) == []
    assert sol.info["k_used"] == "all"  # k doubled to 2 = every server


def test_obviously_infeasible_instance_is_rejected():
    servers = pd.DataFrame({
        "server_id": ["A"], "location_x": [0.0], "location_y": [0.0],
        "setup_cost": [1.0], "capacity_streams": [5],
    })
    dmas = pd.DataFrame({
        "dma_id": ["x"], "location_x": [1.0], "location_y": [0.0], "demand_streams": [6],
    })
    with pytest.raises(InfeasibleInstanceError):
        solve_mip(make_instance(servers, dmas))


def test_bad_input_is_rejected():
    servers = pd.DataFrame({"server_id": ["A"], "location_x": [0.0]})
    dmas = pd.DataFrame({"dma_id": ["x"], "location_x": [0.0], "location_y": [0.0],
                         "demand_streams": [1]})
    with pytest.raises(ValueError, match="missing column"):
        make_instance(servers, dmas)


def test_pulp_backend_agrees_with_highs():
    pytest.importorskip("pulp")
    inst = tiny_instance()
    highs = solve_mip(inst, k=None, mip_gap=0.0, solver="highs")
    pulp_sol = solve_mip(inst, k=None, mip_gap=0.0, solver="pulp",
                         warm_start=greedy(inst, k=None).assignment)
    assert violations(inst, pulp_sol.assignment) == []
    assert evaluate(inst, pulp_sol.assignment).total == pytest.approx(
        evaluate(inst, highs.assignment).total)


def test_cli_end_to_end(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    assert main(["generate", "--n-servers", "25", "--n-dmas", "80", "--out", str(data)]) == 0
    assert main(["solve", "--servers", str(data / "servers.csv"), "--dmas", str(data / "dmas.csv"),
                 "--k", "5", "--certify", "--out", str(out)]) == 0
    for name in ["assignments.csv", "active_servers.csv", "summary.json", "solution_map.png"]:
        assert (out / name).exists(), name
    assignments = pd.read_csv(out / "assignments.csv")
    assert len(assignments) == 80
    assert assignments["dma_id"].is_unique
