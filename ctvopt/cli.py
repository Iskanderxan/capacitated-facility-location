"""Command-line interface:  python -m ctvopt {generate,solve,benchmark} ...

Run ``python -m ctvopt <command> --help`` for every option.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmark import run_benchmark, to_markdown
from .data import InfeasibleInstanceError, generate_instance, load_instance, sample_instance, save_instance
from .greedy import greedy
from .mip import SolverError, solve_lp_relaxation, solve_mip
from .plot import plot_solution
from .solution import assert_valid, evaluate, write_outputs


def _k_arg(value: str) -> int | None:
    if value.lower() in {"all", "full", "none", "0"}:
        return None
    k = int(value)
    if k < 1:
        raise argparse.ArgumentTypeError("k must be >= 1 or 'all'")
    return k


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--servers", default="data/servers.csv", help="path to servers.csv")
    p.add_argument("--dmas", default="data/dmas.csv", help="path to dmas.csv")
    p.add_argument("--sample-servers", type=int, help="randomly keep this many servers")
    p.add_argument("--sample-dmas", type=int, help="randomly keep this many DMAs")
    p.add_argument("--seed", type=int, default=42, help="seed for --sample-*")


def _load(args) -> "Instance":  # noqa: F821
    inst = load_instance(args.servers, args.dmas)
    if args.sample_servers or args.sample_dmas:
        inst = sample_instance(inst, args.sample_servers, args.sample_dmas, args.seed)
    return inst


def cmd_generate(args) -> int:
    inst = generate_instance(args.n_servers, args.n_dmas, args.seed, args.capacity_ratio)
    s, d = save_instance(inst, args.out)
    print(f"Wrote {s} ({inst.n_servers} servers) and {d} ({inst.n_dmas} DMAs).")
    return 0


def cmd_solve(args) -> int:
    inst = _load(args)
    inst.check_obvious_infeasibility()
    print(f"Instance: {inst.n_servers} servers, {inst.n_dmas} DMAs "
          f"({inst.n_servers * inst.n_dmas:,} possible server-DMA pairs)")

    base = greedy(inst, k=args.k or 10)
    assert_valid(inst, base)
    base_cost = evaluate(inst, base.assignment)
    print(f"Greedy baseline:  total {base_cost.total:>14,.2f}  "
          f"({base_cost.n_open} servers, {base.solve_seconds:.2f}s)")

    if args.method == "greedy":
        sol = base
    else:
        try:
            sol = solve_mip(inst, k=args.k, time_limit=args.time_limit, mip_gap=args.gap,
                            solver=args.solver, formulation=args.formulation,
                            warm_start=base.assignment, verbose=args.verbose)
        except SolverError as exc:
            print(f"MIP did not return a solution ({exc}); falling back to greedy.")
            sol = base
    assert_valid(inst, sol)
    costs = evaluate(inst, sol.assignment)

    extra = {}
    print()
    print(f"Method:           {sol.method}  (status: {sol.status})")
    if "k_used" in sol.info:
        print(f"Candidates/DMA:   {sol.info['k_used']}   model: {sol.info['n_variables']:,} variables, "
              f"{sol.info['n_constraints']:,} constraints")
    print(f"Solve time:       {sol.solve_seconds:.2f}s")
    print(f"Active servers:   {costs.n_open} of {inst.n_servers}")
    print(f"Setup cost:       {costs.setup:>14,.2f}")
    print(f"Delivery cost:    {costs.delivery:>14,.2f}")
    print(f"Total cost:       {costs.total:>14,.2f}")
    if sol is not base:
        print(f"vs greedy:        {(base_cost.total - costs.total) / base_cost.total * 100:.2f}% cheaper")
    if sol.lower_bound is not None and sol.info.get("k_used") != "all":
        print("Note: the solver's bound proves optimality of the k-restricted model only. "
              "Use --certify for a bound on the full problem.")

    if args.certify:
        bound = solve_lp_relaxation(inst, k=None)
        gap = (costs.total - bound) / bound * 100
        extra["full_problem_lp_bound"] = round(bound, 2)
        extra["max_gap_to_optimal_pct"] = round(gap, 4)
        print(f"Certificate:      full-problem LP bound {bound:,.2f} -> this solution is at most "
              f"{gap:.2f}% above the true optimum")

    out = Path(args.out)
    paths = write_outputs(inst, sol, out, extra)
    if not args.no_plot:
        paths["map"] = plot_solution(inst, sol, out / "solution_map.png")
    print("\nWrote: " + ", ".join(str(p) for p in paths.values()))
    return 0


def cmd_benchmark(args) -> int:
    inst = _load(args)
    print(f"Benchmarking on {inst.n_servers} servers x {inst.n_dmas} DMAs "
          f"(time limit {args.time_limit:.0f}s per MIP) ...")
    df, bound = run_benchmark(inst, args.ks, time_limit=args.time_limit, mip_gap=args.gap,
                              include_weak=not args.no_weak)
    md = to_markdown(df, bound)
    print()
    print(md)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark.csv", index=False)
    (out / "benchmark.md").write_text(md)
    print(f"Wrote {out / 'benchmark.csv'} and {out / 'benchmark.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ctvopt",
                                     description="Capacitated facility location for CTV ad servers.")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="create a synthetic servers.csv / dmas.csv")
    g.add_argument("--n-servers", type=int, default=200)
    g.add_argument("--n-dmas", type=int, default=800)
    g.add_argument("--capacity-ratio", type=float, default=2.5,
                   help="total capacity / total demand (default 2.5)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="data")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("solve", help="solve an instance and write results + map")
    _add_data_args(s)
    s.add_argument("--method", choices=["mip", "greedy"], default="mip")
    s.add_argument("--solver", choices=["highs", "pulp"], default="highs",
                   help="highs = SciPy/HiGHS (default), pulp = PuLP/CBC (pip install pulp)")
    s.add_argument("--k", type=_k_arg, default=10,
                   help="candidate servers per DMA, or 'all' for the full model (default 10)")
    s.add_argument("--formulation", choices=["strong", "weak"], default="strong")
    s.add_argument("--time-limit", type=float, default=60.0, help="seconds (default 60)")
    s.add_argument("--gap", type=float, default=0.01,
                   help="stop once the solver proves the solution is within this relative gap (default 0.01 = 1%%)")
    s.add_argument("--certify", action="store_true",
                   help="also solve the full LP relaxation to bound the distance to the true optimum")
    s.add_argument("--out", default="outputs")
    s.add_argument("--no-plot", action="store_true")
    s.add_argument("--verbose", action="store_true", help="show solver log")
    s.set_defaults(func=cmd_solve)

    b = sub.add_parser("benchmark", help="compare greedy and MIP at several k against a proven bound")
    _add_data_args(b)
    b.add_argument("--ks", type=_k_arg, nargs="+", default=[3, 5, 10, 20, None],
                   help="values of k to try; 'all' = full model (default: 3 5 10 20 all)")
    b.add_argument("--time-limit", type=float, default=120.0)
    b.add_argument("--gap", type=float, default=0.01)
    b.add_argument("--no-weak", action="store_true", help="skip the weak-formulation comparison")
    b.add_argument("--out", default="outputs")
    b.set_defaults(func=cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (InfeasibleInstanceError, ValueError, FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
