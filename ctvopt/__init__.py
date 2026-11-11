"""ctvopt - capacitated facility location for CTV ad pod stitching servers."""

from .candidates import build_candidates
from .data import (
    Instance,
    InfeasibleInstanceError,
    generate_instance,
    load_instance,
    make_instance,
    sample_instance,
    save_instance,
)
from .greedy import greedy
from .mip import SolverError, build_model, solve_lp_relaxation, solve_mip
from .plot import plot_solution
from .solution import Solution, evaluate, violations, write_outputs

__version__ = "1.0.0"

__all__ = [
    "Instance", "InfeasibleInstanceError", "Solution", "SolverError",
    "build_candidates", "build_model", "evaluate", "generate_instance", "greedy",
    "load_instance", "make_instance", "plot_solution", "sample_instance", "save_instance",
    "solve_lp_relaxation", "solve_mip", "violations", "write_outputs",
]
