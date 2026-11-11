"""Map of a solution: all servers (active vs inactive), all DMAs, assignment lines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file; works on servers without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from .data import Instance  # noqa: E402
from .solution import Solution, evaluate  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
ACTIVE = "#2a78d6"     # categorical slot 1
DMA = "#eb6834"        # categorical slot 2
INACTIVE = "#a3a29c"
LINE = "#8a8984"


def plot_solution(
    instance: Instance,
    solution: Solution,
    path: str | Path | None = None,
    title: str = "Ad pod stitching server network",
    show: bool = False,
):
    a = solution.assignment
    is_open = np.zeros(instance.n_servers, dtype=bool)
    is_open[np.unique(a)] = True
    costs = evaluate(instance, a)

    fig, ax = plt.subplots(figsize=(12, 7.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    many = instance.n_dmas > 5_000
    segments = np.stack([instance.dma_xy, instance.server_xy[a]], axis=1)
    ax.add_collection(LineCollection(segments, colors=LINE, linewidths=0.4 if many else 0.6,
                                     alpha=0.35 if many else 0.55, zorder=1))

    ax.scatter(*instance.server_xy[~is_open].T, s=26, marker="s", facecolors="none",
               edgecolors=INACTIVE, linewidths=0.9, zorder=2,
               label=f"Inactive server ({int((~is_open).sum())})")
    ax.scatter(*instance.dma_xy.T, s=4 if many else 9, color=DMA, linewidths=0, zorder=3,
               label=f"DMA ({instance.n_dmas})")
    ax.scatter(*instance.server_xy[is_open].T, s=58, marker="s", color=ACTIVE,
               edgecolors=SURFACE, linewidths=1.2, zorder=4,
               label=f"Active server ({int(is_open.sum())})")

    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d4d3cd")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.set_xlabel("location_x", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("location_y", color=TEXT_SECONDARY, fontsize=9)

    fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=14, color=TEXT_PRIMARY, fontweight="bold")
    ax.set_title(
        f"{solution.method}  ·  total cost {costs.total:,.0f}  =  setup {costs.setup:,.0f}"
        f"  +  delivery {costs.delivery:,.0f}",
        loc="left", fontsize=10, color=TEXT_SECONDARY, pad=26,
    )
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
                       fontsize=9, borderaxespad=0.2, handletextpad=0.4, columnspacing=1.4)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path
