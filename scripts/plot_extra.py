"""Sufficiency curves with the 20 random orders (follow-up to the advisor's review).

Replaces figures/final/sufficiency_curves.* with a version that shows, for the
clean (forward) direction only, the discovery order and the median and 5-95%
range of 20 random orders. Larger fonts and direct labels instead of a legend.

    uv run --no-project --with matplotlib --with numpy python -m scripts.plot_extra
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/final"
from scripts.palette import FIGSIZE_FULL, GRAY, SAGE_DARK, TERRACOTTA

RANKED = SAGE_DARK
RANDOM = TERRACOTTA

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def comma(v, _):
    return f"{v:g}".replace(".", ",")


def panel(ax, run, title):
    summary = json.loads((ROOT / "data/final/summary.json").read_text())
    values = summary[run]["curves"]["all"]
    seeds = json.loads((ROOT / f"data/extra/{run}_seeds_summary.json").read_text())

    grid = (0, 4, 8, 16, 32, 64, 128, 384)
    ranked = {n: values[f"ranked/{n}/forward"]["estimate"] for n in grid}
    ranked.update({int(k.split("/")[0]): v for k, v in seeds["ranked"].items()
                   if k.endswith("/forward")})
    sizes = sorted(ranked)
    ax.plot(sizes, [100 * ranked[n] for n in sizes], "-o", color=RANKED, lw=2, ms=4,
            label="Ordem da descoberta")
    for n in grid:
        lo, hi = values[f"ranked/{n}/forward"]["ci"]
        ax.plot([n, n], [100 * lo, 100 * hi], color=RANKED, lw=1)

    rnd = seeds["random"]["forward"]
    rgrid = [0] + sorted(int(n) for n in rnd) + [384]
    end = values["random/384/forward"]["estimate"]
    med = [0] + [rnd[str(n)]["median"] for n in rgrid[1:-1]] + [end]
    p5 = [0] + [rnd[str(n)]["p5"] for n in rgrid[1:-1]] + [end]
    p95 = [0] + [rnd[str(n)]["p95"] for n in rgrid[1:-1]] + [end]
    ax.fill_between(rgrid, [100 * v for v in p5], [100 * v for v in p95],
                    color=RANDOM, alpha=0.2, lw=0, label="Aleatórias: faixa 5–95%")
    ax.plot(rgrid, [100 * v for v in med], "-s", color=RANDOM, lw=2, ms=4,
            label="Aleatórias: mediana")

    ax.axhline(80, color=GRAY, lw=1, ls=":")
    ax.set_xscale("symlog", linthresh=4)
    ax.set_xlim(0, 420)
    ax.set_ylim(-5, 118)
    ticks = (0, 4, 8, 16, 32, 64, 128, 384)
    ax.set_xticks(ticks, [str(n) for n in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(comma))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$\it{Heads}$ mantidas")


def main():
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL, sharey=True)
    panel(axes[0], "curve_bracket", "Delimitadores")
    panel(axes[1], "curve_indent", "Indentação")
    axes[0].set_ylabel("Efeito recuperado (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(t) for t in ("Ordem da descoberta", "Aleatórias: mediana",
                                       "Aleatórias: faixa 5–95%")]
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "sufficiency_curves.pdf")
    fig.savefig(OUT / "sufficiency_curves.png", dpi=180)
    print("saved", OUT / "sufficiency_curves.pdf")


if __name__ == "__main__":
    main()
