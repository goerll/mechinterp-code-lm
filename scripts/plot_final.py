"""Publication artifacts derived from final/summary.json; no hand-entered results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/final"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def decimal_comma(fig):
    """Brazilian decimal separator on numeric axes; categorical labels untouched."""
    comma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), matplotlib.ticker.ScalarFormatter):
                axis.set_major_formatter(comma)


def save(fig, name):
    decimal_comma(fig)
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".png"), dpi=180)
    plt.close(fig)


def main():
    summary = json.loads((ROOT / "data/final/summary.json").read_text())
    if "boundary_run" in summary:
        labels = [
            (t, w) for t in ("increase", "continuation", "dedent") for w in (2, 4)
        ]
        matrix = []
        for task, width in labels:
            matrix.append(
                [
                    summary[run]["transfer"]["by_condition"][
                        f"{task}|w{width}|comments{distance}"
                    ]["candidate_accuracy"]["estimate"]
                    for run in ("behavior_run", "boundary_run")
                    for distance in (0, 8)
                ]
            )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(
            range(4),
            [
                "Original\n0 comentários",
                "Original\n8 comentários",
                "Deslocada\n0 comentários",
                "Deslocada\n8 comentários",
            ],
        )
        translations = {
            "increase": "Aumento",
            "continuation": "Continuação",
            "dedent": "Redução",
        }
        ax.set_yticks(range(6), [f"{translations[t]} / {w} espaços" for t, w in labels])
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(
                    j,
                    i,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                )
        fig.colorbar(im, ax=ax, label="Acurácia")
        save(fig, "boundary_behavior")
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(3)
        for shift, (run, label, color) in enumerate(
            (
                ("behavior_run", "Fronteira original", "#446f9b"),
                ("boundary_run", "Um espaço no sufixo", "#d68438"),
            )
        ):
            values = [
                summary[run]["natural"]["by_condition"][t]["candidate_accuracy"]
                for t in translations
            ]
            ys = [v["estimate"] for v in values]
            err = [
                [y - v["ci"][0] for y, v in zip(ys, values)],
                [v["ci"][1] - y for y, v in zip(ys, values)],
            ]
            ax.bar(
                x + (shift - 0.5) * 0.34,
                ys,
                0.34,
                label=label,
                color=color,
                yerr=err,
                capsize=3,
            )
        ax.set_xticks(x, list(translations.values()))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Reprodução do recuo original")
        ax.legend(frameon=False)
        save(fig, "natural_behavior")
    runs = [
        r
        for r in (
            "bracket_validation",
            "indent_validation",
            "indent_boundary_validation",
        )
        if r in summary
    ]
    if runs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        names = [
            "L14H10",
            "L15H15",
            "L18H13",
            "L18H14",
            "L15H3",
            "L14H0",
            "L15H0",
            "L18H0",
        ]
        x = np.arange(len(names))
        labels = {
            "bracket_validation": "Delimitadores",
            "indent_validation": "Indentação original",
            "indent_boundary_validation": "Indentação / fronteira deslocada",
        }
        for i, run in enumerate(runs):
            effects = summary[run]["validation"]["all_eligible"]["effects"]
            vals = [effects[h + "/resample/damage"]["estimate"] for h in names]
            ax.bar(x + (i - (len(runs) - 1) / 2) * 0.25, vals, 0.25, label=labels[run])
        ax.set_xticks(x, names, rotation=40, ha="right")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("Dano / diferença limpa–corrompida")
        ax.legend(frameon=False, fontsize=8)
        save(fig, "head_effects")
    curve_runs = [r for r in ("curve_bracket", "curve_indent") if r in summary]
    if curve_runs:
        fig, axes = plt.subplots(
            1, len(curve_runs), figsize=(6 * len(curve_runs), 4), squeeze=False
        )
        for ax, run in zip(axes[0], curve_runs):
            values = summary[run]["curves"]["all"]
            sizes = (0, 4, 8, 16, 32, 64, 128, 384)
            for kind, color in (("ranked", "#286b92"), ("random", "#a7662d")):
                for direction, style in (("forward", "-"), ("reverse", "--")):
                    rows = [values[f"{kind}/{n}/{direction}"] for n in sizes]
                    label = ("Descoberta" if kind == "ranked" else "Aleatória") + (
                        " / limpa" if direction == "forward" else " / reversa"
                    )
                    ax.plot(
                        sizes,
                        [r["estimate"] for r in rows],
                        style,
                        color=color,
                        label=label,
                    )
                    if direction == "forward":
                        ax.fill_between(
                            sizes,
                            [r["ci"][0] for r in rows],
                            [r["ci"][1] for r in rows],
                            color=color,
                            alpha=0.12,
                        )
            ax.axhline(0.8, color="gray", lw=1, ls=":")
            ax.set_xscale("symlog", linthresh=4)
            ax.set_xlim(0, 410)
            ax.set_xticks(sizes, [str(n) for n in sizes], rotation=40)
            ax.set_title(
                "Delimitadores"
                if run == "curve_bracket"
                else "Indentação / fronteira deslocada"
            )
            ax.set_xlabel("Heads retidas; todas as MLPs ativas")
            ax.set_ylabel("Efeito recuperado")
            ax.legend(fontsize=7, frameon=False)
        save(fig, "sufficiency_curves")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.1, 1.1, 1.3, "Tokens +\nposições"),
        (1.8, 1.1, 1.4, "Espaço\nresidual"),
        (3.6, 1.1, 1.8, "Atenção\n16 queries\nK/V compartilhados"),
        (5.9, 1.1, 1.3, "MLP"),
        (8.1, 1.1, 1.7, "Logits do\npróximo token"),
    ]
    for x, y, w, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, 0.9, boxstyle="round,pad=.06", fc="#edf2f5", ec="#486478"
            )
        )
        ax.text(x + w / 2, y + 0.45, text, ha="center", va="center", fontsize=9)
    for start, end in ((1.4, 1.8), (3.2, 3.6), (5.4, 5.9), (7.2, 8.1)):
        ax.annotate(
            "",
            (end, 1.55),
            (start, 1.55),
            arrowprops={"arrowstyle": "->", "color": "#486478"},
        )
    ax.text(
        5.4,
        2.6,
        "24 blocos sequenciais; normalizações e somas residuais omitidas",
        ha="center",
        fontsize=9,
    )
    ax.text(
        4.5,
        0.45,
        "Intervenção em z:\numa query-head, posição ou conjunto",
        ha="center",
        fontsize=9,
    )
    ax.text(8.9, 0.45, "Métrica:\nmargem entre alvos", ha="center", fontsize=9)
    save(fig, "architecture_interventions")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    rows = [
        (3.1, "Entrada limpa\nabridor (", "Ativação limpa", r"$m_C$"),
        (1.9, "Contrafactual\nabridor [", "Ativação do par", r"$m_K$"),
        (0.7, "Mesmo contrafactual\nabridor [", "Ativação substituída", r"$m_I$"),
    ]
    for y, prompt, activation, metric in rows:
        for x, width, label in (
            (0.1, 2, prompt),
            (2.6, 1.8, "Modelo / até h"),
            (5, 2.2, activation),
            (8, 1.8, "Restante / " + metric),
        ):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    width,
                    0.6,
                    boxstyle="round,pad=.04",
                    fc="#f3e5d5" if y == 0.7 and x == 5 else "#edf2f5",
                    ec="#486478",
                )
            )
            ax.text(x + width / 2, y + 0.3, label, ha="center", va="center", fontsize=9)
        for start, end in ((2.1, 2.6), (4.4, 5), (7.2, 8)):
            ax.annotate(
                "",
                (end, y + 0.3),
                (start, y + 0.3),
                arrowprops={"arrowstyle": "->", "color": "#486478"},
            )
    ax.plot([7.0, 7.6, 7.6], [3.1, 2.9, 1.0], color="#b25632", lw=1.7)
    ax.annotate(
        "",
        (7.22, 1.0),
        (7.6, 1.0),
        arrowprops={"arrowstyle": "->", "color": "#b25632", "lw": 1.7},
    )
    ax.text(
        5,
        0.15,
        r"Restauração normalizada: $(m_I-m_K)/(m_C-m_K)$",
        ha="center",
        fontsize=11,
    )
    save(fig, "patching_protocol")
    print("Final figures generated in", OUT)


if __name__ == "__main__":
    main()
