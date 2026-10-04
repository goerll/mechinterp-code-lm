"""Write LaTeX numerical macros and tables directly from the final summaries."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    s = json.loads((ROOT / "data/final/summary.json").read_text())
    out = ROOT / "tables/final"
    out.mkdir(parents=True, exist_ok=True)
    macros = {}

    def pct(value):
        return f"{100 * value:.1f}".replace(".", ",") + r"\%"

    for name, run in (
        ("Bracket", "bracket_validation"),
        ("IndentOriginal", "indent_validation"),
        ("IndentShift", "indent_boundary_validation"),
    ):
        if run not in s:
            continue
        r = s[run]
        for suffix, key in (
            ("PairAccuracy", "pair_accuracy"),
            ("BroadAccuracy", "candidate_accuracy"),
            ("FullAccuracy", "full_accuracy"),
        ):
            macros[name + suffix] = pct(r["behavior"][key]["estimate"])
        for suffix, key in (
            ("Damage", "set_task/damage"),
            ("Restoration", "set_task/restore"),
            ("Sufficiency", "keep_task_mlps1/forward"),
        ):
            macros[name + suffix] = pct(
                r["validation"]["all_eligible"]["effects"][key]["estimate"]
            )
    for name, run in (
        ("NaturalOriginal", "behavior_run"),
        ("NaturalShift", "boundary_run"),
    ):
        macros[name + "Accuracy"] = pct(
            s[run]["natural"]["candidate_accuracy"]["estimate"]
        )
        macros[name + "FullAccuracy"] = pct(
            s[run]["natural"]["full_accuracy"]["estimate"]
        )
    for name, run in (("Bracket", "curve_bracket"), ("Indent", "curve_indent")):
        if run not in s:
            continue
        effects = s[run]["curves"]["all"]
        first = next(
            (
                n
                for n in (4, 8, 16, 32, 64, 128, 384)
                if all(
                    effects[f"ranked/{n}/{direction}"]["estimate"] >= 0.8
                    for direction in ("forward", "reverse")
                )
            ),
            None,
        )
        macros[name + "CurveThreshold"] = (
            str(first) if first is not None else "não atingido"
        )
    (out / "numbers.tex").write_text(
        "\n".join(
            chr(92) + "newcommand{" + chr(92) + k + "}{" + v + "}"
            for k, v in macros.items()
        )
        + "\n"
    )
    labels = [
        "L14H10",
        "L15H15",
        "L18H13",
        "L18H14",
        "L15H3",
        "L14H0",
        "L15H0",
        "L18H0",
    ]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Head & Delimitadores & Indentação original & Fronteira deslocada \\",
        r"\midrule",
    ]
    for h in labels:
        cols = []
        for run in (
            "bracket_validation",
            "indent_validation",
            "indent_boundary_validation",
        ):
            if run in s:
                v = s[run]["validation"]["all_eligible"]["effects"][
                    h + "/resample/damage"
                ]
                cols.append(f"{v['estimate']:.3f}".replace(".", ","))
            else:
                cols.append("--")
        lines.append(h + " & " + " & ".join(cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "heads.tex").write_text("\n".join(lines) + "\n")
    curves = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Heads & Delim. limpa & Delim. reversa & Indent. limpa & Indent. reversa \\",
        r"\midrule",
    ]
    for n in (0, 4, 8, 16, 32, 64, 128, 384):
        values = []
        for run in ("curve_bracket", "curve_indent"):
            for direction in ("forward", "reverse"):
                values.append(
                    f"{s[run]['curves']['all'][f'ranked/{n}/{direction}']['estimate']:.3f}".replace(
                        ".", ","
                    )
                    if run in s
                    else "--"
                )
        curves.append(str(n) + " & " + " & ".join(values) + r" \\")
    curves += [r"\bottomrule", r"\end{tabular}"]
    (out / "curves.tex").write_text("\n".join(curves) + "\n")
    additional = []
    if "transfer_boundary_validation" in s:
        additional.append(r"\subsection{Transferência causal para as novas tarefas}")
        additional.append(
            "As análises a seguir distinguem os pares elegíveis dos pares resolvidos entre todos os comprimentos de espaços. "
        )
        for task, label in (
            ("increase", "aumento"),
            ("continuation", "continuação"),
            ("dedent", "redução"),
        ):
            data = s["transfer_boundary_validation"]["validation_by_task"][task]
            additional.append(
                f"Na tarefa de {label}, há {data['all_eligible']['n']} pares elegíveis e {data['broad_solved']['n']} pares resolvidos. "
            )
            if data["broad_solved"]["n"]:
                eff = data["broad_solved"]["effects"]["set_task/damage"]["estimate"]
                additional.append(
                    f"O dano do conjunto de indentação no subconjunto resolvido é {pct(eff)}. "
                )
            else:
                additional.append(
                    "Os efeitos sobre a margem nessa família não são apresentados como mecanismo de resolução correta. "
                )
        additional.append("\n")
    for event_run, setting in (
        ("delimiter_events", "prefixos autoregressivos"),
        ("delimiter_events_fim", "lacunas FIM"),
    ):
        if event_run not in s:
            continue
        data = s[event_run]
        b = data["behavior"]
        additional.append(r"\subsection{Eventos de fechamento em " + setting + "}")
        additional.append(
            f"Nos {b['n']} pares de expressões Python, a acurácia pareada condicionada aos eventos é {pct(b['candidate_accuracy']['estimate'])}, enquanto o próximo token irrestrito inicia os dois fechamentos esperados em {pct(b['full_accuracy']['estimate'])} dos pares. "
        )
        for h in ("L14H10", "L15H15", "L18H13", "set_task"):
            if h + "/damage" in data["event_effects"]:
                v = data["event_effects"][h + "/damage"]
                value = f"{v['estimate']:.3f}".replace(".", ",")
                low = f"{v['ci'][0]:.3f}".replace(".", ",")
                high = f"{v['ci'][1]:.3f}".replace(".", ",")
                label = "do conjunto" if h == "set_task" else "de " + h
                additional.append(
                    f"O efeito {label} sobre a margem dos eventos é {value} (intervalo por família: [{low}; {high}]). "
                )
        additional.append("\n")
    for run, label in (
        ("curve_bracket", "delimitadores"),
        ("curve_indent", "indentação"),
    ):
        if run not in s:
            continue
        additional.append(r"\subsection{Mediação em " + label + "}")
        for key, value in s[run]["mediation"]["all"].items():
            path, direction = key.split("/")
            source, destination = path.split("-")
            effect = "dano" if direction == "damage" else "restauração"
            estimate = f"{value['estimate']:.5f}".replace(".", ",")
            low = f"{value['ci'][0]:.5f}".replace(".", ",")
            high = f"{value['ci'][1]:.5f}".replace(".", ",")
            additional.append(
                f"A rota de {source} para {destination} apresenta {effect} normalizado de {estimate}, com intervalo [{low}; {high}]. "
            )
        additional.append(
            "Essas intervenções transferem a saída de um nó mediador e podem incluir caminhos indiretos.\n"
        )
    if "content_transfer" in s:
        additional.append(r"\subsection{Intercâmbio de ativações e hipótese de coluna}")
        additional.append(
            "Foram transplantadas saídas na posição final de exemplos de outras famílias, preservando o alvo de espaços ou alterando-o. O comparador entre estilos preserva a coluna-alvo e altera a relação entre coluna e profundidade. Esses testes verificam invariância parcial, não constituem uma abstração causal completa.\n"
        )
        additional.append(r"\begin{center}\small\begin{tabular}{lrrrr}\toprule")
        additional.append(
            r"Head & Mesmo estilo/alvo & Outro alvo & Outra largura/alvo & Outro alvo \\ \midrule"
        )
        for h in ("L14H10", "L18H13", "L18H14", "L15H3"):
            values = []
            for cond in ("same_style", "same_column_cross_style"):
                for side in ("clean", "corrupt"):
                    key = f"{cond}/{side}/{h}/absolute"
                    v = s["content_transfer"]["content"]["all"].get(key)
                    values.append(f"{v['estimate']:.3f}" if v else "--")
            additional.append(h + " & " + " & ".join(values) + r" \\")
        additional.append(r"\bottomrule\end{tabular}\end{center}")
        additional.append(
            "Os valores são mudanças absolutas de margem normalizadas pelo contraste pareado. Há 256 receptores no controle lexical e 70 com doador disponível entre estilos. As quatro candidatas preservam melhor a margem quando o alvo e o estilo se mantêm. Para L15H3, porém, preservar a coluna ao trocar o estilo perturba mais a margem que o comparador com outro alvo. Isso limita uma interpretação dessa saída como código invariável apenas da coluna absoluta; não demonstra ausência de informação sobre coluna.\n"
        )
    (out / "additional_results.tex").write_text("\n".join(additional) + "\n")
    print("Wrote final tables and numerical macros.")


if __name__ == "__main__":
    main()
