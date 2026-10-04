"""Frozen final transfer tasks and auditable standard-library indentation holes."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

NAMES = (
    "dispatch",
    "convert",
    "resolve",
    "expand",
    "filter_items",
    "combine",
    "lookup",
    "extract",
    "assemble",
    "decode_item",
    "format_item",
    "merge_items",
    "select_item",
    "update_item",
    "parse_item",
    "process_item",
)


def fim(prefix, suffix):
    return "<fim-prefix>" + prefix + "<fim-suffix>" + suffix + "<fim-middle>"


def shift_boundary(row):
    """Move one missing space to the suffix; reconstructed source is unchanged."""
    result = dict(row)
    for side in ("clean", "corrupt"):
        key = side + "_prompt"
        target = side + "_target"
        if key in row:
            assert row[target].endswith(" ")
            result[key] = row[key].replace("<fim-suffix>", "<fim-suffix> ", 1)
            result[target] = row[target][:-1]
    if "prompt" in row:
        result["prompt"] = row["prompt"].replace("<fim-suffix>", "<fim-suffix> ", 1)
        result["target"] = row["target"][:-1]
    result["boundary"] = "one_space_in_suffix"
    return result


def transfer_pairs():
    rows = []
    for family, name in enumerate(NAMES):
        for task in ("increase", "continuation", "dedent"):
            for width in (2, 4):
                for depth in (1, 2):
                    for distance in (0, 8):
                        versions = []
                        for offset in (0, 1):
                            d = depth + offset
                            prefix = (
                                f"def {name}(item):\n"
                                + " " * width
                                + "if outer:\n"
                                + " " * (2 * width)
                                + "if middle:\n"
                                + " " * (3 * width)
                                + "pass\n"
                            )
                            keyword = (
                                "if enabled:",
                                "for element in item:",
                                "while pending:",
                                "with manager():",
                            )[family % 4]
                            if task == "dedent":
                                keyword = "if enabled:"
                            prefix += " " * (d * width) + keyword + "\n"
                            if task != "increase":
                                prefix += " " * ((d + 1) * width) + "pass\n"
                            prefix += "".join(
                                f"# context note {i}\n" for i in range(distance)
                            )
                            suffix = "else: pass\n" if task == "dedent" else "pass\n"
                            target = (d if task == "dedent" else d + 1) * width
                            ast.parse(prefix + " " * target + suffix)
                            versions.append((fim(prefix, suffix), target))
                        if family % 2:
                            versions.reverse()
                        rows.append(
                            {
                                "label": f"transfer_f{family}_{task}_w{width}_d{depth}_c{distance}",
                                "family": f"new_{family}",
                                "task": task,
                                "width": width,
                                "depth": depth,
                                "comment_lines": distance,
                                "clean_prompt": versions[0][0],
                                "corrupt_prompt": versions[1][0],
                                "clean_target": " " * versions[0][1],
                                "corrupt_target": " " * versions[1][1],
                            }
                        )
    return rows


def delimiter_completion_pairs():
    """Valid Python expression prefixes; evaluate closer events across merged tokens."""
    rows = []
    mapping = {"(": ")", "[": "]", "{": "}"}
    for family, name in enumerate(NAMES):
        for j, opener in enumerate(mapping):
            other = list(mapping)[(j + 1) % 3]
            for depth in (1, 2):
                for terms in (2, 12):
                    base = f"def {name}(x):\n    return " + ("( " if depth == 2 else "")
                    body = "x + " + " + ".join(str(i) for i in range(1, terms + 1))
                    clean = base + opener + " " + body
                    corrupt = base + other + " " + body
                    for prefix, target in (
                        (clean, mapping[opener]),
                        (corrupt, mapping[other]),
                    ):
                        ast.parse(prefix + target + (")" if depth == 2 else "") + "\n")
                    rows.append(
                        {
                            "label": f"completion_f{family}_{j}_d{depth}_t{terms}",
                            "family": f"completion_{family}",
                            "task": "bracket_completion",
                            "clean_prompt": clean,
                            "corrupt_prompt": corrupt,
                            "clean_target": mapping[opener],
                            "corrupt_target": mapping[other],
                            "depth": depth,
                            "terms": terms,
                        }
                    )
    return rows


def natural_holes(per_module=12):
    modules = (
        "json.decoder",
        "json.encoder",
        "configparser",
        "tokenize",
        "pathlib",
        "functools",
        "contextlib",
        "inspect",
    )
    rows = []
    for module in modules:
        path = Path(importlib.util.find_spec(module).origin)
        source = path.read_text()
        lines = source.splitlines(keepends=True)
        # Only actual AST statement starts, excluding strings/comment pseudo-code.
        starts = {
            node.lineno
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.stmt) and hasattr(node, "lineno")
        }
        candidates = []
        for index, line in enumerate(lines):
            if (
                index + 1 not in starts
                or "\t" in line[: len(line) - len(line.lstrip())]
            ):
                continue
            count = len(line) - len(line.lstrip(" "))
            if not 0 < count <= 24 or not line.strip() or index < 4:
                continue
            previous = next(
                (
                    s
                    for s in reversed(lines[:index])
                    if s.strip() and not s.lstrip().startswith("#")
                ),
                "",
            )
            previous_indent = len(previous) - len(previous.lstrip(" "))
            task = (
                "increase"
                if previous.rstrip().endswith(":") and count > previous_indent
                else "dedent"
                if count < previous_indent
                else "continuation"
            )
            candidates.append((index, count, task))
        # Even deterministic coverage across behavior strata; no model selection.
        chosen = []
        for task in ("increase", "continuation", "dedent"):
            pool = [c for c in candidates if c[2] == task]
            if pool:
                chosen += [
                    pool[min(len(pool) - 1, j * len(pool) // max(1, per_module // 3))]
                    for j in range(min(per_module // 3, len(pool)))
                ]
        for index, count, task in chosen:
            prefix = "".join(lines[max(0, index - 24) : index])
            suffix = lines[index][count:] + "".join(lines[index + 1 : index + 5])
            rows.append(
                {
                    "label": f"natural_{module}_{index + 1}",
                    "family": module,
                    "task": task,
                    "prompt": fim(prefix, suffix),
                    "target": " " * count,
                    "source_line": index + 1,
                    "source_module": module,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "python_version": platform.python_version(),
                    "source_license": "Python Software Foundation License Version 2",
                }
            )
    return rows


def save(path):
    payload = {
        "schema_version": 1,
        "transfer": transfer_pairs(),
        "natural": natural_holes(),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    p = save("data/final/transfer_and_natural.json")
    print({k: len(v) if isinstance(v, list) else v for k, v in p.items()})
