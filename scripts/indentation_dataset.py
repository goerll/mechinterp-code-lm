"""Deterministic matched FIM prompts for next-line indentation prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

FIM_PREFIX = "<fim-prefix>"
FIM_SUFFIX = "<fim-suffix>"
FIM_MIDDLE = "<fim-middle>"


@dataclass(frozen=True)
class IndentationPair:
    label: str
    split: str
    clean_prompt: str
    corrupt_prompt: str
    negative_control_prompt: str
    clean_target_spaces: int
    corrupt_target_spaces: int
    clean_statement_spaces: int
    corrupt_statement_spaces: int
    depth_band: str
    compound: str
    distance: str
    template_id: int

    def to_dict(self) -> dict:
        return asdict(self)


TEMPLATES = (
    ("def transform(source):", "value = normalize(source)", "return value"),
    ("def collect(items):", "value = list(items)", "return value"),
    ("def evaluate(record):", "value = inspect(record)", "return value"),
    ("def aggregate(samples):", "value = prepare(samples)", "return value"),
    ("def render(node):", "result = visit(node)", "return result"),
    ("def validate(entry):", "result = check(entry)", "return result"),
    ("def serialize(payload):", "result = encode(payload)", "return result"),
    ("def summarize(values):", "result = reduce(values)", "return result"),
)
COMPOUNDS = {
    "if": "if enabled:",
    "for": "for item in values:",
    "while": "while pending:",
    "with": "with context():",
}


def _fim(prefix: str, suffix: str) -> str:
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def generate_indentation_pairs() -> list[IndentationPair]:
    pairs = []
    for template_id, (header, setup, suffix) in enumerate(TEMPLATES):
        split = "discovery" if template_id < 4 else "confirmation"
        for depth_band, levels in (("nested", (4, 8)), ("deep", (8, 12))):
            for compound, final_line in COMPOUNDS.items():
                for distance in ("short", "long"):
                    history = [header]
                    if depth_band == "deep":
                        history += ["    if outer:", "        if middle:", f"            {setup}"]
                    elif depth_band == "nested":
                        history += ["    if outer:", f"        {setup}"]
                    else:
                        history += [f"    {setup}"]
                    if distance == "long":
                        pad = 12 if depth_band == "deep" else 8
                        history += [" " * pad + "trace = value", " " * pad + "values = [value]"]
                    low, high = levels
                    clean_stmt, corrupt_stmt = (high, low) if template_id % 2 == 0 else (low, high)
                    base = "\n".join(history) + "\n"
                    clean_prefix = base + " " * clean_stmt + final_line + "\n"
                    corrupt_prefix = base + " " * corrupt_stmt + final_line + "\n"
                    # Same-length lexical control, chosen to tokenize as one replacement.
                    negative_prefix = clean_prefix.replace(
                        header, header.replace("def ", "def control_", 1), 1
                    )
                    clean_target = clean_stmt + 4
                    corrupt_target = corrupt_stmt + 4
                    pairs.append(IndentationPair(
                        label=f"t{template_id}_{depth_band}_{compound}_{distance}",
                        split=split,
                        clean_prompt=_fim(clean_prefix, suffix + "\n"),
                        corrupt_prompt=_fim(corrupt_prefix, suffix + "\n"),
                        negative_control_prompt=_fim(negative_prefix, suffix + "\n"),
                        clean_target_spaces=clean_target,
                        corrupt_target_spaces=corrupt_target,
                        clean_statement_spaces=clean_stmt,
                        corrupt_statement_spaces=corrupt_stmt,
                        depth_band=depth_band, compound=compound, distance=distance,
                        template_id=template_id,
                    ))
    return pairs


def dataset_fingerprint(pairs: list[IndentationPair]) -> str:
    text = json.dumps([p.to_dict() for p in pairs], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def save_dataset(path: Path) -> dict:
    pairs = generate_indentation_pairs()
    payload = {"schema_version": 1, "fingerprint_sha256": dataset_fingerprint(pairs),
               "n_pairs": len(pairs),
               "splits": {s: sum(p.split == s for p in pairs) for s in ("discovery", "confirmation")},
               "pairs": [p.to_dict() for p in pairs]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    result = save_dataset(Path("data/indentation/matched_fim_pairs.json"))
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))
