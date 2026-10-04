"""Deterministic matched-pair datasets for delimiter prediction.

Each clean/corrupt pair differs at exactly one pending opener.  The lexical
scaffold, length, nesting depth, and distractors are otherwise identical.  This
lets activation patching ask whether an internal component carries information
about opener identity, rather than merely general punctuation fluency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
OPENERS = tuple(OPEN_TO_CLOSE)


@dataclass(frozen=True)
class BracketPair:
    label: str
    split: str
    clean_prompt: str
    corrupt_prompt: str
    negative_control_prompt: str
    clean_opener: str
    corrupt_opener: str
    clean_target: str
    corrupt_target: str
    target_char_index: int
    depth: int
    distance: str
    distractors: int
    template_id: int

    def to_dict(self) -> dict:
        return asdict(self)


SCAFFOLDS = (
    ("def transform(source):\n    seed = normalize(source)\n    result = root", "value"),
    ("function transform(source) {\n  const seed = normalize(source);\n  return root", "value"),
    ("def collect(items):\n    cache = prepare(items)\n    output = root", "entry"),
    ("function collect(items) {\n  let cache = prepare(items);\n  const output = root", "entry"),
    ("def evaluate(record):\n    baseline = inspect(record)\n    answer = root", "record"),
    ("function evaluate(record) {\n  const baseline = inspect(record);\n  let answer = root", "record"),
    ("def aggregate(samples):\n    prepared = preprocess(samples)\n    summary = root", "sample"),
    ("function aggregate(samples) {\n  const prepared = preprocess(samples);\n  var summary = root", "sample"),
)


def _other_opener(opener: str, offset: int = 1) -> str:
    index = OPENERS.index(opener)
    return OPENERS[(index + offset) % len(OPENERS)]


def _body(distance: str, distractors: int, atom: str) -> str:
    parts = [atom]
    if distance == "long":
        parts.extend(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
    else:
        parts.extend(["alpha", "beta"])
    matched = ("call(item)", "array[index]", "{key: value}")
    parts.extend(matched[:distractors])
    return ", ".join(parts)


def _make_prompt(
    scaffold: str,
    atom: str,
    *,
    target_opener: str,
    depth: int,
    distance: str,
    distractors: int,
) -> tuple[str, int]:
    text = scaffold
    # Outer pending delimiters use a fixed mixture.  Only the innermost opener
    # is manipulated between the clean and corrupt members of a pair.
    for level in range(depth - 1):
        outer = OPENERS[level % len(OPENERS)]
        text += f" {outer} level_{level}, branch"
    text += f" {target_opener} "
    target_char_index = len(text) - 2
    text += _body(distance, distractors, atom)
    return text, target_char_index


def generate_bracket_pairs() -> list[BracketPair]:
    pairs: list[BracketPair] = []
    for template_id, (scaffold, atom) in enumerate(SCAFFOLDS):
        split = "discovery" if template_id < 4 else "confirmation"
        for clean_opener in OPENERS:
            for depth in (1, 2, 3, 4):
                for distance in ("short", "long"):
                    distractors = 0 if distance == "short" else 2
                    corrupt_opener = _other_opener(clean_opener, template_id % 2 + 1)
                    clean, char_index = _make_prompt(
                        scaffold,
                        atom,
                        target_opener=clean_opener,
                        depth=depth,
                        distance=distance,
                        distractors=distractors,
                    )
                    corrupt = (
                        clean[:char_index]
                        + corrupt_opener
                        + clean[char_index + 1 :]
                    )
                    first_left = clean.index("(")
                    first_right = clean.index(")", first_left)
                    negative_control = (
                        clean[:first_left]
                        + "["
                        + clean[first_left + 1 : first_right]
                        + "]"
                        + clean[first_right + 1 :]
                    )
                    label = (
                        f"t{template_id}_{clean_opener}{corrupt_opener}_"
                        f"d{depth}_{distance}"
                    )
                    pairs.append(
                        BracketPair(
                            label=label,
                            split=split,
                            clean_prompt=clean,
                            corrupt_prompt=corrupt,
                            negative_control_prompt=negative_control,
                            clean_opener=clean_opener,
                            corrupt_opener=corrupt_opener,
                            clean_target=OPEN_TO_CLOSE[clean_opener],
                            corrupt_target=OPEN_TO_CLOSE[corrupt_opener],
                            target_char_index=char_index,
                            depth=depth,
                            distance=distance,
                            distractors=distractors,
                            template_id=template_id,
                        )
                    )
    return pairs


def dataset_fingerprint(pairs: list[BracketPair]) -> str:
    payload = json.dumps([pair.to_dict() for pair in pairs], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def save_dataset(path: Path) -> dict:
    pairs = generate_bracket_pairs()
    payload = {
        "schema_version": 1,
        "fingerprint_sha256": dataset_fingerprint(pairs),
        "n_pairs": len(pairs),
        "splits": {
            split: sum(pair.split == split for pair in pairs)
            for split in ("discovery", "confirmation")
        },
        "pairs": [pair.to_dict() for pair in pairs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    destination = Path("data/bracket_matching/matched_pairs.json")
    result = save_dataset(destination)
    print(json.dumps({key: result[key] for key in result if key != "pairs"}, indent=2))
