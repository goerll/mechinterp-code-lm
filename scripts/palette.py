"""Shared figure palette, anchored on the sage green of the thesis TikZ diagrams.

Each colour has a fixed meaning across all data figures:

* SAGE_DARK  - main series: discovery order; shifted gap boundary.
* TERRACOTTA - contrast series: random orders; original gap boundary.
* SLATE      - a third category when needed (delimiter task in head effects).
* GRAY       - references and controls (80% line, zero line).
"""

from matplotlib.colors import LinearSegmentedColormap

SAGE = "#aac1a7"  # TikZ diagrams (fills and strokes)
SAGE_LIGHT = "#e1f0df"  # TikZ diagram light fill
SAGE_DARK = "#4f7a4c"
TERRACOTTA = "#b5653a"
TERRACOTTA_LIGHT = "#ecd3c3"
SLATE = "#5d6f7d"
GRAY = "#8a8f94"

# Lighter, cleaner fills for bars (same hues and meanings as above): solid dark
# fills side by side read as muddy, so bars use these mid tones.
SAGE_FILL = "#86b07f"
TERRACOTTA_FILL = "#e0915f"
SLATE_FILL = "#a3b3c2"

SAGE_CMAP = LinearSegmentedColormap.from_list(
    "sage", ["#ffffff", SAGE_LIGHT, SAGE, SAGE_DARK, "#2c4a2b"]
)

FIGSIZE_FULL = (6.4, 3.0)  # drawn at final \textwidth size, so 10 pt stays 10 pt
