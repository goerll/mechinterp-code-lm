# mechinterp-code-lm

Replication code and notebooks for the bachelor's thesis "Mechanistic Interpretability Applied to Language Models for Programming" (Instituto Federal Catarinense, Campus Blumenau).

The thesis investigates which internal circuits in SantaCoder are causally responsible for structural code-processing capabilities, using causal ablation via TransformerLens.

---

## Research stages

1. **Induction heads** — replication and characterization of induction heads in SantaCoder using synthetic repeated-token tasks
2. **Bracket matching** — causal analysis of delimiter correspondence behavior
3. **Indentation tracking** — investigation of scope-tracking mechanisms (planned, second semester)

---

## Repository structure
```md
mechinterp-code-lm/
├── notebooks/
│   ├── induction_heads/
│   │   ├── 00_sanity_check.ipynb
│   │   ├── 01_load_baseline_model.ipynb
│   │   ├── 02_test_santacoder_import.ipynb
│   │   ├── 03_tokenization_and_prompts.ipynb
│   │   ├── 04_detection_baseline.ipynb
│   │   ├── 05_causal_tests.ipynb
│   │   ├── 06_detection_reanalysis.ipynb
│   │   └── colab/
│   │       └── induction_exhaustive.ipynb
│   ├── bracket_matching/
│   │   ├── 00_setup.ipynb
│   │   ├── 01_analysis.ipynb
│   │   └── colab/
│   │       └── bracket_exhaustive.ipynb
│   └── indentation/
│       └── colab/
├── scripts/
│   ├── benchmark.py
│   └── patch_santacoder.py
└── pyproject.toml
```

---

## Setup

Requires Python ≥ 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/goerll/mechinterp-code-lm
cd mechinterp-code-lm
uv sync
```

---

## SantaCoder compatibility patch

`HookedTransformer.from_pretrained("santacoder")` fails on `transformers==4.57.6` because SantaCoder's remote `modeling_gpt2_mq.py` imports `SequenceSummary`, which was removed from `transformers.modeling_utils`. `scripts/patch_santacoder.py` removes that import from the local HF cache. It runs automatically via `scripts/verify_santacoder_setup.py` and is called by `scripts/benchmark.py` before loading the model — any new code path that loads SantaCoder must do the same.

---

## Reference
 
- Elhage et al. (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
- Olsson et al. (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
