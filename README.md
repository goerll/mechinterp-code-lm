# mechinterp-code-lm

Code, data and results for the bachelor's thesis *Interpretabilidade mecanística
aplicada a modelos de linguagem para programação* (Mechanistic interpretability
applied to language models for programming), by Estevão Goerll Nascimento, advised by
Prof. Dr. Paulo Cesar Rodacki Gomes. Bachelor's program in Computer Science, Instituto
Federal Catarinense (IFC), Campus Blumenau, 2026.

The thesis uses causal interventions (activation patching and ablation) to find which
attention heads in [SantaCoder](https://huggingface.co/bigcode/santacoder), a
1.1B-parameter code model, are responsible for two structural capabilities:
choosing the right closing delimiter and predicting Python indentation. A first stage
characterizes induction heads to validate the tooling on a known phenomenon.

Main findings:

- Small sets of heads (3 for delimiters, 4 for indentation), selected on a discovery
  split, account for about 49% and 78% of the measured effect on held-out templates.
  Their effects are non-additive, and two heads (L14H10, L18H13) take part in both tasks.
- These sets are not sufficient on their own. With all MLPs active, 64 of the 384 heads,
  taken in discovery order, are needed to recover at least 80% of the effect; 128
  heads in random order recover at most 13%.
- Measured indentation depends on the fill-in-the-middle gap boundary. On Python
  standard library code, accuracy goes from 32% to 92% when one space is moved from the
  gap to the suffix, without changing the program.

## Repository layout

```
scripts/      experiment, analysis and plotting code (see below)
tests/        unit tests for datasets, statistics and intervention hooks
data/final/   frozen datasets, per-example results and summary of every final experiment
data/colab/   independent rerun of two experiments on a Google Colab T4 GPU
outputs/      discovery-split rankings, induction confirmation, model equivalence check
figures/final/, tables/final/   figures and tables used in the thesis
notebooks/    induction-head sweep and Colab reproduction notebook
```

## Setup

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv). Dependency versions
are pinned in `uv.lock`.

```bash
git clone https://github.com/goerll/mechinterp-code-lm
cd mechinterp-code-lm
uv sync --frozen
uv run python scripts/verify_santacoder_setup.py   # checks versions and loads the model
```

The local environment installs CPU-only PyTorch. GPU experiments run either on
[Modal](https://modal.com) or on any machine with an NVIDIA GPU (see below).

**SantaCoder loading patch.** SantaCoder's remote code (`modeling_gpt2_mq.py`) imports
`SequenceSummary`, which no longer exists in the pinned `transformers==4.57.6`.
`scripts/patch_santacoder.py` removes that import from the local Hugging Face cache
only; the weights are not changed. Every script calls it before loading the model.

## Reproducing the tables and figures (no GPU)

All numbers, tables and figures in the thesis are computed from the saved results in
`data/final/`. Bootstrap intervals use a fixed seed, so the output is identical:

```bash
uv run python -m scripts.summarize_final   # data/final/*.json -> data/final/summary.json
uv run python -m scripts.final_tables      # -> tables/final/
uv run python -m scripts.plot_final        # -> figures/final/
uv run --with pytest python -m pytest -q
```

## Rerunning the experiments (GPU)

Each final experiment is a mode of `scripts/run_experiment.py`, which needs an NVIDIA
GPU and refuses to overwrite an existing file:

```bash
python -m scripts.run_experiment --mode bracket --save-path data/rerun/bracket.json
python -m scripts.compare_runs data/final/bracket_validation.json data/rerun/bracket.json
```

`scripts/final_modal.py` runs the same modes with the same inputs on Modal, which was
used for the original runs (NVIDIA A10). The full set of final runs took about 27 GPU
minutes (`data/final/budget_ledger.json`).

| Mode | Experiment | Saved result |
|---|---|---|
| `bracket` | Delimiter candidates, held-out confirmation | `bracket_validation.json` |
| `indent` | Indentation candidates, original gap boundary | `indent_validation.json` |
| `indent_boundary` | Indentation candidates, shifted gap boundary | `indent_boundary_validation.json` |
| `behavior` | Indentation transfer and standard-library gaps | `behavior_run.json` |
| `boundary` | Same, shifted gap boundary | `boundary_run.json` |
| `transfer_boundary` | Interventions on the transfer pairs | `transfer_boundary_validation.json` |
| `curve_bracket`, `curve_indent` | Sufficiency curves | `curve_bracket.json`, `curve_indent.json` |
| `delimiter_events`, `delimiter_events_fim` | Closing events, autoregressive and FIM | `delimiter_events.json`, `delimiter_events_fim.json` |
| `content` | Activation interchange between contexts (not discussed in the thesis) | `content_transfer.json` |
| `diagnostic` | Top tokens for a few examples | `token_diagnostic.json` |

`notebooks/reproduction_colab.ipynb` (in Portuguese) installs the pinned versions on
Google Colab, reruns `behavior` and `bracket` and compares them with the saved results.
On a T4, every value matched within 1e-4 logits (`data/colab/`).

### Earlier stages

- **Induction heads.** `notebooks/induction_head_sweep.ipynb` is the exhaustive
  zero-ablation sweep over all 384 heads used to pick the candidates (top 5 by causal
  effect plus the top detection head). Its results, from a rerun on a Colab T4, are in
  `outputs/induction_heads/induction_exhaustive_results.csv`;
  `scripts/induction_calibration_modal.py` evaluates them on new seeds
  (`outputs/induction_heads/calibration_confirmation.json`).
- **Discovery rankings.** `scripts/bracket_modal.py` and `scripts/indentation_modal.py`
  patch every head on the discovery split. The rankings in
  `outputs/bracket_matching/causal_discovery.json` and
  `outputs/indentation/discovery_causal.json` define the candidates and the order used
  in the sufficiency curves.
- **Model conversion check.** `scripts/model_equivalence_modal.py` compares
  TransformerLens with the native implementation
  (`outputs/reproducibility/model_equivalence.json`).

## Reading the saved results

- `pair_correct`: the margin between the two targets has the right sign for both
  members of a pair.
- `*_candidate_correct`: the target wins among the three closers, or among the 24
  space tokens for indentation.
- `*_full_correct`: the target is the top token in the whole vocabulary.
- `gap`: clean margin minus corrupted margin. Interventions use pairs with
  `gap >= 0.5` and equal token lengths.
- Intervention effects are saved in logits per example. `summary.json` normalizes
  them as a ratio of sums (sum of effects over sum of gaps) and gives 95% intervals
  from a bootstrap over templates.

## License

Code is released under the MIT license (`LICENSE`). Data, results, figures and tables
are released under CC BY 4.0 (`LICENSE-DATA.md`). The standard-library snippets in
`data/final/transfer_and_natural.json` come from CPython and remain under the Python
Software Foundation License (`data/final/PYTHON_LICENSE.txt`).
