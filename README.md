# Wigin TLLM

Benchmark a language model for **chronological consistency**: does a model
built with a knowledge cutoff of year *Y* actually behave as though it has
never seen anything after *Y*?

Standard language models are trained on text from every period at once. Used
for backtesting or historical analysis they suffer from **lookahead bias** —
a model asked to reason about 2015 has already read about everything that
happened since. This tool measures whether a model genuinely respects its
temporal boundary, and how good it is once it does.

Everything runs on your own machine against your own models.

---

## How it works

```
facts.json ── corpus ──> known/unknown probe sets per cutoff year
                                   │
 models.json ──────────> STAGE 1 · consistency
                         · limits: size, params, pinned SHA, time budget
                         · SVD gate: spectral copy of a reference → year = 0
                         · probes: must know its era, must not know the future
                                   │  leak score < −3.0?
                          no ──────┼────── yes
                           │               │
                     final = 0.0   STAGE 2 · quality
                                   · continue prompts, duel references
                                   · LLM judge picks winners
                                   │
                     final = 0.7 × consistency + 0.3 × quality
```

**Stage 1 — consistency.** For each cutoff year, the model is scored on the
log-probability it assigns to two probe sets:

| Probe set | Content | Requirement |
|---|---|---|
| `unknown` | facts from **after** the cutoff | must **not** recognise them |
| `known` | facts from **before** the cutoff | **must** recognise them |

```
year score = median(unknown) − median(known)    # more negative is better
leak score = mean over EVERY year in scope      # a missing year scores worst
```

Both sets are load-bearing: without `unknown` a model that memorised
everything passes, without `known` an empty model that knows nothing passes.

**Stage 2 — quality.** The model continues a set of incomplete texts and the
completions are judged head-to-head against reference models. The share of
duels won is the quality score. Quality has no absolute scale, so it always
needs references to duel against.

**Qualification.** A model whose leak score does not clear −3.0 never enters
stage 2 and scores 0.0 outright — no amount of quality buys back a failed
consistency check.

**Similarity.** Copies are detected by comparing *singular value spectra*
rather than weights: a copy disguised as `W' = P·W·Q` with orthogonal `P`,
`Q` has near-zero cosine similarity to the original but an identical
spectrum. This powers the stage-1 gate, the standalone `similarity` check,
and `--pairwise` dedup over a set of models.

Every formula, threshold and edge case: [docs/scoring.md](docs/scoring.md).
What to optimise as a model author: [docs/model-guide.md](docs/model-guide.md).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pip install -e ".[chronogpt]"   # scoring native ChronoGPT checkpoints
pip install -e ".[openai]"      # LLM judge and prompt generation
```

Python ≥ 3.10. CPU is enough for small models; a GPU is strongly recommended
at production sizes.

**API key** — only `--judge openai` and prompt generation need one. Put it in
a `.env` file (gitignored, loaded at CLI startup; real environment variables
always win):

```bash
cp .env.example .env    # then set OPENAI_API_KEY=sk-...
```

Everything else — consistency, similarity, `--judge overlap`, the shipped
static prompts, the offline demo — needs no key at all.

## Two-minute demo

Offline, no keys: trains six tiny models on CPU and benchmarks four of them.

```bash
python examples/run_local.py --rebuild
```

```
#  alice — honest and well trained
  leak score       -10.1264   (threshold -3.0000) — clears it
  win rate            0.500   over 6 prompts against 2 reference(s)
Final score  0.8500   = 0.7 x 1.000 + 0.3 x 0.500

#  bob — honest, but never saw part of the material
  leak score       -10.3043   (threshold -3.0000) — clears it
  win rate            0.000   over 6 prompts against 2 reference(s)
Final score  0.7000   = 0.7 x 1.000 + 0.3 x 0.000

#  mallory — trained on facts from after its cutoff
Final score  0.0000   — stage 1 not cleared, so stage 2 never runs

#  null-model — barely trained at all
Final score  0.0000   — stage 1 not cleared, so stage 2 never runs
```

Note that bob's raw leak score is *better* than alice's, yet alice wins: both
saturate at 1.000 once normalised, and stage 2 decides. That is the single
most useful thing to understand about the scoring.

## Evaluating your model

The full workflow, in the order a model author works:

```bash
# 1. Build probe sets from dated facts, calibrated against a real model.
#    Calibration places `epsilon` by MEASURING a model that respects the
#    cutoffs — an uncalibrated corpus usually separates nothing.
wigin-tllm corpus --facts facts.json --out ./corpus \
    --calibrate-with local:./reference-2013

# 2. Generate the stage-2 completion prompts (fresh per round, unlearnable).
wigin-tllm prompts --data ./corpus

# 3. Stage 1 — does my model respect its cutoff?
#    --against also runs the SVD copy-gate, as a real evaluation does.
wigin-tllm consistency --submission models.json --data ./corpus \
    --against chronogpt

# 4. Is it too close to something published? Are my own years distinct?
wigin-tllm similarity --model local:./my-model --against chronogpt
wigin-tllm similarity --pairwise local:./m-2013,local:./m-2014,local:./m-2015

# 5. Stage 2 — how good is it, against references?
wigin-tllm quality --submission models.json --data ./corpus \
    --against chronogpt --judge openai

# Or everything at once, and the final score:
wigin-tllm benchmark --submission models.json --data ./corpus \
    --against chronogpt --judge openai
```

- `--submission models.json` is a `{year: model}` manifest — one model per
  year, the number that matters. `--model` plus `--years` scores a single
  checkpoint instead; a partial check is always flagged, because a real
  evaluation counts every missing year as worst-possible.
- Model references are `owner/repo@<40-char-commit-sha>` (HuggingFace,
  pinned) or `local:/path/to/dir`.
- `--against` names the reference models: `chronogpt` (the published
  baselines), a JSON file, or a comma-separated list.

**Skip step 1** by starting from [`examples/sample/`](examples/sample/): a
ready-made corpus over cutoffs 2013–2024, built from 52 real dated facts with
production-style thresholds, plus static stage-2 prompts and a manifest
template. Calibrate it before trusting the numbers — see
[its README](examples/sample/README.md).

## Configuration

Every knob lives in one dataclass, loadable from JSON (`--config file.json`)
or environment (`WIGIN_TLLM_MIN_EVAL_SCORE=-3.5`). The defaults mirror a
production evaluation; the full table is in
[docs/scoring.md](docs/scoring.md#configuration-reference).

| The knobs you are most likely to touch | Default |
|---|---|
| `min_eval_score` — the bar stage 1 must clear | −3.0 |
| `leak_weight` / `quality_weight` — the final blend | 0.7 / 0.3 |
| `max_parameters` / `max_model_bytes` | 2B / 10 GiB |
| `known_threshold` / `unknown_threshold` — per-side probe tolerance | 0.25 both |
| `svd_threshold` — below this spectral distance = a copy | 0.01 |

## Library use

```python
from wigin_tllm import Corpus, EvaluationConfig, benchmark
from wigin_tllm.scoring.baselines import CHRONOGPT_BASELINES
from wigin_tllm.scoring.judge import OpenAIJudge

report = benchmark(
    {"2015": "local:./my-model"},
    Corpus("./corpus"),
    config=EvaluationConfig(),
    references=CHRONOGPT_BASELINES,   # {year: [refs]} or a plain list
    judge=OpenAIJudge(),
)
print(report.consistency.leak_score, report.quality.win_rate, report.final_score)
```

## Extending

| Extension point | Base class | Ships with |
|---|---|---|
| Who judges quality | `Judge` | `OpenAIJudge`, `ReferenceOverlapJudge`, `ScriptedJudge` |
| Where prompts come from | `PromptGenerator` | `OpenAIPromptGenerator`, `StaticPromptGenerator` |
| How completions are produced | `CompletionProvider` | `ModelCompletionProvider`, `StaticCompletionProvider` |
| Custom architectures | `models/architectures/` | `miniformer`, `nanochrono`, `chronogpt` |

Models are always loaded with `trust_remote_code=False` — no code from a
model directory is ever executed. A custom architecture must be added to
`models/architectures/` and reviewed.

## Tests and layout

```bash
pytest              # 181 tests, ~6s, fully stubbed
pytest -m slow      # real weights, real forward passes
```

```
wigin_tllm/
├── corpus.py        build and calibrate probe sets
├── benchmark/       the stages: artifact → consistency → similarity → quality
├── scoring/         the measurements themselves (probes, spectra, duels)
├── models/          load weights behind one uniform interface
├── report.py        rendering
├── config.py        every knob, one dataclass
└── cli.py
```

## License

MIT — see [LICENSE](LICENSE). The ChronoGPT architecture in
`wigin_tllm/models/chronogpt.py` is vendored from
[manelalab/chrono-gpt](https://huggingface.co/manelalab)
([arXiv:2510.11677](https://arxiv.org/abs/2510.11677)).
