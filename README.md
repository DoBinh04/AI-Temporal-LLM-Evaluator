# Wigin TLLM

Scores language models on **chronological consistency**: does a model built
with a cutoff of year *Y* actually behave as though it has never seen anything
after *Y*?

Standard language models are trained on text from every period at once. Used
for backtesting or historical analysis they suffer from **lookahead bias** — a
model asked to reason about 2015 has already read about everything that
happened since. This tool measures whether a model genuinely respects a
temporal boundary, and ranks a field of submissions accordingly.

---

## How scoring works

```
submissions ─► Stage 1: consistency probes ─► anti-copy ─► Stage 2: quality duels ─► rank
```

**Stage 1 — chronological consistency.** For each cutoff year a model faces
two probe sets and is scored on the log-probability it assigns to each
continuation:

| Probe set | Content | Requirement |
|---|---|---|
| `unknown` | facts from **after** the cutoff | must **not** recognise them |
| `known` | facts from **before** the cutoff | **must** recognise them |

Both checks are needed. Without `unknown` a model that memorised everything
would pass; without `known` an empty model that knows nothing would pass.

```
year score = median(unknown) − median(known)      # more negative is better
leak score = mean(year scores over all years)     # missing years score worst
```

**Anti-copy.** Three independent layers, each catching what the previous one
cannot:

1. **Weight hash** — byte-identical resubmissions; first submitter keeps them.
2. **SVD baseline gate** — lightly-modified copies of a published reference
   model.
3. **SVD pairwise dedup** — submitters copying each other.

Layers 2 and 3 compare *singular value spectra* rather than the weights
themselves. A copy disguised as `W' = P·W·Q` with orthogonal `P`, `Q` has
almost zero cosine similarity to the original but an identical spectrum, so
the disguise fails.

**Stage 2 — quality.** Qualified models continue the same set of incomplete
texts; every pair is judged head-to-head over two cutoff years. The share of
duels won is the quality score. Completion positions are swapped at random and
mapped back, so a judge that favours whichever it sees first cannot decide the
outcome.

The prompts are **generated fresh each round by an LLM**, across eight
categories (reading comprehension, world knowledge, commonsense and causal
reasoning, logical inference, temporal reasoning, language understanding and
modelling). A fixed prompt file would be learnable — a model tuned to it would
score well without being better. Prompts are deliberately timeless so a 2013
model is not asked about 2020.

**Final score.**

```
final = 0.7 × normalised_leak_score + 0.3 × quality_win_rate
```

Submissions with a positive final score are ordered best-first and given a
1-based rank. A score of 0 means the submission failed outright; it is
reported with the reason, but carries no rank.

Full details: [docs/scoring.md](docs/scoring.md).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# optional extras
pip install -e ".[chronogpt]"   # scoring native ChronoGPT checkpoints
pip install -e ".[openai]"      # the OpenAI-backed quality judge
```

Requires Python ≥ 3.10. CPU is enough for small models; a GPU is strongly
recommended at production sizes.

## Try it

The demo builds a small synthetic world, trains seven real (tiny) models on
CPU, and runs the complete pipeline offline — no network, no API keys:

```bash
python examples/run_local.py --rebuild
```

```
rank   submitter            leak    norm   quality    final  status
1      alice             -6.6303   1.000     1.000   1.0000  qualified
2      bob              -10.0583   1.000     0.000   0.7000  qualified
-      copycat            0.0000   0.000     0.000   0.0000  duplicate_weights
-      eve                0.0000   0.000     0.000   0.0000  duplicate_of_earlier_submission
-      mallory            0.0000   0.000     0.000   0.0000  failed_consistency_check
-      null-model         0.0000   0.000     0.000   0.0000  failed_consistency_check
-      plagiarist         0.0000   0.000     0.000   0.0000  svd_gate_failed
```

Each submitter exercises a different branch: `alice` is honest and
well-trained, `bob` is honest but never saw part of the material (clean, yet
loses the duels), `mallory` trained on future facts, `null-model` learned
nothing, `copycat` is byte-identical to alice, `eve` is alice plus
imperceptible noise, and `plagiarist` copied the published baseline.

Note that bob's leak score is *better* than alice's, and both saturate at
1.000 once normalised — chronological consistency cannot separate them at all.
Stage 2 is what puts alice first.

(Leak scores vary in the third decimal between runs: CPU float reductions are
not bit-reproducible. Ranks, duel outcomes and statuses are stable.)

## Check your own model

Before submitting anything, run the pre-flight check. It looks at one model in
isolation — no anti-copy, no ranking — so you can run it as often as you like
while iterating:

```bash
# will my artefact be accepted at all?  (no probe data needed)
wigin-tllm check --model local:./my-model

# ...and how would it score?
wigin-tllm check --model local:./my-model --data ./my-data --years 2015
```

```
Artefact
------------------------------------------------------------------------
  [ok  ]  weight size                      812.4 MB (limit 10.0 GiB)
  [ok  ]  architecture                     llama
  [ok  ]  loads without trust_remote_code  on cuda
  [ok  ]  parameters                       1355.8M (limit 2.00B)
  [ok  ]  tokenizer                        8 tokens, round-trip fine
  [ok  ]  scoring                          prefers the sensible continuation in 4/4 probes
  [ok  ]  generation                       e.g. "Paris, the capital of France"

Chronological consistency
------------------------------------------------------------------------
  2015  [FAIL]  score +0.0000   normalised 0.000
      known    median   -0.8120   28/30 above epsilon -11.51   (must recognise)
      unknown  median   -6.4001   9/30 above epsilon -11.51   (must not recognise)
      recognises post-cutoff facts as readily as pre-cutoff ones — the
      training data reaches beyond the cutoff
```

Every failure comes with the numbers behind it and a diagnosis, so you know
whether to fix the training cutoff, train longer, or recalibrate the probes.
Exit code is non-zero when something blocks acceptance, so it drops into CI.

## Run an evaluation

```bash
# generate this round's stage-2 prompts, then score
wigin-tllm prompts --data ./my-data --per-category 13
wigin-tllm run --data ./my-data --judge openai --baselines chronogpt

# or generate them inline as part of the run
wigin-tllm run --data ./my-data --judge openai --generate-prompts openai

wigin-tllm validate --submission models.json --years 2013-2024
wigin-tllm show --data ./my-data --round 1
```

Generating prompts as a separate step writes them to
`completion_prompts.json`, so you can review the set and reproduce the round.
Both paths need `OPENAI_API_KEY`.

`--baselines` takes `chronogpt` (the bundled published references) or a path
to a JSON file mapping year to model references. Without it the SVD baseline
gate has nothing to compare against and passes everything — the other two
anti-copy layers still apply.

Or from Python:

```python
from wigin_tllm import EvaluationConfig, run_evaluation
from wigin_tllm.datasource import LocalDataSource
from wigin_tllm.scoring.judge import OpenAIJudge

results = run_evaluation(
    LocalDataSource("./my-data"),
    config=EvaluationConfig(leak_weight=0.7, quality_weight=0.3),
    judge=OpenAIJudge(),
)
for s in results.submitters:
    print(s.rank, s.submitter_id, s.final_score, s.disqualified_reason)
```

Data directory layout is documented in
[docs/data-format.md](docs/data-format.md).

## Docker

```bash
docker build -t wigin-tllm .
docker run --rm -v "$PWD/my-data:/data" wigin-tllm run --data /data
```

## Extending

Every boundary is an interface, so the pieces most likely to differ per
deployment can be replaced without touching the scoring code:

| Extension point | Base class | Ships with |
|---|---|---|
| Where inputs come from | `DataSource` | `LocalDataSource`, `HttpDataSource`, `InMemoryDataSource` |
| Who judges quality | `Judge` | `OpenAIJudge`, `ReferenceOverlapJudge`, `ScriptedJudge` |
| Where stage-2 prompts come from | `PromptGenerator` | `OpenAIPromptGenerator`, `StaticPromptGenerator` |
| How completions are produced | `CompletionProvider` | `ModelCompletionProvider`, `StaticCompletionProvider` |
| Which models count as baselines | `SvdGate(baselines=…)` | `CHRONOGPT_BASELINES`, or none |
| Custom architectures | `models/architectures/` | `miniformer`, `nanochrono`, `chronogpt` |

Models are always loaded with `trust_remote_code=False`. Custom architectures
must be added to `models/architectures/` and reviewed, never executed out of a
submitted directory — see [docs/architecture.md](docs/architecture.md).

## Tests

```bash
pytest                            # 214 tests, ~20s
pytest tests/test_end_to_end.py   # real weights, real forward passes
```

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Module map, data flow, design decisions |
| [docs/scoring.md](docs/scoring.md) | Every formula, threshold, and edge case |
| [docs/data-format.md](docs/data-format.md) | Input/output file formats |

## License

MIT — see [LICENSE](LICENSE). The ChronoGPT architecture in
`wigin_tllm/models/chronogpt.py` is vendored from
[manelalab/chrono-gpt](https://huggingface.co/manelalab)
([arXiv:2510.11677](https://arxiv.org/abs/2510.11677)).
