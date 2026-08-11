# Wigin TLLM

Benchmark a language model for **chronological consistency**: does a model
built with a cutoff of year *Y* actually behave as though it has never seen
anything after *Y*?

Standard language models are trained on text from every period at once. Used
for backtesting or historical analysis they suffer from **lookahead bias** — a
model asked to reason about 2015 has already read about everything that
happened since. This measures whether a model genuinely respects a temporal
boundary, and how good it is once it does.

Everything here runs on your own machine against your own models.

---

## What gets measured

```
final = 0.7 x consistency + 0.3 x quality
```

**Consistency (stage 1)** — for each cutoff year the model faces two probe
sets and is scored on the log-probability it assigns to each continuation:

| Probe set | Content | Requirement |
|---|---|---|
| `unknown` | facts from **after** the cutoff | must **not** recognise them |
| `known` | facts from **before** the cutoff | **must** recognise them |

```
year score = median(unknown) - median(known)     # more negative is better
leak score = mean over EVERY year in scope       # a missing year scores worst
```

Both probe sets are load-bearing. Without `unknown` a model that memorised
everything passes; without `known` an empty model that knows nothing passes.

Stage 1 also runs the production gauntlet: size and parameter limits, pinned
revisions, an optional per-model time budget, and — when references are given
— an **SVD gate** that scores a year worst-possible if its model is
spectrally a copy of that year's reference.

**Quality (stage 2)** — the model continues a set of incomplete texts, and
those completions are judged head-to-head against reference models. The share
of duels won is the quality score. Quality has no absolute scale, so this
needs something to compare against.

**Qualification gates stage 2.** A submission whose leak score does not clear
`min_eval_score` never enters the duels and scores **0.0** outright — no
amount of quality buys back a failed consistency check.

**Similarity** — separately, how close is the model to a published one? The
comparison is between *singular value spectra*, not weights: a copy disguised
as `W' = P·W·Q` with orthogonal `P`, `Q` has almost zero cosine similarity to
the original but an identical spectrum. The same primitive powers the stage-1
gate and `--pairwise` dedup over a set of models.

Full detail, with every threshold and edge case: [docs/scoring.md](docs/scoring.md).
What to actually optimise: [docs/model-guide.md](docs/model-guide.md).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pip install -e ".[chronogpt]"   # scoring native ChronoGPT checkpoints
pip install -e ".[openai]"      # LLM judge and prompt generation
```

Python >= 3.10. CPU is enough for small models; a GPU is strongly recommended
at production sizes.

Anything that talks to an LLM — `--judge openai`, `--generate-prompts openai`,
the `prompts` command — reads its key from the environment. The easiest way
is a `.env` file in the working directory, which the CLI loads at startup:

```bash
cp .env.example .env    # then put your key in it
```

```dotenv
OPENAI_API_KEY=sk-...
#JUDGE_MODEL=gpt-4o-mini     # optional: judge model override
#PROMPT_MODEL=gpt-4o-mini    # optional: prompt-generation model
#OPENAI_BASE_URL=...         # optional: any OpenAI-compatible endpoint
```

A variable set in the real environment always beats the file, and `.env` is
gitignored so the key stays out of version control. The CLI refuses to start
a judged run without the key rather than failing mid-tournament. Everything
else — consistency, similarity, `--judge overlap`, the shipped static
prompts, the offline demo — needs no key at all.

## The workflow

```bash
# 1. build probe sets from your dated facts, calibrated against a real model
wigin-tllm corpus --facts facts.json --out ./corpus \
    --calibrate-with local:./reference-2013

# 2. generate the stage-2 completion prompts
wigin-tllm prompts --data ./corpus

# 3. does my model respect its cutoff?
#    (--against also runs the SVD copy-gate, as a real evaluation does)
wigin-tllm consistency --submission models.json --data ./corpus \
    --against chronogpt

# 4. is it too close to something published? are my own years distinct?
wigin-tllm similarity --model local:./my-model --against chronogpt
wigin-tllm similarity --pairwise local:./m-2013,local:./m-2014,local:./m-2015

# 5. how good is it, against references?
wigin-tllm quality --submission models.json --data ./corpus \
    --against chronogpt --judge openai

# ...or all of it, and the final score
wigin-tllm benchmark --submission models.json --data ./corpus \
    --against chronogpt --judge openai
```

Every scoring command takes either `--submission models.json` (one model per
year — the number that matters) or `--model` plus `--years` for a single
checkpoint. A partial check is always flagged, because a real evaluation
counts every missing year as worst-possible.

## Try it

The demo builds a corpus, trains six tiny models on CPU and benchmarks four
of them — offline, no API keys, about two minutes:

```bash
python examples/run_local.py --rebuild
```

```
=== Corpus calibration ===
  year      epsilon    known   unknown  threshold  verdict
  2013      -7.6002  100.0%    13.3%     25.0%  separates
  2014      -6.7433   50.0%    12.5%     25.0%  separates
  2015      -5.7907   33.3%    11.1%     25.0%  separates
Calibrated. Tightest margin 8.3% (year 2015) — comfortably clear of the threshold.

#  alice — honest and well trained
  leak score       -10.1264   (threshold -3.0000) — clears it
  win rate            0.500   over 6 prompts against 2 reference(s)
Final score  0.8500   = 0.7 x 1.000 + 0.3 x 0.500

#  bob — honest, but never saw part of the material
  leak score       -10.3043   (threshold -3.0000) — clears it
  win rate            0.000   over 6 prompts against 2 reference(s)
Final score  0.7000   = 0.7 x 1.000 + 0.3 x 0.000

#  mallory — trained on facts from after its cutoff
  leak score         0.0000   (threshold -3.0000) — below the bar
Final score  0.0000   — stage 1 not cleared, so stage 2 never runs and the score is 0
#  null-model — barely trained at all
Final score  0.0000   — stage 1 not cleared, so stage 2 never runs and the score is 0
```

Note that bob's raw leak score is *better* than alice's, yet alice wins: both
saturate at 1.000 once normalised, and stage 2 decides. That is the single
most useful thing to understand about the scoring.

## A ready-made corpus

[`examples/sample/`](examples/sample/) ships a corpus over the production
year range (cutoffs 2013–2024), built from 52 real dated facts with
production-style thresholds (known 0.70 / unknown 0.10), plus 16 static
stage-2 prompts and a manifest template:

```bash
cp examples/sample/models.example.json models.json   # then edit

wigin-tllm benchmark --submission models.json \
    --data examples/sample/corpus --config examples/sample/config.json \
    --against chronogpt --judge openai
```

The shipped `epsilon` (−11.51) suits models with a ~50k-token vocabulary and
was not placed by measurement — calibrate before trusting the numbers, and
add facts before grading anything seriously: four per year smoke-tests a
pipeline, it does not grade a model. Details in
[examples/sample/README.md](examples/sample/README.md).

## Calibration is the part that matters

A probe set that is not calibrated measures nothing. Set `epsilon` too strict
and every model looks ignorant; too loose and every model looks like a leaker.
Worse, if the honest hit rate lands near `threshold`, floating-point noise
moving one probe across `epsilon` flips a whole year between pass and fail.

`wigin-tllm corpus --calibrate-with` places `epsilon` by **measuring** a model
that genuinely respects the cutoffs, and reports how much room is left before
a verdict would turn on a single probe.

## Library use

```python
from wigin_tllm import Corpus, EvaluationConfig, benchmark
from wigin_tllm.scoring.judge import OpenAIJudge

report = benchmark(
    {"2015": "local:./my-model"},
    Corpus("./corpus"),
    config=EvaluationConfig(),
    references="chronogpt",
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

Models are always loaded with `trust_remote_code=False` — no code from a model
directory is ever executed. A custom architecture must be added to
`models/architectures/` and reviewed.

## Tests

```bash
pytest                            # 173 tests, ~6s
pytest -m slow                    # real weights, real forward passes
```

## Layout

```
wigin_tllm/
├── corpus.py        build and calibrate probe sets
├── benchmark/       artifact -> consistency -> similarity -> quality
├── scoring/         the measurements themselves
├── models/          load weights behind one uniform interface
├── report.py        rendering
└── cli.py
```

## License

MIT — see [LICENSE](LICENSE). The ChronoGPT architecture in
`wigin_tllm/models/chronogpt.py` is vendored from
[manelalab/chrono-gpt](https://huggingface.co/manelalab)
([arXiv:2510.11677](https://arxiv.org/abs/2510.11677)).
