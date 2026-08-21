# Dev corpus — cutoff 2022 (hardened)

A development dataset for **stage 1** only: one cutoff year, 2022, with a
50-probe set on each side, deliberately **hardened** so that a clean model no
longer saturates the consistency score (see BAO_CAO_V2.md).

```
facts-known.json                              50 dated facts, 2015-2022 — the `known` source
facts-unknown.json                            50 dated facts, 2023-2025 — the `unknown` source
corpus-calibrated/years.json                  [2022]
corpus-calibrated/benchmarks/2022/known.json  50 probes from 2015-2022 — the model MUST recognise these
corpus-calibrated/benchmarks/2022/unknown.json 50 probes from 2023-2025 — it must NOT
raw_scores.json                               per-probe scores of the three chrono checkpoints
```

The two facts files are the source of truth, split by which side of the 2022
cutoff each fact falls on; both use the standard `{"facts": [...]}` schema,
so each is also a valid `--facts` input on its own.

`examples/sample/` stays the shipped demo corpus (12 cutoffs, 4 facts a year);
this one trades year coverage for depth, so a verdict is not decided by a
single probe crossing epsilon. The two sets share no phrases.

Earlier revisions also carried a `corpus/` twin built with the conventional
default `epsilon` −11.51; it exercised the pipeline but separated nothing
(CALIBRATION.md, round 1) and has been removed. **`corpus-calibrated/` is the
only probe set here** — its `epsilon` (−4.5314) was *measured* against the
clean reference `manelalab/chrono-gpt-v1-20221231` (safetensors revision
`4d37df72…`, verified bit-identical to the pinned main revision): reference
known rate 78%, unknown rate 4%, verdict *separates*.

## Use it

```bash
wigin-tllm consistency --model local:./checkpoints/my-model --years 2022 \
    --data data/dev/corpus-calibrated --config examples/sample/config.json
```

There is one year in scope, so the leak score *is* the 2022 year score and it
must be below −3.0 to qualify. Stage 2 is not covered here: there is no
`completion_prompts.json`, so use `examples/sample/corpus` for quality runs.

## What "hardened" means

The corpus is calibrated so the clean 2022 reference still PASSes, but no
longer maxes out the normalised score (BAO_CAO_V2.md has the full acceptance
run):

| model (cutoff) | leak score | normalised | verdict |
|---|---|---|---|
| 20221231 (clean) | −4.5655 | 0.522 | **PASS** |
| 20231231 (saw 2023) | 0.0 | — | **FAIL — leaker** |
| 20241231 (saw 2023–24) | 0.0 | — | **FAIL — leaker** |

Two levers, both purely in the data:

* **`known` is harder.** Front-page anchors every 1.5B model nails at ≈ 0
  log-prob (Patriots, Biden, Nadal) were replaced by second-tier but
  well-documented events (Oroville, Bucha, Kobani, Grenfell, Lochte, Manabe
  …), dragging the known median from −3.02 towards the recognition
  threshold.
* **`unknown` keeps the anti-prior rule but not the obscurity tail.** Probes
  stay unguessable from the prompt (every one scores below −3.5 per token on
  the clean reference), while ultra-obscure ones that the model scored below
  ≈ −14 were left out; with them, the unknown median sat so deep that the leak
  score saturated at the −6.0 normalisation cap and difficulty changes on the
  known side earned nothing.

## Rebuild

`facts-known.json` + `facts-unknown.json` are the source; the probe sets are
derived. After editing either, merge the two and rebuild:

```bash
python -c "import json; m=[json.load(open(f'data/dev/facts-{s}.json'))['facts'] for s in ('known','unknown')]; \
json.dump({'facts': m[0]+m[1]}, open('/tmp/dev-facts.json','w'), indent=2)"
wigin-tllm corpus --facts /tmp/dev-facts.json --out data/dev/corpus-calibrated --years 2022 \
    --config examples/sample/config.json --device cuda \
    --calibrate-with manelalab/chrono-gpt-v1-20221231@4d37df723313ff0c156795002fc0abc30de6abf6
```

Facts keep their real years, so other cutoffs come from the same files —
`--years 2018-2024` builds a probe pair per year instead (the sets get
lopsided at the extremes: 2015-2022 supplies the whole `known` side).

## How the facts were chosen

Generated and then verified against the probe-quality rules in
[docs/scoring.md](../../docs/scoring.md): a phrase must be unknowable before
its year (nothing pre-announced) and not derivable from its own prompt. On
top of that, every candidate was **pre-screened against the clean 2022
reference model** (`tools/dump_scores.py` regenerates the per-probe scores):
an `unknown` probe the clean model scored above −3.5 per token is guessable
from the prompt by priors alone (geography, name momentum, famous-default)
and was rejected; `known` probes were then *selected by measured difficulty*
— hard enough to pull the median down, while at least 39/50 stay above the
calibrated epsilon so the corpus still separates with margin.

Every phrase is 1–2 GPT-2 BPE tokens. The probe score is a *summed*
log-probability, so a scalar epsilon is only comparable across probes of
similar token length — the round-1 failure in CALIBRATION.md is what mixed
1–7-token phrases do to it. `tests/test_dev_corpus.py` pins the invariants:
no phrase appears twice, none appears in `examples/sample/`, every year
carries at least six facts, each side file holds only its own years, the
probe sets stay in sync with the facts files,
and the phrase-length distribution is matched across the two sides.
