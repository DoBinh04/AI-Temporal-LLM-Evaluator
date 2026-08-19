# Dev corpus — cutoff 2022

A development dataset for **stage 1** only: one cutoff year, 2022, with a
50-probe set on each side.

```
facts.json                          100 dated facts, 2015-2025
corpus/years.json                   [2022]
corpus/benchmarks/2022/known.json   50 probes from 2015-2022 — the model MUST recognise these
corpus/benchmarks/2022/unknown.json 50 probes from 2023-2025 — it must NOT
corpus-calibrated/                  the same probes with a MEASURED epsilon — use this one to judge a model
```

`examples/sample/` stays the shipped demo corpus (12 cutoffs, 4 facts a year);
this one trades year coverage for depth, so a verdict is not decided by a
single probe crossing epsilon. The two sets share no phrases.

## Use it

```bash
wigin-tllm consistency --model local:./checkpoints/my-model --years 2022 \
    --data data/dev/corpus --config examples/sample/config.json
```

There is one year in scope, so the leak score *is* the 2022 year score and it
must be below −3.0 to qualify. Stage 2 is not covered here: there is no
`completion_prompts.json`, so use `examples/sample/corpus` for quality runs.

## Rebuild

`facts.json` is the source; the probe sets are derived. After editing it:

```bash
wigin-tllm corpus --facts data/dev/facts.json --out data/dev/corpus --years 2022 --config examples/sample/config.json
```

Facts keep their real years, so other cutoffs come from the same file —
`--years 2018-2024` builds a probe pair per year instead (the sets get
lopsided at the extremes: 2015-2022 supplies the whole `known` side).

## Two corpora: default epsilon vs calibrated

`corpus/` ships the default `epsilon` **−11.51**, placed by convention — it
exercises the pipeline but separates nothing (see CALIBRATION.md). To judge a
model use **`corpus-calibrated/`**, whose `epsilon` (−8.8792) was *measured*
against the clean reference `manelalab/chrono-gpt-v1-20221231` (safetensors
revision `4d37df72…`, verified bit-identical to the pinned main revision):
reference known rate 94%, unknown rate 4%, verdict *separates*. The
acceptance run in CALIBRATION.md shows the clean 2022 checkpoint PASSing on it
while the 2023/2024 checkpoints FAIL as leakers with a monotone signal.

To re-place epsilon after editing the facts:

```bash
wigin-tllm corpus --facts data/dev/facts.json --out data/dev/corpus-calibrated --years 2022 \
    --config examples/sample/config.json --device cuda \
    --calibrate-with manelalab/chrono-gpt-v1-20221231@4d37df723313ff0c156795002fc0abc30de6abf6
```

## How the facts were chosen

Generated and then verified by independent passes, against the probe-quality
rules in [docs/scoring.md](../../docs/scoring.md): a phrase must be unknowable
before its year (nothing pre-announced) and not derivable from its own prompt.
On top of that, every candidate was **pre-screened against the clean 2022
reference model**: an `unknown` probe the clean model still scored highly is
guessable from the prompt by priors alone (city-of-the-named-country,
most-famous-musician, first-name-to-surname momentum) and was replaced; a
`known` probe the reference could not recognise is too obscure for a 1.5B
model, or sits in the last weeks before the cutoff where training data thins
out, and was replaced too. `tools/dump_scores.py` regenerates the per-probe
scores behind that screen.

Every phrase is 1–2 GPT-2 BPE tokens. The probe score is a *summed*
log-probability, so a scalar epsilon is only comparable across probes of
similar token length — the round-1 failure in CALIBRATION.md is what mixed
1–7-token phrases do to it. `tests/test_dev_corpus.py` pins the invariants:
no phrase appears twice, none appears in `examples/sample/`, every year
carries at least six facts, and the phrase-length distribution is matched
across the two sides.
