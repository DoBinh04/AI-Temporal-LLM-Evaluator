# Dev corpus — cutoff 2022

A development dataset for **stage 1** only: one cutoff year, 2022, with a
50-probe set on each side.

```
facts.json                          100 dated facts, 2015-2025
corpus/years.json                   [2022]
corpus/benchmarks/2022/known.json   50 probes from 2015-2022 — the model MUST recognise these
corpus/benchmarks/2022/unknown.json 50 probes from 2023-2025 — it must NOT
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

## Epsilon is not calibrated

The shipped `epsilon` is the default **−11.51**, placed by convention rather
than by measurement — the same caveat as the sample corpus. Numbers from this
corpus are for exercising the pipeline, not for judging a model. To place it
by measurement:

```bash
wigin-tllm corpus --facts data/dev/facts.json --out data/dev/corpus --years 2022 --config examples/sample/config.json --calibrate-with manelalab/chrono-gpt-v1-20131231@8e3e454b59a27d96ed3773f5c58a10e84e4f3f12
```

## How the facts were chosen

Generated and then verified by independent passes, against the probe-quality
rules in [docs/scoring.md](../../docs/scoring.md): a phrase must be unknowable
before its year (nothing pre-announced) and not derivable from its own prompt.
Roughly a third of the candidates were rejected, most of them for prompts that
handed over a famous first name and asked for the surname — those score the
name, not the news.

Two invariants matter beyond correctness, and `tests/test_dev_corpus.py` pins
both: no phrase appears twice, and the phrase-length distribution is matched
across the two sides. The probe score is a *summed* log-probability, so
systematically longer phrases on one side would shift
`median(unknown) − median(known)` on token count rather than on knowledge.
