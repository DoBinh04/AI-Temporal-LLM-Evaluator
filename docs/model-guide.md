# Building a model that scores well

Ordered by leverage, not by mechanism. For the formulas themselves see
[scoring.md](scoring.md).

## 1. The score is a gap, not a level

```
year score = median(unknown) - median(known)
```

Cutting the future out aggressively is only half the job. A model that also
forgets its own era scores exactly as badly as one that leaks:

| | median unknown | median known | score |
|---|---|---|---|
| Clean | −8.0 | −2.0 | **−6.0** |
| Empty | −9.0 | −9.0 | 0.0 |
| Leaking | −2.0 | −2.0 | 0.0 |

Widen the gap from **both** ends: confident about its own period, ignorant of
what came after.

## 2. Consistency saturates — know when to stop

`normalised` reaches 1.0 at a leak score of `leak_best_score` (−6.0 by
default). At −8, or −12, it is still 1.0. Every hour spent past that point
earns nothing.

Once you are there, the remaining 30% is the only lever left. The demo shows
this directly: `bob` has a better raw leak score than `alice` and still ranks
below, because both are saturated and stage 2 decides.

`wigin-tllm benchmark` says so explicitly when you cross the line.

## 3. A missing year is a cliff, not a slope

The mean divides by **every** year in scope. A year with no model contributes
the worst possible score.

| Perfect years of 12 | leak score | clears −3.0? |
|---|---|---|
| 7 | −3.50 | yes |
| 6 | −3.00 | **no** — the test is strictly below |

Benchmark with `--submission models.json`, not a single `--model`, or the
number you get will be better than the one you would really score. The tool
warns when the check is partial.

## 4. Stage 2 scores completions, not answers

The prompts are incomplete text: *"The process by which plants convert
sunlight into sugar is called"*. A model that replies *"Sure! That process is
called photosynthesis…"* loses to one that simply continues the sentence — the
judge scores "natural continuation of the prompt".

Prompts are generated fresh each round across eight categories and are
deliberately timeless, so there is nothing to memorise and no risk of a 2013
model being asked about 2020.

## 5. Stage 2 does not open until stage 1 clears

The final score is not a blend of two independent halves. A leak score that
does not clear `min_eval_score` means no duels and a final score of 0.0 —
quality work is worthless until consistency is fixed. Fix stage 1 first;
the standalone `quality` command still gives you the number in the meantime.

## 6. Do not start from a published checkpoint

Similarity compares singular value spectra, so lightly fine-tuning a released
model is caught, and rotating or permuting the weights does not help:
`σ(P·W·Q) = σ(W)`. In an evaluation with references, this is a **gate**: a
year whose model reads as a copy of that year's reference scores
worst-possible before a single probe runs.

```bash
wigin-tllm similarity --model local:./my-model --against chronogpt
wigin-tllm similarity --pairwise local:./m-2013,local:./m-2014,local:./m-2015
```

Run both before you submit: the first against the published models, the
second across your own year-models to check they are actually diverging from
each other.

## Traps that cost points quietly

- **Custom architectures must be merged first.** Models load with
  `trust_remote_code=False`, so an architecture that is not already in
  `models/architectures/` cannot be scored at all.
- **`epsilon` is tied to vocabulary size.** An uninformed guess scores about
  `ln(1/V)`. Building your own probe sets without calibrating against a real
  model will mislead you completely — use `wigin-tllm corpus --calibrate-with`.
- **A verdict near the threshold is a verdict decided by noise.** If the
  honest hit rate sits close to `threshold`, one probe crossing `epsilon`
  flips the year. Calibration reports the margin; keep it wide.
- **Quality against a single reference is ambiguous.** A draw scores for
  neither side, so a win rate of 0 could mean losing everything or drawing
  everything. Use two or more references.

## What you do not need to know

How the reports are rendered, how the spectra are stored, how prompts are
generated. Understanding the two probe sets, the gap they measure, and the
saturation point is enough to make every decision that matters.
