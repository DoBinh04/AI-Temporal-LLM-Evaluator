# Scoring reference

Every formula the pipeline applies, with its rationale and edge cases.

To see these numbers for your own model, run `wigin-tllm benchmark`. It
reports each probe set's median, hit count and verdict, a diagnosis when a
year fails, and the headline scores. For what to *do* about them, see
[model-guide.md](model-guide.md).

---

## Stage 1 — chronological consistency

### The measurement

For a probe `(prompt, phrase)` the model is scored on the summed
log-probability it assigns to the phrase given the prompt, under teacher
forcing:

```
score(prompt, phrase) = Σ log P(tokenᵢ | prompt, token₁…tokenᵢ₋₁)
```

Implementation notes (`scoring/leak.py`):

- Probes are scored **column-wise**: one forward pass yields token *t* of
  every probe at once, so the number of forward passes is the longest phrase
  length, not the number of probes.
- Padding is on the right and no attention mask is used. This is safe because
  attention is causal: tokens after the position being read cannot influence
  it. The logit is read at `len(unpadded_sequence) - 1`.
- A BOS token is prepended when the model declares one.
- There is no KV cache; each step re-runs the growing sequence. Cost is
  `O(max_phrase_len × batch × T²)` and this dominates stage 1.

### The two probe sets

| Set | Content | Desired outcome |
|---|---|---|
| `unknown` | facts from **after** the cutoff | model does not recognise them |
| `known` | facts from **before** the cutoff | model recognises them |

`probe()` reports the same quantity for both — `recognised`: *did more than
`threshold` of the probes score above `epsilon`?* — and `assess_year()` reads
it in opposite directions:

```python
passed = (not unknown_result.recognised) and known_result.recognised
```

Both checks are load-bearing:

- Without `unknown`, a model trained on everything passes.
- Without `known`, an empty model that assigns low probability to *everything*
  passes. This is the "empty model" failure mode.

### Probe calibration

`threshold` and `epsilon` travel with the probe set, not the config, so they
can be tuned per year and per set.

| Field | Default | Meaning |
|---|---|---|
| `epsilon` | `-11.51` | log-probability above which a probe counts as recognised (≈ ln 1e-5) |
| `threshold` | `0.10` | tolerated fraction of recognised probes |

**`epsilon` must match your vocabulary size.** An uninformed guess scores
about `ln(1/V)`. At `V = 50000` that is ≈ −10.8, so −11.51 sits just below
guessing. At `V = 130` (the demo) guessing scores ≈ −4.9 and −11.51 is
unreachable — the demo therefore uses −3.0. Getting this wrong makes every
model look like a leaker or like an empty model.

### Year score

```
score(year) = median(unknown) − median(known)     if passed
score(year) = 0.0  (WORST_SCORE)                  otherwise
```

More negative is better: it is the gap between what the model has forgotten
and what it has retained.

| Archetype | median unknown | median known | score |
|---|---|---|---|
| Clean | −8.0 | −2.0 | **−6.0** ✅ |
| Empty | −9.0 | −9.0 | 0.0 ❌ |
| Leaker | −2.0 | −2.0 | 0.0 ❌ |

`WORST_SCORE = 0.0` is also assigned for: missing years, oversized models, too
many parameters, unpinned revisions, duplicate weights, gate failures,
timeouts, and load errors.

### The stage-1 gauntlet

Before a single probe runs, each model faces the checks a real evaluation
applies, in this order:

| Check | Config | On failure |
|---|---|---|
| weight size | `max_model_bytes` | all of its years score worst |
| pinned revision | `require_pinned_revision` | all of its years score worst |
| parameter count | `max_parameters` | all of its years score worst |
| SVD gate | references + `svd_threshold` | **that year** scores worst |
| time budget | `max_eval_seconds` | the years not reached score worst |

The **SVD gate** runs when references are supplied (`--against` on
`consistency`, or the references given to `benchmark`): each year's model is
compared spectrally against every reference for that year and judged on the
*minimum* distance — being far from one published variant does not excuse
copying another. A gated year fails before its probes are ever run. A year
with no references passes by construction.

The **time budget** clock starts when the model starts loading. Before each
year the elapsed time is checked; once the budget is spent, the remaining
years are skipped and count worst-possible in the mean. `None` (the default)
disables the budget.

### Leak score

```
leak_score = mean(year_scores over ALL years in scope)
```

The denominator is the full year count, not the number submitted. This is
what makes a missing year expensive:

| Perfect years (of 12) | leak score | clears −3.0? |
|---|---|---|
| 12 | −6.00 | ✅ normalises to 1.00 |
| 10 | −5.00 | ✅ 0.67 |
| 7 | −3.50 | ✅ 0.17 |
| 6 | −3.00 | ❌ (not strictly below) |

At least 7 of 12 perfect years are needed to qualify at all.

---

## Similarity

Compares the **singular value spectrum** of each 2D weight matrix rather than
the weights themselves.

```python
spectra[name] = svdvals(W)                       # per 2D weight matrix
k = max(1, int(len(sigma) * svd_top_ratio))      # top 25%
distance = mean over matrices of ||sigma_cand[:k] - sigma_ref[:k]||   # L2-normalised
too_close = distance < svd_threshold                                  # 0.01
```

**Why spectra, not weights.** A copy can be disguised as `W' = P.W.Q` with
orthogonal `P`, `Q`, compensated in an adjacent layer so behaviour is
unchanged. Cosine similarity of the raw weights collapses toward zero — the
disguise defeats it. Singular values are invariant under orthogonal
transforms, `sigma(P.W.Q) = sigma(W)`, so the spectrum still matches.
`tests/test_svd_gate.py` pins this property.

L2-normalising before comparison additionally defeats a global rescale.

**Known limitation.** After L2 normalisation a single-element slice is always
`[1.0]`, so any matrix with `min(shape) <= 4` contributes distance 0 at the
default `top_ratio`. Distances are averaged across all comparable matrices, so
this only matters for models built entirely from very small 2D tensors.

The same primitive is used three ways:

- **As a report** (`similarity --model X --against Y`): how close is my model
  to a published one? Informational; a `too_close` verdict marks the
  benchmark as not accepted.
- **As a stage-1 gate** (`consistency --against`, or `benchmark` with
  references): a year whose model reads as a copy of that year's reference
  scores worst-possible.
- **As pairwise dedup** (`similarity --pairwise a,b,c`): all pairs of a set
  are compared and, of any duplicate pair, the *first-listed* keeps its place
  — the order of the list is the precedence order. Use it to check that your
  own year checkpoints are actually different models.

---

## Clearing the bar

A model clears stage 1 when its mean year score is strictly below
`min_eval_score` (−3.0 by default).

### Normalisation

```
                    min_eval_score − score
normalised = clamp( ───────────────────────── , 0, 1 )
                  min_eval_score − leak_best_score
```

Defaults map −3.0 → 0.0 and −6.0 → 1.0. Clamping means `WORST_SCORE` can never
earn credit, and the formula stays monotonic even if the two bounds are
supplied the other way run.

---

## Stage 2 — quality

Needs a judge, a non-empty prompt set, and at least one reference model to
duel against; without all three only stage 1 can be scored.

### Prompts

Stage 2 does not ask questions — it hands the model **incomplete text to
continue**. That is what a language model does natively, so a base checkpoint
and an instruction-tuned one can be compared on the same footing.

Prompts are generated fresh for each run by an LLM across eight categories:

| Category | Tests |
|---|---|
| `reading_comprehension` | understanding a short passage |
| `language_understanding` | context, tone and intent |
| `world_knowledge` | science, geography, history |
| `commonsense_reasoning` | everyday physical and social sense |
| `language_modeling` | coherent narrative continuation |
| `causal_reasoning` | cause and effect |
| `logical_inference` | drawing a conclusion from premises |
| `temporal_reasoning` | order and sequence of events |

Two properties matter:

- **Fresh per run.** A fixed set would be learnable, and a model tuned to it
  would score well without being better.
- **Timeless.** No dates, no recent events. A prompt about 2019 is
  unanswerable for a model with a 2015 cutoff, and stage 2 measures quality,
  not chronology — that is stage 1's job.

Generation samples at `temperature=1.0` because the goal is variety, with one
request per category. Pass a seed to make a run reproducible. If generation
is configured and yields nothing the run aborts rather than quietly skipping
stage 2, which would change what the run measures.

### Year selection

The oldest year is always evaluated (comparability across runs), plus
`quality_year_samples - 1` drawn at random from the rest (so effort cannot be
concentrated on a predictable year). Set `quality_seed` for reproducibility.

### Duels

Every pair meets once per evaluated year. For each prompt the two completions
are presented in random order and the verdict is mapped back:

```python
swap = rng.random() < 0.5
verdict = {"a": "b", "b": "a", "tie": "tie"}[raw] if swap else raw
```

Judges — LLMs especially — tend to prefer whichever completion they see first.
Randomising position spreads that bias evenly instead of letting it decide
duels. `tests/test_quality.py` verifies that a judge which *always* says
"first" gives nobody a systematic advantage.

A duel is won by taking the majority of prompts; equal counts are a draw and
score for neither side.

```
win_rate = duels_won / (opponents faced)     per year, then averaged
```

### Reference opponents

Reference models can be entered into the tournament without being scored
themselves. They do two things:

- **They make quality measurable at all.** Quality has no absolute scale; a
  model on its own has nothing to be better than.
- **They anchor the scale.** Without one, a win rate says only "better than
  whoever else entered this run", which is not comparable between runs.

```bash
wigin-tllm quality --submission models.json --data ./corpus \
    --against chronogpt --judge openai
```

**Use at least two.** A drawn duel scores for neither side, so a win rate of
0 covers both "lost every duel" and "drew every duel". Against a single
opponent those are indistinguishable in the number; the record logged as
`1W-1D-0L` is what separates them. Two opponents also give the rate
real resolution — 0, 0.5 or 1 instead of just 0 or 1.

A model that is missing or fails to load yields empty completions: it loses
its duels rather than aborting the run.

### Judge hardening

`OpenAIJudge` pins `temperature=0` and a fixed `seed`, and constrains the
response to a strict JSON schema with `enum: [a, b, tie]` so there is no free
text to parse. It scores on factual accuracy, how naturally the text
continues, coherence, and knowledge demonstrated.

Completions are wrapped in `<completion>` tags with a system prompt
instructing the judge to treat their content as data, and truncated (500 chars
for the prompt, 300 per completion). A submitted model can be trained to emit
text aimed at the judge, so this is a real boundary, not decoration.

---

## Final score

```
final = leak_weight · normalised_leak + quality_weight · quality_win_rate
      = 0.7 · normalised_leak + 0.3 · win_rate
```

**Qualification gates stage 2.** A submission whose leak score does not clear
`min_eval_score` never enters the duels and its final score is **0.0**
outright — no amount of quality can buy back a failed consistency check. The
standalone `quality` command still measures a win rate regardless, for
authors who want the number while stage 1 is still being fixed.

Other special cases:

- Consistency not yet measured → no final score.
- Qualified but quality unmeasured (no judge, no references, or no prompts) →
  no final score; the report shows what consistency alone would contribute.

---

## Configuration reference

| Key | Default | Effect |
|---|---|---|
| `max_model_bytes` | 10 GiB | size limit, checked before download |
| `max_parameters` | 2×10⁹ | parameter limit, checked after load |
| `max_eval_seconds` | none | per-model stage-1 time budget (none = unlimited) |
| `min_eval_score` | −3.0 | the bar a model must clear |
| `leak_best_score` | −6.0 | score normalising to 1.0 |
| `quality_max_new_tokens` | 50 | completion length |
| `quality_year_samples` | 2 | cutoff years duelled |
| `quality_seed` | none | reproducible year draw and swaps |
| `leak_weight` / `quality_weight` | 0.7 / 0.3 | final-score blend |
| `svd_threshold` | 0.01 | minimum spectral distance |
| `svd_top_ratio` | 0.25 | fraction of the spectrum compared |
| `require_pinned_revision` | true | reject unpinned HF references |
| `device` | auto | `cpu` / `cuda` / `mps` |
| `probe_threshold` | 0.25 | probe hit-rate threshold when building a corpus |
| `known_threshold` | none | per-side override for `known` sets (e.g. 0.70) |
| `unknown_threshold` | none | per-side override for `unknown` sets (e.g. 0.10) |
| `calibration_margin` | 0.5 | headroom the calibrated rate aims for |

Load from JSON with `EvaluationConfig.from_json(path)` or from the
environment with `EvaluationConfig.from_env()`
(`WIGIN_TLLM_MIN_EVAL_SCORE=-3.5`, …). Unknown keys are rejected rather
than silently ignored.
