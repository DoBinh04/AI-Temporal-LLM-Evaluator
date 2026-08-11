# Scoring reference

Every formula the pipeline applies, with its rationale and edge cases.

To see these numbers for your own model, run `wigin-tllm benchmark`. It
reports each probe set's median, hit count and verdict, a diagnosis when a
year fails, and the headline scores. For what to *do* about them, see
[model-guide.md](model-guide.md).

**The pipeline at a glance:**

1. [**Corpus**](#the-corpus) — dated facts become a `known`/`unknown` probe
   pair per cutoff year, with `epsilon` placed by calibration.
2. [**Stage 1 — consistency**](#stage-1--chronological-consistency) — each
   year's model runs the gauntlet (limits, SVD gate, time budget), then both
   probe sets. Mean over all years = the leak score.
3. [**Qualification**](#qualification-and-normalisation) — leak score must
   clear `min_eval_score`, or the final score is 0 and nothing else runs.
4. [**Stage 2 — quality**](#stage-2--quality) — completion duels against
   reference models under a judge. Share of duels won = the win rate.
5. [**Final**](#final-score) — `0.7 × normalised leak + 0.3 × win rate`.

[Similarity](#similarity) is the copy detector behind the stage-1 gate and
two standalone checks.

---

## The corpus

The raw material is one flat list of dated facts:

```json
{"facts": [{"year": 2016, "prompt": "The 2016 Nobel Prize in Literature went to the songwriter", "phrase": "Bob Dylan"}, ...]}
```

`wigin-tllm corpus` turns it into **two probe sets per cutoff year**. Every
fact is used for every year — only the side changes:

```
fact.year <= Y  →  benchmarks/Y/known.json     (the model MUST recognise it)
fact.year >  Y  →  benchmarks/Y/unknown.json   (the model must NOT)
```

So the same fact migrates: "ChatGPT (2022)" is *unknown* for cutoffs
2013–2021 and *known* for 2022 onward. Facts dated after the last evaluated
year exist purely to keep the newest cutoff's `unknown` set non-empty — a
corpus should always extend one year past the years it scores.

**A probe must be unknowable before its year.** Not pre-announced earlier
(Olympic hosts, spacecraft targets, rocket names are public years in
advance), and not derivable from the prompt by language or general knowledge
alone ("the Paris cathedral" → Notre-Dame needs no cutoff knowledge). A
probe that fails this either flags clean models as leakers or lets leakers
pass.

### Calibration

`threshold` and `epsilon` travel with each probe set on disk, not with the
config, so they can differ per year and per side:

| Field | Default | Meaning |
|---|---|---|
| `epsilon` | −11.51 | log-probability above which a probe counts as *recognised* (≈ ln 1e-5) |
| `threshold` | 0.10 | tolerated fraction of recognised probes |

**`epsilon` must match the vocabulary size.** An uninformed guess scores
about `ln(1/V)` per token. At `V = 50000` that is ≈ −10.8, so −11.51 sits
just below guessing; at `V = 130` (the demo) guessing scores ≈ −4.9 and
−11.51 is unreachable. Getting this wrong makes every model look like a
leaker, or like an empty one.

That is why `--calibrate-with` exists: it **measures** a model that genuinely
respects the cutoffs and places `epsilon` so that model's post-cutoff hit
rate lands at `threshold × calibration_margin` — under the line with room to
spare, so a verdict is never decided by one probe crossing on floating-point
noise. The calibration report shows the margin per year; NaN probe scores
(sporadic CPU kernels produce them) are dropped rather than allowed to poison
the quantile.

The two sides may carry different thresholds (`known_threshold` /
`unknown_threshold`): the requirements are asymmetric — recognise *most* of
your own era, almost *none* of the future — so a production probe set
typically runs 0.70 / 0.10.

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

### The verdict

`probe()` reports the same quantity for both sets — `recognised`: *did more
than `threshold` of the probes score above `epsilon`?* — and `assess_year()`
reads it in opposite directions:

```python
passed = (not unknown_result.recognised) and known_result.recognised
```

Both checks are load-bearing:

- Without `unknown`, a model trained on everything passes.
- Without `known`, an empty model that assigns low probability to
  *everything* passes.

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

### The gauntlet

Before a single probe runs, each model faces the checks a real evaluation
applies, in this order:

| Check | Config | On failure |
|---|---|---|
| pinned reference (`owner/repo@<40-hex-sha>`) | `require_pinned_revision` | all of its years score worst |
| weight size (before download) | `max_model_bytes` | all of its years score worst |
| revision is a real commit (after download) | `require_pinned_revision` | all of its years score worst |
| parameter count (after load) | `max_parameters` | all of its years score worst |
| SVD gate, per year | references + `svd_threshold` | **that year** scores worst |
| time budget, per year | `max_eval_seconds` | the years not reached score worst |

The **SVD gate** runs when references are supplied (`--against` on
`consistency`, or the references given to `benchmark`): each year's model is
compared spectrally against every reference for that year and judged on the
*minimum* distance — being far from one published variant does not excuse
copying another. A gated year fails before its probes are ever run; a year
with no references passes by construction.

The **time budget** clock starts at model load. Once spent, the remaining
years are skipped and count worst-possible. `None` (the default) disables it.

`WORST_SCORE = 0.0` also covers missing years and load errors — every way a
year can fail lands on the same worst value.

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

## Qualification and normalisation

A model **clears stage 1** when its leak score is strictly below
`min_eval_score` (−3.0). One that does not is done: stage 2 never runs and
the final score is 0.0.

For those that clear, the leak score maps onto [0, 1]:

```
                     min_eval_score − score
normalised = clamp( ─────────────────────────── , 0, 1 )
                  min_eval_score − leak_best_score
```

Defaults map −3.0 → 0.0 and −6.0 → 1.0. Clamping means `WORST_SCORE` can
never earn credit, and — because the top is clamped too — **consistency
saturates**: at a leak score of −8 or −12 the normalised value is still 1.0,
and quality is the only lever left.

---

## Similarity

Compares the **singular value spectrum** of each 2D weight matrix rather
than the weights themselves:

```python
spectra[name] = svdvals(W)                       # per 2D weight matrix
k = max(1, int(len(sigma) * svd_top_ratio))      # top 25%
distance = mean over matrices of ||sigma_cand[:k] − sigma_ref[:k]||   # L2-normalised
too_close = distance < svd_threshold                                  # 0.01
```

**Why spectra, not weights.** A copy can be disguised as `W' = P·W·Q` with
orthogonal `P`, `Q`, compensated in an adjacent layer so behaviour is
unchanged. Cosine similarity of the raw weights collapses toward zero — the
disguise defeats it. Singular values are invariant under orthogonal
transforms, `σ(P·W·Q) = σ(W)`, so the spectrum still matches.
L2-normalising before comparison additionally defeats a global rescale.
`tests/test_svd_gate.py` pins both properties.

**Known limitation.** After L2 normalisation a single-element slice is
always `[1.0]`, so any matrix with `min(shape) ≤ 4` contributes distance 0
at the default `top_ratio`. Distances are averaged across all comparable
matrices, so this only matters for models built entirely from very small 2D
tensors.

The same primitive is used three ways:

| Use | Command | Effect |
|---|---|---|
| stage-1 gate | `consistency --against`, or `benchmark` with references | a copied year scores worst |
| report | `similarity --model X --against Y` | informational; `too_close` marks the benchmark not accepted |
| pairwise dedup | `similarity --pairwise a,b,c` | all pairs compared; of a duplicate pair the *first-listed* keeps its place |

---

## Stage 2 — quality

Needs a judge, a non-empty prompt set, and at least one reference model to
duel against; without all three only stage 1 can be scored.

### Prompts

Stage 2 does not ask questions — it hands the model **incomplete text to
continue**. That is what a language model does natively, so a base
checkpoint and an instruction-tuned one compare on the same footing.

Prompts are generated fresh per run by an LLM across eight categories:

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

- **Fresh per run** — a fixed set would be learnable, and a model tuned to
  it would score well without being better.
- **Timeless** — no dates, no recent events. A prompt about 2019 is
  unanswerable for a 2015-cutoff model, and stage 2 measures quality, not
  chronology — that is stage 1's job.

Generation samples at `temperature=1.0` (the goal is variety), one request
per category; pass a seed for reproducibility. If generation is configured
and yields nothing, the run aborts rather than quietly skipping stage 2.

### Year selection

The oldest year is always evaluated (comparability across runs), plus
`quality_year_samples − 1` drawn at random from the rest (so effort cannot
be concentrated on a predictable year). Set `quality_seed` to reproduce.

### Duels

Every pair meets once per evaluated year. For each prompt the two
completions are presented in random order and the verdict mapped back:

```python
swap = rng.random() < 0.5
verdict = {"a": "b", "b": "a", "tie": "tie"}[raw] if swap else raw
```

Judges — LLMs especially — tend to prefer whichever completion they see
first; randomising position spreads that bias evenly instead of letting it
decide duels. `tests/test_quality.py` verifies that a judge which *always*
says "first" gives nobody a systematic advantage.

A duel is won on the majority of prompts; equal counts are a draw and score
for neither side.

```
win_rate = duels_won / opponents_faced     per year, then averaged
```

### Reference opponents

References enter the tournament without being scored themselves. They make
quality measurable at all (a model alone has nothing to be better than) and
they anchor the scale (a rate stays comparable between runs instead of
depending on who else entered).

**Use at least two.** A drawn duel scores for neither side, so against a
single opponent a win rate of 0 covers both "lost every duel" and "drew
every duel" — only the logged `1W-1D-0L` record separates them. Two
opponents also give the rate real resolution: 0, 0.5 or 1 instead of 0 or 1.

A model that is missing or fails to load yields empty completions: it loses
its duels rather than aborting the round.

### Judge hardening

`OpenAIJudge` pins `temperature=0` and a fixed `seed`, and constrains the
response to a strict JSON schema (`enum: [a, b, tie]`) so there is no free
text to parse. It scores factual accuracy, natural continuation, coherence,
and knowledge demonstrated.

Completions are wrapped in `<completion>` tags with a system prompt telling
the judge to treat that content as data, and truncated (500 chars for the
prompt, 300 per completion). A submitted model can be trained to emit text
aimed at the judge, so this is a real boundary, not decoration.

---

## Final score

```
final = leak_weight · normalised_leak + quality_weight · win_rate
      = 0.7 · normalised_leak + 0.3 · win_rate
```

- **Not qualified** (leak score ≥ `min_eval_score`) → final = **0.0**, stage
  2 skipped. The standalone `quality` command still measures a win rate for
  authors who want the number while stage 1 is being fixed.
- Consistency not yet measured → no final score.
- Qualified but quality unmeasured (no judge, references, or prompts) → no
  final score; the report shows what consistency alone would contribute.

---

## Configuration reference

| Key | Default | Effect |
|---|---|---|
| `max_model_bytes` | 10 GiB | size limit, checked before download |
| `max_parameters` | 2×10⁹ | parameter limit, checked after load |
| `max_eval_seconds` | none | per-model stage-1 time budget (none = unlimited) |
| `min_eval_score` | −3.0 | the bar a model must clear |
| `leak_best_score` | −6.0 | leak score normalising to 1.0 |
| `leak_weight` / `quality_weight` | 0.7 / 0.3 | final-score blend |
| `quality_max_new_tokens` | 50 | completion length |
| `quality_year_samples` | 2 | cutoff years duelled |
| `quality_seed` | none | reproducible year draw and swaps |
| `svd_threshold` | 0.01 | below this spectral distance = a copy |
| `svd_top_ratio` | 0.25 | fraction of each spectrum compared |
| `require_pinned_revision` | true | reject unpinned HF references |
| `device` | auto | `cpu` / `cuda` / `mps` |
| `probe_threshold` | 0.25 | probe hit-rate threshold when building a corpus |
| `known_threshold` | none | per-side override for `known` sets (e.g. 0.70) |
| `unknown_threshold` | none | per-side override for `unknown` sets (e.g. 0.10) |
| `calibration_margin` | 0.5 | headroom the calibrated hit rate aims for |

Load from JSON with `EvaluationConfig.from_json(path)` or from the
environment with `EvaluationConfig.from_env()`
(`WIGIN_TLLM_MIN_EVAL_SCORE=-3.5`, …). Unknown keys are rejected rather than
silently ignored.

### Environment variables

Secrets and endpoints stay out of the config file:

| Variable | Used by | Default |
|---|---|---|
| `OPENAI_API_KEY` | `OpenAIJudge`, `OpenAIPromptGenerator` | — (required for `--judge openai` and prompt generation) |
| `JUDGE_MODEL` | `OpenAIJudge` | `gpt-4o-mini` |
| `PROMPT_MODEL` | `OpenAIPromptGenerator` | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | the OpenAI SDK itself | api.openai.com — point at any OpenAI-compatible server (vLLM, Ollama, …) |
| `HF_TOKEN` | huggingface_hub | — (only for private/gated repos) |

The CLI also loads a `.env` file from the working directory at startup
(`cp .env.example .env`); a variable already set in the real environment
always wins over the file, and `.env` is gitignored.

In library use both classes take the key directly:
`OpenAIJudge(api_key=..., model=...)`. The offline judges
(`ReferenceOverlapJudge`, `ScriptedJudge`) and every non-quality stage need
none of these.
