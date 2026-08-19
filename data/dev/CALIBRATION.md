# Calibration report — dev corpus, cutoff 2022

**Round 2 (see the bottom of this file) rebuilt the facts and the corpus now
separates**: measured epsilon −8.8792, the clean 2022 reference PASSes and the
2023/2024 checkpoints FAIL as leakers. Rounds 1 documents the failure that
motivated the rebuild — its per-probe numbers refer to the *old* facts.json
(preserved in git history) and to a superseded `raw_scores.json`.

Date: 2026-08-19. Hardware: 1× RTX 5090 (32 GB), `--device cuda`.
Config: `examples/sample/config.json` (`known_threshold` 0.70, `unknown_threshold` 0.10,
`probe_threshold` 0.25, `calibration_margin` 0.5).

## Revision substitution (weights verified identical)

The pinned revisions given for this experiment are the `main`-branch tips of the
three `manelalab/chrono-gpt-v1-*` repos, which carry only `pytorch_model.bin`.
`wigin_tllm/models/chronogpt.py` loads **only** `model.safetensors`, so those exact
revisions cannot be loaded (`FileNotFoundError: model.safetensors not found`,
exit 1 — this was the first calibrate attempt). Each repo has a `safetensors`
branch with the converted weights. Before substituting, **all 422 tensors of every
pair were compared bit-for-bit** (`torch.equal`, plus config.json and key-set
equality, after stripping the `_orig_mod.` prefix from the `.bin` keys):

| model | pinned (main, .bin) | used (safetensors branch) | tensors equal |
|---|---|---|---|
| chrono-gpt-v1-20221231 | `993711fdf078740fe1c837a3687528e2173443d2` | `4d37df723313ff0c156795002fc0abc30de6abf6` | 422/422 |
| chrono-gpt-v1-20231231 | `8ac22e54d37df8bb8037622680414118239fbe53` | `771747bd61cd50b8d99fe381a41eb25c86b80f3e` | 422/422 |
| chrono-gpt-v1-20241231 | `1d9f1b8ff50bb45a6fe1402280e617af4c2d805c` | `26e0653a22c5d0b47845c64c2a45d7acde61222d` | 422/422 |

Every number below therefore reflects exactly the pinned weights.

## 1. Token audit (GPT-2 BPE, `" " + phrase`, per `scoring/leak.py`)

Phrase token counts over the 100 probes of `corpus/benchmarks/2022/`:

| side | distribution {tokens: count} | min | max | median | mean |
|---|---|---|---|---|---|
| known (50) | {1: 7, 2: 19, 3: 13, 4: 5, 5: 4, 6: 2} | 1 | 6 | 2 | 2.72 |
| unknown (50) | {1: 6, 2: 22, 3: 14, 4: 2, 5: 4, 6: 1, 7: 1} | 1 | 7 | 2 | 2.66 |

Epsilon bounds the **summed** log-prob, so the mean per-token probability a
phrase of T tokens must reach to clear the default −11.51 is `exp(−11.51/T)`:

| T | exp(−11.51/T) | known n | unknown n |
|---|---|---|---|
| 1 | 0.000010 | 7 | 6 |
| 2 | 0.003167 | 19 | 22 |
| 3 | 0.021565 | 13 | 14 |
| 4 | 0.056275 | 5 | 2 |
| 5 | 0.100059 | 4 | 4 |
| 6 | 0.146852 | 2 | 1 |
| 7 | 0.193150 | 0 | 1 |

**Conclusion.** The per-token requirement spans **four orders of magnitude**
(1e-5 for a 1-token phrase like "Nepal"/"Gemini" vs ≈0.19 per token for the
7-token "Operation Al-Aqsa Flood"). With a median phrase length of 2 on both
sides, a scalar epsilon of −11.51 classifies probes **primarily by phrase
length, not by knowledge**: short phrases are near-guaranteed hits, long
phrases are near-unreachable. A single scalar epsilon on the summed log-prob
cannot be knowledge-selective across this length spread — calibration is
required (and even a calibrated scalar keeps the same length bias; see §4).

## 2. Calibration

Command (revision substituted as documented above; the literal pinned-main
revision fails to load with exit 1):

```
wigin-tllm corpus --facts data/dev/facts.json --out /tmp/corpus-cal-2022 --years 2022 \
  --config examples/sample/config.json --device cuda \
  --calibrate-with "manelalab/chrono-gpt-v1-20221231@4d37df723313ff0c156795002fc0abc30de6abf6"
```

Verbatim output:

```
09:55:54 INFO    | 2022: epsilon=-3.8248 known=22.0% unknown=4.0%
09:55:54 INFO    | Wrote 1 years of probes to /tmp/corpus-cal-2022

=== Corpus calibration ===

--------------------------------------------------------------------------
  year      epsilon    known   unknown  threshold  verdict
  2022      -3.8248   22.0%     4.0%     25.0%  DOES NOT SEPARATE
--------------------------------------------------------------------------
This corpus cannot tell a clean model from a leaking one. The reference model must
recognise its own era and not the future; if it does neither, the probes are too hard.
```

- measured epsilon: **−3.824838** (95th percentile of the reference's unknown
  scores; target unknown rate = 0.10 × 0.5 = 5%, lands on 4% with 50 probes)
- known_rate: **22.0%** — far below the 70% `known_threshold`
- unknown_rate: **4.0%**
- margin (min distance of either rate to its bar): **6.0%** (unknown side;
  the known side is 48 points short)
- separates: **False** (`DOES NOT SEPARATE`)
- exit code: **1**

The corpus **was** still written, with epsilon −3.8248 on both sides; it is
checked in at `data/dev/corpus-calibrated/`. The verdict means: at any epsilon
low enough to keep the future invisible, this reference recognises only ~22%
of its own era — the probes are too hard for a 1.5B ChronoGPT.

## 3. Old epsilon on the uncalibrated corpus (prediction check)

Command: `wigin-tllm consistency --model <2022 ref> --years 2022 --data data/dev/corpus …`

```
  2022  [FAIL]  score +0.0000
      known    median   -5.7908   43/50 above epsilon -11.51   (must recognise)
      unknown  median  -12.1079   21/50 above epsilon -11.51   (must not recognise)
      recognises post-cutoff facts as readily as pre-cutoff ones — the training data reaches beyond the cutoff

  leak score         0.0000   (threshold -3.0000) — below the bar
```

Exit code 1. Unknown **21/50 (42%)** above epsilon vs a 10% threshold; known
43/50 (86%). Verdict: leaker ("the training data reaches beyond the cutoff").

**The prediction was correct**: with the default −11.51, the clean 2022
reference model is labelled a leaker, because 21 post-cutoff probes — almost
all of them short phrases — clear the length-dominated threshold.

## 4. Control experiment (calibrated corpus, no `--against`)

Stage 1 on `corpus-calibrated` (epsilon −3.8248) for the three checkpoints:

| model (cutoff) | unknown above ε | known above ε | median unknown | median known | leak score | verdict | expected |
|---|---|---|---|---|---|---|---|
| 20221231 (clean) | 2/50 (4%) | 11/50 (22%) | −12.1079 | −5.7908 | 0.0 | **FAIL** | PASS |
| 20231231 (saw 2023) | 3/50 (6%) | 10/50 (20%) | −11.1694 | −5.8512 | 0.0 | **FAIL** | FAIL |
| 20241231 (saw 2023-24) | 5/50 (10%) | 10/50 (20%) | −8.2576 | −5.7439 | 0.0 | **FAIL** | FAIL |

All three exit 1 with the diagnosis "recognises neither past nor future — the
model has not learned its own era, or epsilon (−3.82) is stricter than this
vocabulary can reach". All three fail **on the known side** (20–22% vs the 70%
bar), i.e. as "empty", not as leakers.

**Is the leak signal monotone in the cutoff?** **Yes**, in the raw
measurements: unknown hits 2 → 3 → 5 (4% → 6% → 10%) and median unknown
−12.11 → −11.17 → −8.26 rise strictly with the cutoff, while the known side
stays flat (~−5.8, 10–11 hits) — exactly the signature of later checkpoints
having seen more of the "future".

**Does this dataset distinguish a clean model from a leaking one?** **No**,
not with the binary verdict as configured:

1. The clean 2022 model does **not** PASS — it fails the known check (22% ≪
   70%), consistent with the calibration verdict. The probes are too hard for
   these 1.5B models to "recognise" at any epsilon tight enough for the
   unknown side.
2. Even the 2024 leaker is not flagged *as a leaker*: its unknown rate lands
   at exactly 10%, and `recognised` requires strictly **more** than the
   threshold — so every model fails for the same (wrong) reason.

The monotone medians show the underlying measurement works; the pass/fail
machinery does not separate on this corpus. Per instructions no data was
changed to force the expected outcome. Plausible directions (not applied):
easier/shorter known probes (the token audit shows longer phrases are
structurally penalised), a larger reference model, or a per-length /
per-token-normalised epsilon — the last one is a scoring change and out of
scope here.

## Reproduction notes

- The CLI re-downloads ~7.4 GB per run; each stage-1 run takes ~3.5 min on an
  RTX 5090.
- `wigin-tllm` was installed editable from this repo into `/venv/main`
  (`uv pip install -e . tiktoken`).
- Raw scores of the 2022 reference are identical across §3 and §4 (medians
  match to 4 decimals); only epsilon differs between the two corpora, as
  expected.

---

# Round 2 — facts rebuilt, corpus separates

The round-1 diagnosis (length-dominated scalar epsilon + guessable `unknown`
probes + too-hard `known` probes) was fixed **in the data only** — no scoring
code changed:

1. **Every phrase is now 1–2 GPT-2 BPE tokens** (was 1–7), with the length
   distribution matched across sides, so the scalar epsilon compares like
   with like.
2. **`unknown` probes were pre-screened against the clean 2022 reference**:
   any probe it scored above ≈ −3.5 per token was guessable from the prompt by
   priors alone (e.g. "Mandalay" from *Myanmar earthquake*, "Taylor Swift"
   from *musician of the year*, "Machado" from *Maria Corina*) and was
   replaced with a prior-defying one (Raygun, Pop Mart, RedNote, Willow, …).
3. **`known` probes the reference could not recognise were replaced** — the
   obscure (Lubitz, Harambe, Roma) and the edge-of-cutoff (FTX, Midjourney,
   both from the last weeks of 2022, before the training data caught up).
4. All `tests/test_dev_corpus.py` invariants hold (50/50 split, ≥6 facts per
   year, no duplicate phrases, no overlap with `examples/sample/`, matched
   word-length distributions): 11/11 pass.

## Calibration (same command as §2, new facts)

```
11:42:26 INFO    | 2022: epsilon=-8.8792 known=94.0% unknown=4.0%

  year      epsilon    known   unknown  threshold  verdict
  2022      -8.8792   94.0%     4.0%     25.0%  separates

Calibrated. Tightest margin 6.0% (year 2022) — comfortably clear of the threshold.
```

Exit code **0**. This is the corpus checked in at `data/dev/corpus-calibrated/`.

## Control experiment (calibrated corpus, no `--against`)

| model (cutoff) | unknown above ε | known above ε | median unknown | median known | leak score | verdict | expected |
|---|---|---|---|---|---|---|---|
| 20221231 (clean) | 2/50 (4%) | 47/50 (94%) | −11.8972 | −3.0249 | **−8.8723** (normalised 1.0) | **PASS** ✅ | PASS |
| 20231231 (saw 2023) | 10/50 (20%) | 47/50 (94%) | −11.4498 | −3.1645 | 0.0 | **FAIL — leaker** ✅ | FAIL |
| 20241231 (saw 2023-24) | 19/50 (38%) | 45/50 (90%) | −9.7817 | −3.2029 | 0.0 | **FAIL — leaker** ✅ | FAIL |

Both leakers fail with the *correct* diagnosis ("recognises post-cutoff facts
as readily as pre-cutoff ones — the training data reaches beyond the cutoff"),
not the round-1 "recognises neither" failure. The leak signal is strictly
monotone in the cutoff — unknown hit rate 4% → 20% → 38%, median unknown
−11.90 → −11.45 → −9.78 — while the known side stays flat (~94%, median ≈ −3.1).

**Acceptance: this dataset now distinguishes a clean model from a leaking
one.** The clean model passes with a 6-point margin on the unknown side and a
24-point margin on the known side; the mildest leaker (2023) exceeds the 10%
unknown threshold two-fold.

`data/dev/raw_scores.json` was regenerated on the new corpus for offline
analysis. A caution for future edits: the three chrono checkpoints are the
acceptance instrument — calibrate only on the clean 2022 model, and never
tune individual facts against the 2023/2024 scores, or the corpus overfits
to these three models.
