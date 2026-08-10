# Data formats

`LocalDataSource` reads and writes a single directory. Only `benchmarks/` and
`submissions/` are required.

```
data/
├── round.json               optional — {"current_round": 1}
├── years.json               optional — [2013, 2014, ...]
├── benchmarks/
│   └── 2013/
│       ├── known.json       facts from before the cutoff
│       └── unknown.json     facts from after the cutoff
├── submissions/
│   └── 1.json               one file per round
├── quality_questions.json   optional — stage 2 questions
├── results/                 written: full round outcome
│   └── 1.json
└── eval_details/            written: one line per (submitter, year)
    └── 1.jsonl
```

## `round.json`

```json
{ "current_round": 1 }
```

Defaults to round 1 if absent. A "round" is just a batch identifier — it
carries no scheduling semantics.

## `years.json`

```json
[2013, 2014, 2015]
```

If absent, the years are inferred from the subdirectories of `benchmarks/`.

## `benchmarks/<year>/<known|unknown>.json`

```json
{
  "items": [
    { "prompt": "in 2013 the summit was held in", "phrase": "geneva" },
    { "prompt": "the treaty of that year was signed by", "phrase": "norvale" }
  ],
  "threshold": 0.10,
  "epsilon": -11.51
}
```

| Field | Required | Meaning |
|---|---|---|
| `items[].prompt` | yes | context shown to the model |
| `items[].phrase` | yes | continuation whose log-probability is measured |
| `threshold` | no (0.10) | tolerated fraction of probes scoring above `epsilon` |
| `epsilon` | no (−11.51) | log-probability above which a probe counts as recognised |

**Calibrate `epsilon` to your vocabulary size** — see
[scoring.md](scoring.md#probe-calibration). A phrase is scored with a leading
space (`" " + phrase`), which matters for BPE tokenizers.

An empty `items` list is legal and yields a sentinel median of −20.0 with no
failure, which is how a year with no future facts is handled.

## `submissions/<round>.json`

List form:

```json
[
  {
    "submitter_id": "alice",
    "submitted_at": "2026-07-01T08:00:00",
    "models": {
      "2013": "alice/chrono-2013@a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
      "2014": "alice/chrono-2014@b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1"
    }
  }
]
```

Mapping form is also accepted:

```json
{ "alice": { "submitted_at": "...", "models": { "2013": "..." } } }
```

| Field | Required | Meaning |
|---|---|---|
| `submitter_id` | yes | any string; identifies the submitter |
| `models` | yes | `{ "<year>": "<model reference>" }` |
| `submitted_at` | strongly recommended | ISO-8601; decides anti-copy priority |

`submitted_at` is the tie-breaker for every anti-copy rule. A submission
without one sorts last, so omitting it forfeits priority.

### Model references

| Scheme | Form | Notes |
|---|---|---|
| HuggingFace | `owner/repo@<40-hex-sha>` or `hf:owner/repo@<sha>` | pinned by default |
| Local | `local:/absolute/path/to/model_dir` | for offline runs and testing |

Pinning to a commit SHA is what makes a submission immutable: a branch name —
even one that looks like a SHA — can be repointed at different weights later.
`verify_pinned_revision()` resolves the revision and rejects it unless it
matches itself. Set `require_pinned_revision = false` to allow branches and
local paths.

A model directory must contain:

```
config.json          standard HuggingFace config
model.safetensors    weights (pytorch_model.bin is read but discouraged)
tokenizer.json       plus tokenizer_config.json etc.
```

## `quality_questions.json`

```json
{
  "questions": [
    { "prompt": "what caused the 2008 financial crisis", "reference": "subprime mortgage lending" }
  ]
}
```

A bare list is also accepted. `reference` is optional and ignored by LLM
judges; `ReferenceOverlapJudge` uses it as the expected answer. With no
questions, stage 2 is skipped.

## `results/<round>.json` (written)

`submitters` is the single source of truth and is ordered best-first, so the
leading entry is the strongest model of the round.

```json
{
  "round_id": 1,
  "submitters": [
    {
      "submitter_id": "alice",
      "leak_score": -6.75,
      "year_scores": { "2013": -6.9, "2014": -6.7, "2015": -6.6 },
      "qualified": true,
      "normalized_leak": 1.0,
      "quality_win_rate": 1.0,
      "final_score": 1.0,
      "rank": 1,
      "disqualified_reason": ""
    }
  ]
}
```

### `disqualified_reason` values

| Value | Meaning |
|---|---|
| `model_too_large` | over `max_model_bytes` |
| `too_many_parameters` | over `max_parameters` |
| `revision_not_pinned` | reference is not a real commit SHA |
| `duplicate_weights` | identical bytes already claimed |
| `duplicate_of_earlier_submission` | spectrum matches an earlier submitter |
| `svd_gate_failed` | too close to a published baseline |
| `eval_timeout` | exceeded `max_eval_seconds` |
| `model_load_failed` | unusable artefact |
| `error:<ExceptionType>` | anything else attributable to the submission |
| `failed_consistency_check` | no year passed both probe sets |

## `eval_details/<round>.jsonl` (written)

One JSON object per line, appended as each year finishes:

```json
{"submitter_id": "alice", "year": 2013, "model_ref": "local:/models/alice/2013",
 "passed": true, "score": -6.9, "score_unknown": -9.1, "score_known": -2.2, "reason": ""}
```

## HTTP data source

`HttpDataSource` expects the same shapes over these endpoints:

| Method | Path | Returns / accepts |
|---|---|---|
| GET | `/rounds/current` | `{"current_round": 1}` |
| GET | `/years?round_id=1` | `{"years": [...]}` |
| GET | `/benchmark/<year>?kind=unknown` | benchmark object |
| GET | `/submissions/<round>` | list or `{"submissions": …}` |
| GET | `/quality/questions` | `{"questions": [...]}` |
| POST | `/eval/detail` | `{"round": 1, ...year evaluation}` |
| POST | `/eval/results` | `{"round": 1, "results": {...}}` |

Authentication is the caller's choice: pass `token=` for a bearer header,
`client_cert=(cert, key)` for mutual TLS, or nothing on a trusted network.

## Cache database

`<data_dir>/evaluations.db` (SQLite) holds the resume cache, the unsynced
outbox, stored round results, saved spectra, and the weight-hash registry. It
is derived state — deleting it costs re-computation, not correctness — except
for the weight-hash registry, which is the record of who claimed which weights
first.
