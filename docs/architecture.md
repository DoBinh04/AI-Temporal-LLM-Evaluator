# Architecture

## Module map

```
wigin_tllm/
├── types.py            Data model — no torch, no numpy
├── config.py           EvaluationConfig: every scoring knob, explicit
├── pipeline.py         Orchestrator: sequencing, limits, failure policy
├── storage.py          SQLite: resume cache, outbox, weight-hash registry
├── cli.py              check / run / validate / show
├── check.py            Single-model pre-flight check
├── report.py           Results table rendering
├── logging_setup.py    Log configuration
│
├── datasource/         Where inputs come from and results go
│   ├── base.py         DataSource (abstract)
│   ├── local.py        LocalDataSource — a directory
│   ├── http.py         HttpDataSource — a remote service
│   └── memory.py       InMemoryDataSource — fixtures
│
├── scoring/            All the mathematics
│   ├── leak.py         Stage 1 log-probability scoring
│   ├── svd_gate.py     Spectral anti-copy (baseline gate + pairwise dedup)
│   ├── judge.py        Judge interface and implementations
│   ├── quality.py      Stage 2 round-robin tournament
│   ├── aggregate.py    Qualification, blending, ranking
│   └── baselines.py    Optional published reference models
│
└── models/             Weights in, uniform interface out
    ├── store.py        Resolve, download (watchdog), hash, measure
    ├── loader.py       Architecture dispatch -> uniform wrapper
    ├── chronogpt.py    Vendored ChronoGPT architecture
    └── architectures/  Registered custom architectures
```

## Data flow

```
                    ┌──────────────┐
                    │  DataSource  │  years, probe sets, submissions,
                    └──────┬───────┘  questions  (local dir / HTTP / memory)
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  preload probe sets  (fail fast)         │
        └──────────────────┬───────────────────────┘
                           ▼
    ╔═══════════════ STAGE 1 ═══════════════════════════════╗
    ║  for each submitter, oldest submission first:         ║
    ║    size check ─► resolve ─► pinned-revision check     ║
    ║      ─► weight-hash claim ─► load ─► parameter check  ║
    ║      ─► SVD baseline gate ─► per-year leak scoring    ║
    ║  results stream to SQLite, then to the DataSource     ║
    ╚═══════════════════════┬═══════════════════════════════╝
                            ▼
              ┌─────────────────────────────┐
              │  SVD pairwise dedup         │  spectra reloaded from SQLite
              └─────────────┬───────────────┘
                            ▼
              ┌─────────────────────────────┐
              │  qualify: score < threshold │
              │  keep best N                │
              └─────────────┬───────────────┘
                            ▼
    ╔═══════════════ STAGE 2 ═══════════════════════════════╗
    ║  AnswerProvider generates answers per year            ║
    ║  round-robin duels, Judge decides, A/B swapped        ║
    ╚═══════════════════════┬═══════════════════════════════╝
                            ▼
        ┌──────────────────────────────────────────┐
        │  final = 0.7·leak + 0.3·quality          │
        │  order by score ─► assign rank ─► persist│
        └──────────────────────────────────────────┘
```

## Design decisions

### The orchestrator holds no mathematics

`pipeline.py` sequences work, enforces limits, and decides what happens on
failure. Every formula lives in `scoring/`. This is why `aggregate.py` can be
tested without torch and why the stubbed integration tests in
`tests/test_pipeline.py` can cover orchestration without loading a model.

### Submitters are processed oldest-first

Anti-copy rules are first-come-first-served. Iterating in submission order
means the earliest submitter of a given set of weights always claims them
before any copier reaches the registry, so the rule needs no tie-breaking
logic of its own.

### One reference, one download

A submitter often uses the same model for several years. References are
grouped so each is resolved and loaded once, then scored against every year it
covers.

### Failures are attributed

| Failure | Attribution | Consequence |
|---|---|---|
| `ModelLoadError`, bad config, corrupt weights | submitter | that submitter scores worst, round continues |
| Gate failures, limits exceeded | submitter | same |
| `RuntimeError` (exhausted download retries), `MemoryError` | us | round aborts |

Recording a worst-possible score because *our* GPU ran out of memory would
penalise someone for our outage, so infrastructure failures stop the round
instead. The distinction is enforced by exception type, and
`ModelLoadError` is caught before the `RuntimeError` branch precisely because
it subclasses it.

### Results are written locally before they are published

Every year's result goes to SQLite first with `synced = 0`, then to the data
source. A sink that is temporarily unreachable delays reporting rather than
losing it; the outbox is flushed at the start of every subsequent run,
including for earlier rounds.

### Rounds are idempotent

A completed round returns its stored results rather than re-scoring. An
interrupted round resumes: per-year rows survive, so only unfinished years are
re-scored. `force=True` clears the round and starts over.

### Downloads run in a subprocess

`snapshot_download` can wedge on a stalled connection, and a hung evaluator is
worse than a failed one. The download runs in a spawned process watched by a
directory-size watchdog, so it can always be killed. Failure modes travel back
as exit codes because exceptions do not cross a process boundary. Retries
escalate: wipe partial state, then wipe the chunk cache, then disable xet
entirely.

### No submitted code is ever executed

- `config.json` is parsed as JSON, never evaluated.
- Weights load from safetensors (or `torch.load(..., weights_only=True)`).
- `trust_remote_code=False`, always.

A custom architecture therefore cannot ship with a submission. It must be
added under `models/architectures/` and reviewed. Each package registers
itself at import:

```python
AutoConfig.register(MyConfig.model_type, MyConfig)
AutoModelForCausalLM.register(MyConfig, MyForCausalLM)
```

`models/architectures/__init__.py` imports every sub-package automatically, so
`import wigin_tllm.models.loader` is enough to make them all available to
`AutoModelForCausalLM` without `trust_remote_code`.

### Probe-set contents are never logged

Only aggregate statistics — medians, counts, ratios — reach the logs. A single
`logger.debug(items)` would leak the evaluation set into anywhere logs are
shipped. Keep it that way when adding logging.

## The uniform model interface

`load_model()` returns a wrapper presenting the same surface whatever the
underlying architecture:

```python
model.forward(input_ids) -> logits
model.encode(text) -> list[int]
model.decode(ids) -> str
model.generate(prompt, max_new_tokens) -> str
model.inner_state_dict() -> dict[str, Tensor]   # for the SVD gate
model.pad_token_id, model.stop_token_ids
```

Two implementations exist: `_ChronoGPTWrapper` (tiktoken GPT-2, custom
sampling loop) and `_HFWrapper` (any HuggingFace causal LM). Dispatch is on
`model_type` in `config.json`, with a fallback for ChronoGPT configs that
carry `model_dim` instead.

This is what keeps `leak.py` and `quality.py` free of tokenizer branching.

## Concurrency and memory

The pipeline is single-threaded by design: it holds one model in memory at a
time and frees it before loading the next. The SVD baseline gate is the
exception — baselines stay resident for all of stage 1, which dominates
memory use when the gate is enabled with large reference models.

Spectra are persisted to SQLite rather than held in memory because pairwise
dedup needs every submitter's spectrum at once, and singular values are orders
of magnitude smaller than the weights they summarise.
