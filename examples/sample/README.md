# Sample corpus

A ready-made corpus over the production year range (cutoffs 2013–2024), built
from 52 real dated facts — four per year, 2013 through 2025. The 2025 facts
exist so that even a 2024 model has a future it must not know.

```
facts.json                  the dated facts the corpus is built from
config.json                 production-style settings (known 0.70 / unknown 0.10)
corpus/                     the built probe sets, one known/unknown pair per year
corpus/completion_prompts.json   16 static stage-2 prompts, two per category
models.example.json         manifest template — copy, point at your checkpoints
```

## Use it

```bash
cp examples/sample/models.example.json models.json   # then edit

# stage 1
wigin-tllm consistency --submission models.json \
    --data examples/sample/corpus --config examples/sample/config.json \
    --against chronogpt

# stage 2 (the shipped static prompts are used automatically)
# --judge openai needs a key: put OPENAI_API_KEY in ./.env (cp .env.example .env)
wigin-tllm quality --submission models.json \
    --data examples/sample/corpus --config examples/sample/config.json \
    --against chronogpt --judge openai

# everything, and the final score
wigin-tllm benchmark --submission models.json \
    --data examples/sample/corpus --config examples/sample/config.json \
    --against chronogpt --judge openai
```

A single checkpoint on selected years works too: `--model local:./my-model
--years 2015` instead of `--submission`.

## Calibrate before trusting the numbers

The shipped `epsilon` is the default **−11.51** (≈ ln 1e-5), which suits
models with a ~50k-token vocabulary — a GPT-2-tokenizer model scoring a short
phrase below that is genuinely not recognising it. It was **not** placed by
measurement. Before drawing conclusions, calibrate against a model you trust
to respect the cutoffs:

```bash
wigin-tllm corpus --facts examples/sample/facts.json --out ./my-corpus \
    --config examples/sample/config.json \
    --calibrate-with manelalab/chrono-gpt-v1-20131231@8e3e454b59a27d96ed3773f5c58a10e84e4f3f12
```

The calibration report says whether the probe sets separate a clean model
from a leaking one, and by how much. See
[docs/scoring.md](../../docs/scoring.md) for what the thresholds mean.

## Growing the corpus

Four facts per year is enough to smoke-test a pipeline, not to grade a model:
with the 0.10 `unknown` threshold, one recognised probe out of a handful
flips a year. Add facts to `facts.json` (any amount, any years) and rebuild —
a real evaluation wants dozens per year, and the later years' `unknown` sets
only get deeper as newer facts are added. The stage-2 prompts are static so
runs are reproducible offline; regenerate a fresh set with
`wigin-tllm prompts` when you want them unlearnable.
