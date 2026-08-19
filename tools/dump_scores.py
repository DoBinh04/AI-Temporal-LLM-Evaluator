"""Dump per-probe raw scores of the chrono-gpt-v1 checkpoints on the dev corpus.

Analysis-only companion to data/dev/CALIBRATION.md: scores every probe of
data/dev/corpus/benchmarks/2022/ with the three ChronoGPT checkpoints and
writes data/dev/raw_scores.json, so every downstream analysis can be redone
offline without re-running a model. Uses the pipeline's own scoring
(wigin_tllm.scoring.leak.score_items) — nothing is reimplemented.

Revisions are the safetensors-branch commits verified bit-identical to the
pinned main-branch revisions (see "Revision substitution" in CALIBRATION.md).

Run from the repo root:

    python tools/dump_scores.py
"""

import json
import math
import os
import sys
import tempfile

import tiktoken

from wigin_tllm.corpus import KNOWN, UNKNOWN, Corpus
from wigin_tllm.models.loader import load_model
from wigin_tllm.models.store import download_model, free_device_memory, get_device
from wigin_tllm.scoring.leak import score_items
from wigin_tllm.types import ModelRef

MODEL_REFS = [
    "manelalab/chrono-gpt-v1-20221231@4d37df723313ff0c156795002fc0abc30de6abf6",
    "manelalab/chrono-gpt-v1-20231231@771747bd61cd50b8d99fe381a41eb25c86b80f3e",
    "manelalab/chrono-gpt-v1-20241231@26e0653a22c5d0b47845c64c2a45d7acde61222d",
]
CORPUS_ROOT = "data/dev/corpus"
YEAR = 2022
OUT_PATH = "data/dev/raw_scores.json"

# The epsilon sweep reports where the corpus would separate for the clean
# 2022 reference; these mirror the config the corpus is evaluated with.
KNOWN_BAR = 0.70
UNKNOWN_BAR = 0.10


def rate_above(scores, epsilon):
    return sum(1 for s in scores if s > epsilon) / len(scores)


def sweep(known_scores, unknown_scores):
    """Print rates over epsilon in [-20, -1] step 0.25; flag separating rows."""
    print(f"\nEpsilon sweep for the clean 2022 model "
          f"(separates = known > {KNOWN_BAR:.0%} and unknown <= {UNKNOWN_BAR:.0%}):")
    print(f"  {'epsilon':>8} {'known_rate':>11} {'unknown_rate':>13}")
    separating = []
    best_gap, best_eps = None, None
    steps = int(round((20.0 - 1.0) / 0.25)) + 1
    for i in range(steps):
        eps = -20.0 + i * 0.25
        kr = rate_above(known_scores, eps)
        ur = rate_above(unknown_scores, eps)
        ok = kr > KNOWN_BAR and ur <= UNKNOWN_BAR
        if ok:
            separating.append(eps)
        # How far this epsilon is from satisfying both conditions at once.
        gap = max(0.0, KNOWN_BAR - kr) + max(0.0, ur - UNKNOWN_BAR)
        if best_gap is None or gap < best_gap:
            best_gap, best_eps = gap, eps
        print(f"  {eps:>8.2f} {kr:>11.1%} {ur:>13.1%}  {'<-- separates' if ok else ''}")
    if separating:
        print(f"\nSeparating epsilons: {separating}")
    else:
        kr = rate_above(known_scores, best_eps)
        ur = rate_above(unknown_scores, best_eps)
        print(f"\nNo epsilon in [-20, -1] separates. Closest is {best_eps:.2f} "
              f"(known {kr:.1%}, unknown {ur:.1%}): combined shortfall "
              f"{best_gap:.1%} — needs known +{max(0.0, KNOWN_BAR - kr):.1%} "
              f"and unknown -{max(0.0, ur - UNKNOWN_BAR):.1%} at the same time.")


def main():
    device = get_device("cuda")
    enc = tiktoken.get_encoding("gpt2")
    corpus = Corpus(CORPUS_ROOT)
    benchmarks = {kind: corpus.benchmark(YEAR, kind) for kind in (KNOWN, UNKNOWN)}

    out = {"models": {}}
    clean_scores = None
    for raw_ref in MODEL_REFS:
        ref = ModelRef.parse(raw_ref)
        print(f"Scoring {raw_ref} ...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = download_model(ref.location, tmp, revision=ref.revision)
            model, _ = load_model(path, device)
            sides = {}
            for kind, benchmark in benchmarks.items():
                scores = score_items(model, device, benchmark.items)
                sides[kind] = [
                    {
                        "prompt": item.prompt,
                        "phrase": item.phrase,
                        "score": score,
                        "tokens": len(enc.encode(" " + item.phrase)),
                    }
                    for item, score in zip(benchmark.items, scores)
                ]
            out["models"][raw_ref] = sides
            if raw_ref == MODEL_REFS[0]:
                clean_scores = {k: [p["score"] for p in v] for k, v in sides.items()}
            del model
            free_device_memory()

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    nan = sum(
        1 for sides in out["models"].values()
        for probes in sides.values()
        for p in probes if math.isnan(p["score"])
    )
    print(f"Wrote {OUT_PATH} ({len(out['models'])} models x "
          f"{sum(len(v) for v in next(iter(out['models'].values())).values())} probes, "
          f"{nan} NaN scores)")

    sweep(clean_scores[KNOWN], clean_scores[UNKNOWN])


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
