"""Stage 1 — chronological consistency scoring.

The core measurement is the summed log-probability a model assigns to a known
continuation given a prompt (teacher forcing). Run over two probe sets:

  * ``unknown`` — facts from *after* the cutoff. High probability here means
    the model has seen the future: a leak.
  * ``known`` — facts from *before* the cutoff. High probability here is
    required; without this check an empty model would pass by knowing nothing.
"""

from __future__ import annotations

import logging

import torch

from ..config import WORST_SCORE
from ..types import Benchmark, BenchmarkItem, ProbeResult, YearAssessment

logger = logging.getLogger(__name__)

# Returned as the median when a probe set is empty. Chosen to be
# unambiguously "very unlikely" without being infinite.
EMPTY_BENCHMARK_MEDIAN = -20.0


def score_items(model, device, items: list[BenchmarkItem]) -> list[float]:
    """Sum of log-probabilities of each item's phrase, given its prompt.

    Items are scored column-wise: one forward pass produces token *t* of every
    item at once, so the number of forward passes is the longest phrase length
    rather than the number of items.

    Right-padding needs no attention mask here: attention is causal, so tokens
    after the position being read cannot influence it.
    """
    if not items:
        return []

    pad_token = model.pad_token_id
    bos = getattr(model, "bos_token_id", None)

    all_prompt_tokens = []
    all_phrase_tokens = []
    for item in items:
        prefix = [bos] if bos is not None else []
        prompt_tokens = prefix + model.encode(item.prompt)
        phrase_tokens = model.encode(" " + item.phrase)
        all_prompt_tokens.append(prompt_tokens if prompt_tokens else [pad_token])
        all_phrase_tokens.append(phrase_tokens if phrase_tokens else [])

    max_phrase_len = max((len(p) for p in all_phrase_tokens), default=0)
    current = [list(p) for p in all_prompt_tokens]
    totals = [0.0] * len(items)

    with torch.no_grad():
        for token_pos in range(max_phrase_len):
            active = [i for i in range(len(items)) if token_pos < len(all_phrase_tokens[i])]
            if not active:
                break

            seqs = [current[i] for i in active]
            max_len = max(len(s) for s in seqs)
            padded = [s + [pad_token] * (max_len - len(s)) for s in seqs]
            lengths = [len(s) for s in seqs]

            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            logits = model(input_ids)

            for batch_idx, item_idx in enumerate(active):
                pos = lengths[batch_idx] - 1
                probs = torch.nn.functional.softmax(logits[batch_idx, pos, :], dim=-1)
                expected_token = all_phrase_tokens[item_idx][token_pos]
                totals[item_idx] += torch.log(probs[expected_token] + 1e-10).item()
                current[item_idx].append(expected_token)

    return list(totals)


def score_one(model, device, prompt: str, phrase: str) -> float:
    """Score a single (prompt, continuation) pair. Useful when debugging one probe."""
    return score_items(model, device, [BenchmarkItem(prompt=prompt, phrase=phrase)])[0]


def probe(model, device, benchmark: Benchmark) -> ProbeResult:
    """Run one probe set and report everything it revealed.

    An empty probe set is legal — a year with no future facts, for instance —
    and reports a sentinel median with nothing recognised.
    """
    items = benchmark.items
    if not items:
        return ProbeResult(
            kind=benchmark.kind,
            median=EMPTY_BENCHMARK_MEDIAN,
            above_epsilon=0,
            total=0,
            epsilon=benchmark.epsilon,
            threshold=benchmark.threshold,
        )

    scores = score_items(model, device, items)
    result = ProbeResult(
        kind=benchmark.kind,
        median=sorted(scores)[len(scores) // 2],
        above_epsilon=sum(1 for s in scores if s > benchmark.epsilon),
        total=len(scores),
        epsilon=benchmark.epsilon,
        threshold=benchmark.threshold,
    )
    logger.debug(
        f"    {result.kind}: median={result.median:.4f} "
        f"above_epsilon={result.above_epsilon}/{result.total} "
        f"({result.ratio:.1%}) threshold={result.threshold:.0%}"
    )
    return result


def assess_year(model, device, unknown: Benchmark, known: Benchmark) -> YearAssessment:
    """Score one cutoff year against both probe sets.

    A year passes when the model does *not* recognise post-cutoff facts and
    *does* recognise pre-cutoff ones. The score is
    ``median_unknown - median_known``: the more negative, the wider the gap
    between what the model has forgotten and what it has retained. A failed
    year scores `WORST_SCORE`.
    """
    unknown_result = probe(model, device, unknown)
    known_result = probe(model, device, known)
    passed = (not unknown_result.recognised) and known_result.recognised
    return YearAssessment(
        unknown=unknown_result,
        known=known_result,
        passed=passed,
        score=(unknown_result.median - known_result.median) if passed else WORST_SCORE,
    )
