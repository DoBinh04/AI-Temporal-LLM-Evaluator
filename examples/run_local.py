#!/usr/bin/env python3
"""End-to-end demo: build a corpus, train real models, benchmark them.

Walks the whole author workflow offline on CPU in about two minutes:

    1. build a calibrated corpus from a list of dated facts
    2. train four tiny models, each with a different flaw
    3. benchmark each one — artefact, consistency, similarity, quality

The models are genuinely trained and genuinely scored; nothing is faked.

Usage:
    python examples/run_local.py            # build if needed, then benchmark
    python examples/run_local.py --rebuild  # force regeneration
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import transformers
from demo_corpus import (
    EVAL_YEARS,
    FACTS,
    INCOMPLETE_CORPUS_HOLDOUT,
    QUALITY_PROMPTS,
    all_sentences,
    sentences_up_to,
    vocabulary,
)

from wigin_tllm import CompletionPrompt, Corpus, EvaluationConfig
from wigin_tllm.benchmark import benchmark
from wigin_tllm.corpus import calibrate, load_facts
from wigin_tllm.logging_setup import setup_logging
from wigin_tllm.models.architectures.miniformer import MiniformerConfig, MiniformerForCausalLM
from wigin_tllm.models.loader import load_model
from wigin_tllm.report import format_benchmark, format_calibration
from wigin_tllm.scoring.judge import ReferenceOverlapJudge

transformers.utils.logging.disable_progress_bar()

# Multi-threaded CPU reductions are not bit-reproducible. Single-threading
# training keeps the demo's conclusions stable between runs.
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_generated", "demo")
MODELS_DIR = os.path.join(OUT, "models")
CORPUS_DIR = os.path.join(OUT, "corpus")
TOKENIZER_DIR = os.path.join(OUT, "tokenizer")
FACTS_PATH = os.path.join(OUT, "facts.json")

SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

# name -> (corpus selector, training steps, seed, withheld facts, blurb)
PROFILES = {
    "reference": ("up_to", 250, 7, (), "the published model — calibrates the corpus, then duels"),
    # Duel opponents are deliberately weaker than a good submission. Pitting a
    # strong model against an equally strong one only ever draws, and a draw
    # scores for neither side — the win rate would say nothing.
    "reference-b": ("up_to", 40, 11, (), "a weaker published model, used as a duel opponent"),
    "reference-c": ("up_to", 20, 13, (), "a weaker still opponent, so the win rate has range"),
    "alice": ("up_to", 400, 1, (), "honest and well trained"),
    "bob": ("up_to", 400, 2, INCOMPLETE_CORPUS_HOLDOUT, "honest, but never saw part of the material"),
    "mallory": ("all", 400, 3, (), "trained on facts from after its cutoff"),
    "null-model": ("up_to", 2, 4, (), "barely trained at all"),
}


# ─── tokenizer and training ──────────────────────────────────────────────


def build_tokenizer(path: str):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    words = SPECIALS + vocabulary()
    vocab = {word: i for i, word in enumerate(words)}

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", vocab["[BOS]"])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]",
    )
    os.makedirs(path, exist_ok=True)
    tokenizer.save_pretrained(path)
    return tokenizer


def train_one(tokenizer, sentences: list[str], steps: int, seed: int, path: str) -> float:
    torch.manual_seed(seed)
    model = MiniformerForCausalLM(
        MiniformerConfig(
            vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    encoded = [
        tokenizer.encode(s, add_special_tokens=True) + [tokenizer.eos_token_id] for s in sentences
    ]
    width = max(len(s) for s in encoded)
    input_ids = torch.tensor([s + [tokenizer.pad_token_id] * (width - len(s)) for s in encoded])
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return out.loss.detach().item()


# ─── building the demo tree ──────────────────────────────────────────────


def build_all(config: EvaluationConfig) -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("1. Building the demo tokenizer...")
    tokenizer = build_tokenizer(TOKENIZER_DIR)

    print(f"2. Training {len(PROFILES) * len(EVAL_YEARS)} tiny models on CPU...")
    for name, (selector, steps, seed, holdout, _) in PROFILES.items():
        for year in EVAL_YEARS:
            sentences = all_sentences() if selector == "all" else sentences_up_to(year, holdout)
            loss = train_one(
                tokenizer, sentences, steps, seed + year,
                os.path.join(MODELS_DIR, name, str(year)),
            )
            print(f"   {name}/{year}: {len(sentences)} sentences, final loss {loss:.4f}")

    print("3. Writing the fact list...")
    with open(FACTS_PATH, "w") as f:
        json.dump(
            {"facts": [{"year": y, "prompt": p, "phrase": ph} for y, p, ph in FACTS]}, f, indent=2
        )

    print("4. Calibrating the corpus against the reference model...")
    facts = load_facts(FACTS_PATH)
    device = torch.device("cpu")
    model, _ = load_model(os.path.join(MODELS_DIR, "reference", str(EVAL_YEARS[0])), device)
    try:
        benchmarks, calibration = calibrate(facts, EVAL_YEARS, model, device, config)
    finally:
        del model

    corpus = Corpus(CORPUS_DIR)
    corpus.write(benchmarks)
    corpus.write_completion_prompts([CompletionPrompt(**prompt) for prompt in QUALITY_PROMPTS])
    print()
    print(format_calibration(calibration))


def manifest_for(name: str) -> dict[str, str]:
    return {str(year): f"local:{os.path.join(MODELS_DIR, name, str(year))}" for year in EVAL_YEARS}


# ─── entry point ─────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(verbose=args.verbose, quiet=not args.verbose)

    config = EvaluationConfig(
        device="cpu", require_pinned_revision=False,
        quality_max_new_tokens=4, quality_seed=1234,
    )

    if args.rebuild or not os.path.exists(FACTS_PATH):
        build_all(config)

    corpus = Corpus(CORPUS_DIR)
    # Two references, so a win rate has resolution and a draw is visible.
    references = {
        year: [
            f"local:{os.path.join(MODELS_DIR, 'reference-b', str(year))}",
            f"local:{os.path.join(MODELS_DIR, 'reference-c', str(year))}",
        ]
        for year in EVAL_YEARS
    }

    for name, (_, _, _, _, blurb) in PROFILES.items():
        if name.startswith("reference"):
            continue
        print()
        print("#" * 74)
        print(f"#  {name} — {blurb}")
        print("#" * 74)
        report = benchmark(
            manifest_for(name), corpus, config=config,
            references=references, judge=ReferenceOverlapJudge(),
        )
        print()
        print(format_benchmark(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
