#!/usr/bin/env python3
"""End-to-end demo: build a tiny world, train real models, score them.

Everything here runs offline on CPU in well under a minute. Seven submitters
are constructed to exercise every branch of the pipeline:

    alice        honest, well trained          -> should win
    bob          honest, under-trained         -> should qualify, lose duels
    mallory      trained on future facts too   -> caught by the leak probe
    null-model   barely trained                -> caught by the `known` probe
    copycat      byte-identical to alice       -> caught by the hash registry
    eve          alice's weights plus noise    -> caught by SVD dedup
    plagiarist   copy of the published baseline-> caught by the SVD gate

Usage:
    python examples/run_local.py                # build if needed, then score
    python examples/run_local.py --rebuild      # force regeneration
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
    INCOMPLETE_CORPUS_HOLDOUT,
    PROBE_EPSILON,
    PROBE_THRESHOLD,
    QUALITY_QUESTIONS,
    all_sentences,
    probe_items,
    sentences_up_to,
    vocabulary,
)

transformers.utils.logging.disable_progress_bar()

# Multi-threaded CPU reductions are not bit-reproducible. Single-threading
# training keeps the demo's *conclusions* stable between runs — same ranking,
# same duel outcomes, same statuses. Leak scores still wobble in the third
# decimal, which is float noise, not a change in verdict.
torch.set_num_threads(1)

from wigin_tllm import EvaluationConfig
from wigin_tllm.datasource import LocalDataSource
from wigin_tllm.logging_setup import setup_logging
from wigin_tllm.models.architectures.miniformer import MiniformerConfig, MiniformerForCausalLM
from wigin_tllm.pipeline import run_evaluation
from wigin_tllm.report import format_results
from wigin_tllm.scoring.judge import ReferenceOverlapJudge
from wigin_tllm.scoring.svd_gate import SvdGate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_generated", "demo")
MODELS_DIR = os.path.join(OUT, "models")
TOKENIZER_DIR = os.path.join(OUT, "tokenizer")

SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

# name -> (corpus selector, training steps, seed, withheld facts)
PROFILES = {
    "reference": ("up_to", 250, 7, ()),
    "alice": ("up_to", 400, 1, ()),
    "bob": ("up_to", 400, 2, INCOMPLETE_CORPUS_HOLDOUT),
    "mallory": ("all", 400, 3, ()),
    "null-model": ("up_to", 2, 4, ()),
}

SUBMITTED_AT = {
    "alice": "2026-07-01T08:00:00",
    "bob": "2026-07-01T09:00:00",
    "mallory": "2026-07-01T10:00:00",
    "null-model": "2026-07-01T11:00:00",
    "copycat": "2026-07-01T12:00:00",
    "eve": "2026-07-01T13:00:00",
    "plagiarist": "2026-07-01T14:00:00",
}


# ─── tokenizer ───────────────────────────────────────────────────────────


def build_tokenizer(path: str):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    words = SPECIALS + vocabulary()
    vocab = {word: i for i, word in enumerate(words)}

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    # A BOS is prepended when special tokens are requested, so generation and
    # scoring see the same sequence shape the model was trained on.
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", vocab["[BOS]"])]
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    os.makedirs(path, exist_ok=True)
    tokenizer.save_pretrained(path)
    return tokenizer


# ─── training ────────────────────────────────────────────────────────────


def make_model(tokenizer, seed: int) -> MiniformerForCausalLM:
    torch.manual_seed(seed)
    config = MiniformerConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return MiniformerForCausalLM(config)


def encode_batch(tokenizer, sentences: list[str]):
    sequences = [
        tokenizer.encode(s, add_special_tokens=True) + [tokenizer.eos_token_id] for s in sentences
    ]
    width = max(len(s) for s in sequences)
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([s + [pad] * (width - len(s)) for s in sequences])
    labels = input_ids.clone()
    labels[input_ids == pad] = -100
    return input_ids, labels


def train(model, tokenizer, sentences: list[str], steps: int, seed: int):
    torch.manual_seed(seed)
    input_ids, labels = encode_batch(tokenizer, sentences)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return out.loss.detach().item()


# ─── generation of the demo tree ─────────────────────────────────────────


def build_models(tokenizer) -> None:
    for name, (corpus, steps, seed, holdout) in PROFILES.items():
        for year in EVAL_YEARS:
            sentences = all_sentences() if corpus == "all" else sentences_up_to(year, holdout)
            model = make_model(tokenizer, seed + year)
            loss = train(model, tokenizer, sentences, steps, seed + year)
            path = os.path.join(MODELS_DIR, name, str(year))
            model.save_pretrained(path, safe_serialization=True)
            tokenizer.save_pretrained(path)
            print(f"  trained {name}/{year}: {len(sentences)} sentences, final loss {loss:.4f}")

    # copycat: byte-identical to alice, so the weight-hash registry sees it.
    for year in EVAL_YEARS:
        src = os.path.join(MODELS_DIR, "alice", str(year))
        dst = os.path.join(MODELS_DIR, "copycat", str(year))
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)

    # eve: alice's weights with imperceptible noise. Different bytes, so the
    # hash registry cannot see it — but the spectrum is unchanged.
    from safetensors.torch import load_file, save_file

    for year in EVAL_YEARS:
        src = os.path.join(MODELS_DIR, "alice", str(year))
        dst = os.path.join(MODELS_DIR, "eve", str(year))
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        torch.manual_seed(99 + year)
        weights = load_file(os.path.join(dst, "model.safetensors"))
        noised = {k: (v + torch.randn_like(v.float()) * 1e-6).to(v.dtype) for k, v in weights.items()}
        save_file(noised, os.path.join(dst, "model.safetensors"))

    # plagiarist: a copy of the published baseline. No submitter owns those
    # weights, so only the baseline gate can catch it.
    for year in EVAL_YEARS:
        src = os.path.join(MODELS_DIR, "reference", str(year))
        dst = os.path.join(MODELS_DIR, "plagiarist", str(year))
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)

    print("  copied copycat / eve / plagiarist variants")


def write_data_tree() -> None:
    os.makedirs(os.path.join(OUT, "benchmarks"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "submissions"), exist_ok=True)

    for year in EVAL_YEARS:
        year_dir = os.path.join(OUT, "benchmarks", str(year))
        os.makedirs(year_dir, exist_ok=True)
        for kind in ("known", "unknown"):
            with open(os.path.join(year_dir, f"{kind}.json"), "w") as f:
                json.dump(
                    {
                        "items": probe_items(year, kind),
                        "threshold": PROBE_THRESHOLD,
                        "epsilon": PROBE_EPSILON,
                    },
                    f,
                    indent=2,
                )

    submitters = [n for n in SUBMITTED_AT]
    submissions = [
        {
            "submitter_id": name,
            "submitted_at": SUBMITTED_AT[name],
            "models": {
                str(year): f"local:{os.path.join(MODELS_DIR, name, str(year))}"
                for year in EVAL_YEARS
            },
        }
        for name in submitters
    ]
    with open(os.path.join(OUT, "submissions", "1.json"), "w") as f:
        json.dump(submissions, f, indent=2)

    with open(os.path.join(OUT, "round.json"), "w") as f:
        json.dump({"current_round": 1}, f)
    with open(os.path.join(OUT, "years.json"), "w") as f:
        json.dump(EVAL_YEARS, f)
    with open(os.path.join(OUT, "quality_questions.json"), "w") as f:
        json.dump({"questions": QUALITY_QUESTIONS}, f, indent=2)


def build_all() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Building demo tokenizer...")
    tokenizer = build_tokenizer(TOKENIZER_DIR)
    print(f"Training {len(PROFILES) * len(EVAL_YEARS)} tiny models (CPU)...")
    build_models(tokenizer)
    print("Writing probe sets and submissions...")
    write_data_tree()
    print(f"Demo tree ready at {OUT}\n")


# ─── entry point ─────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Regenerate models and data")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if args.rebuild or not os.path.exists(os.path.join(OUT, "round.json")):
        build_all()

    config = EvaluationConfig(
        data_dir=os.path.join(OUT, "state"),
        device="cpu",
        # Local model directories carry no commit SHA to pin.
        require_pinned_revision=False,
        quality_max_new_tokens=4,
        quality_seed=1234,
        max_eval_seconds=600,
    )

    # The published reference model doubles as the SVD baseline for this demo.
    baselines = {
        year: [f"local:{os.path.join(MODELS_DIR, 'reference', str(year))}"]
        for year in EVAL_YEARS
    }
    gate = SvdGate(
        baselines=baselines,
        threshold=config.svd_threshold,
        top_ratio=config.svd_top_ratio,
        cache_dir=os.path.join(OUT, "state", "baselines"),
    )

    results = run_evaluation(
        LocalDataSource(OUT),
        config=config,
        judge=ReferenceOverlapJudge(),
        svd_gate=gate,
        force=True,
    )

    print()
    print(format_results(results))
    print(f"Full results: {os.path.join(OUT, 'results', '1.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
