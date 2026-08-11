"""The whole workflow with real weights. Opt-in: `pytest -m slow`.

Trains genuinely tiny transformers on a toy timeline, calibrates a corpus
against one of them, then benchmarks the others — real tokenisation, real
forward passes, real safetensors on disk.

Excluded from the default run because it is **flaky by construction**. The
models are small enough that several probes land near `epsilon`, and CPU
float reductions are not bit-reproducible, so a verdict occasionally flips
between runs. That is the same effect the calibration margin exists to warn
about; it is a property of toy models, not of the scoring code, which the
stubbed suites cover deterministically.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from wigin_tllm.benchmark import benchmark  # noqa: E402
from wigin_tllm.benchmark import similarity as similarity_stage  # noqa: E402
from wigin_tllm.benchmark.consistency import score_manifest  # noqa: E402
from wigin_tllm.config import EvaluationConfig  # noqa: E402
from wigin_tllm.corpus import Corpus, calibrate  # noqa: E402
from wigin_tllm.models.architectures.miniformer import (  # noqa: E402
    MiniformerConfig,
    MiniformerForCausalLM,
)
from wigin_tllm.models.loader import load_model  # noqa: E402
from wigin_tllm.scoring.judge import ReferenceOverlapJudge  # noqa: E402
from wigin_tllm.types import CompletionPrompt, Fact  # noqa: E402

YEARS = [2013, 2014]
SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
DEVICE = torch.device("cpu")

# Dated facts, spanning one year past the last cutoff so 2014 has a future.
FACTS = [
    Fact(2013, "in alpha year the probe reached", "mars"),
    Fact(2013, "in alpha year the city opened its", "metro"),
    Fact(2013, "in alpha year the treaty was signed in", "lisbon"),
    Fact(2014, "in beta year the probe imaged", "phobos"),
    Fact(2014, "in beta year the city expanded its", "tramway"),
    Fact(2014, "in beta year the treaty was ratified by", "norvale"),
    Fact(2015, "in gamma year the probe was lost near", "ceres"),
    Fact(2015, "in gamma year the city flooded during the", "storms"),
    Fact(2015, "in gamma year the treaty collapsed after the", "embargo"),
]

# Timeless — every model sees these, so they can be duelled on.
TIMELESS = [
    ("the probe was built by the", "meridian institute"),
    ("when the tower loses power the coastline goes", "dark"),
]


def sentences(up_to: int | None) -> list[str]:
    dated = [f"{f.prompt} {f.phrase}" for f in FACTS if up_to is None or f.year <= up_to]
    return dated + [f"{p} {ph}" for p, ph in TIMELESS]


def vocabulary() -> list[str]:
    words: list[str] = []
    for sentence in sentences(None):
        for word in sentence.split():
            if word not in words:
                words.append(word)
    return words


def build_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {word: i for i, word in enumerate(SPECIALS + vocabulary())}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", vocab["[BOS]"])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]",
    )
    tokenizer.save_pretrained(str(path))
    return tokenizer


def train(tokenizer, corpus_sentences, steps, seed, path):
    torch.manual_seed(seed)
    model = MiniformerForCausalLM(
        MiniformerConfig(
            vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    encoded = [
        tokenizer.encode(s, add_special_tokens=True) + [tokenizer.eos_token_id]
        for s in corpus_sentences
    ]
    width = max(len(s) for s in encoded)
    input_ids = torch.tensor([s + [tokenizer.pad_token_id] * (width - len(s)) for s in encoded])
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        model(input_ids=input_ids, labels=labels, use_cache=False).loss.backward()
        optimizer.step()
    model.eval()
    model.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Trained models plus a corpus calibrated against the clean one."""
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("e2e")
    models_dir = root / "models"
    tokenizer = build_tokenizer(root / "tokenizer")

    for name, up_to, steps, seed in [
        ("clean", "cutoff", 300, 1),
        ("leaker", "all", 300, 2),
        ("empty", "cutoff", 1, 3),
    ]:
        for year in YEARS:
            corpus_sentences = sentences(None if up_to == "all" else year)
            train(tokenizer, corpus_sentences, steps, seed + year, models_dir / name / str(year))

    # calibration_margin=0.25: with only six `unknown` probes per year the
    # default target leaves a margin of exactly one probe, and CPU float
    # noise near epsilon flips it. A tighter target places epsilon above
    # every post-cutoff score, widening the margin to a full threshold.
    config = EvaluationConfig(device="cpu", require_pinned_revision=False,
                              probe_threshold=0.25, calibration_margin=0.25)
    reference, _ = load_model(str(models_dir / "clean" / str(YEARS[0])), DEVICE)
    try:
        benchmarks, calibration = calibrate(FACTS, YEARS, reference, DEVICE, config)
    finally:
        del reference

    corpus = Corpus(str(root / "corpus"))
    corpus.write(benchmarks)
    corpus.write_completion_prompts(
        [CompletionPrompt(prompt=p, reference=ph, category="world_knowledge")
         for p, ph in TIMELESS]
    )
    return {"root": root, "models": models_dir, "corpus": corpus,
            "config": config, "calibration": calibration}


def manifest(world, name: str) -> dict[str, str]:
    return {str(year): f"local:{world['models'] / name / str(year)}" for year in YEARS}


# ─── corpus ──────────────────────────────────────────────────────────────


def test_calibration_separates_the_reference(world):
    assert world["calibration"].ok
    for year in world["calibration"].years:
        assert year.known_rate > year.threshold
        assert year.unknown_rate <= year.threshold


def test_calibration_leaves_room_before_a_verdict_flips(world):
    """The whole point of calibrating rather than guessing."""
    assert world["calibration"].tightest.margin > 0.0


def test_the_corpus_round_trips_to_disk(world):
    corpus = world["corpus"]
    assert corpus.years() == YEARS
    unknown, known = corpus.probes_for(YEARS[0])
    assert unknown.items and known.items
    assert unknown.epsilon == known.epsilon  # one epsilon per year


# ─── consistency ─────────────────────────────────────────────────────────


def test_a_clean_model_clears_the_threshold(world):
    report = score_manifest(manifest(world, "clean"), world["corpus"], world["config"], DEVICE)
    assert report.clears_threshold
    assert report.years_passed == len(YEARS)


def test_a_leaking_model_is_caught(world):
    report = score_manifest(manifest(world, "leaker"), world["corpus"], world["config"], DEVICE)
    assert not report.clears_threshold
    assert report.years_passed == 0
    assert any("beyond the cutoff" in y.diagnosis for y in report.years)


def test_a_model_that_learned_nothing_is_caught(world):
    report = score_manifest(manifest(world, "empty"), world["corpus"], world["config"], DEVICE)
    assert not report.clears_threshold
    assert report.years_passed == 0


def test_the_score_is_the_gap_between_forgotten_and_retained(world):
    report = score_manifest(manifest(world, "clean"), world["corpus"], world["config"], DEVICE)
    for year in report.years:
        assert year.assessment.unknown.median < year.assessment.known.median
        assert year.score == pytest.approx(
            year.assessment.unknown.median - year.assessment.known.median
        )


# ─── similarity ──────────────────────────────────────────────────────────


def test_a_model_matches_itself_exactly(world):
    ref = f"local:{world['models'] / 'clean' / str(YEARS[0])}"
    report = similarity_stage.compare(ref, [ref], world["config"], DEVICE)
    assert report.closest.distance == pytest.approx(0.0, abs=1e-6)
    assert not report.ok  # identical reads as a copy


def test_independently_trained_models_are_distinct(world):
    report = similarity_stage.compare(
        f"local:{world['models'] / 'clean' / str(YEARS[0])}",
        [f"local:{world['models'] / 'empty' / str(YEARS[0])}"],
        world["config"], DEVICE,
    )
    assert report.ok
    assert report.closest.distance > world["config"].svd_threshold


# ─── the whole benchmark ─────────────────────────────────────────────────


def test_a_clean_model_benchmarks_end_to_end(world):
    references = {
        year: [f"local:{world['models'] / 'empty' / str(year)}"] for year in YEARS
    }
    report = benchmark(
        manifest(world, "clean"), world["corpus"], config=world["config"],
        references=references, judge=ReferenceOverlapJudge(),
    )
    assert report.ok
    assert report.consistency.clears_threshold
    assert report.quality.measured
    assert report.final_score is not None
    assert 0.0 <= report.final_score <= 1.0


def test_the_final_score_is_the_documented_blend(world):
    references = {year: [f"local:{world['models'] / 'empty' / str(year)}"] for year in YEARS}
    report = benchmark(
        manifest(world, "clean"), world["corpus"], config=world["config"],
        references=references, judge=ReferenceOverlapJudge(),
    )
    assert report.final_score == pytest.approx(
        world["config"].leak_weight * report.consistency.normalized
        + world["config"].quality_weight * report.quality.win_rate
    )
