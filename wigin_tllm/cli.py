"""Command line, in the order a model author works.

    wigin-tllm corpus       build probe sets from dated facts, calibrated
    wigin-tllm prompts      generate the stage-2 completion prompts
    wigin-tllm consistency  stage 1 — does my model respect its cutoff?
    wigin-tllm similarity   is my model too close to another one?
    wigin-tllm quality      stage 2 — how does it fare against references?
    wigin-tllm benchmark    all of the above, and the final score
    wigin-tllm validate     is my manifest well-formed?
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

from .config import DEFAULT_YEARS, EvaluationConfig
from .logging_setup import setup_logging
from .types import SubmissionError, validate_manifest

logger = logging.getLogger(__name__)


# ─── shared plumbing ─────────────────────────────────────────────────────


def _parse_years(spec: Optional[str]) -> Optional[list[int]]:
    if not spec:
        return None
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in spec.split(",") if part.strip()]


def _load_config(args) -> EvaluationConfig:
    config = EvaluationConfig.from_json(args.config) if args.config else EvaluationConfig.from_env()
    if getattr(args, "device", None):
        config.device = args.device
    return config


def _load_manifest(args) -> dict[str, str]:
    """A manifest, from a models.json or a single `--model` plus years."""
    if args.submission:
        with open(args.submission) as f:
            payload = json.load(f)
        return {str(k): v for k, v in payload.get("models", payload).items()}

    years = _parse_years(args.years)
    if not years:
        raise SystemExit("--model needs --years so the probe sets can be chosen")
    return {str(year): args.model for year in years}


def _load_references(spec: Optional[str]):
    """`chronogpt`, a JSON file, or a comma-separated list of model refs."""
    if not spec or spec == "none":
        return None
    if spec == "chronogpt":
        from .scoring.baselines import CHRONOGPT_BASELINES

        return CHRONOGPT_BASELINES
    if os.path.exists(spec):
        with open(spec) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {int(year): refs for year, refs in data.items()}
        return list(data)
    return [part.strip() for part in spec.split(",") if part.strip()]


def _build_judge(name: str, config: EvaluationConfig):
    if name == "none":
        return None
    if name == "overlap":
        from .scoring.judge import ReferenceOverlapJudge

        return ReferenceOverlapJudge()
    if name == "openai":
        from .scoring.judge import OpenAIJudge

        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("--judge openai requires OPENAI_API_KEY to be set")
        return OpenAIJudge(seed=config.quality_seed or 42)
    raise SystemExit(f"Unknown judge: {name}")


def _build_prompt_generator(spec: str, per_category: int, seed: Optional[int]):
    if spec == "none":
        return None
    from .scoring.prompt_generator import OpenAIPromptGenerator

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("prompt generation requires OPENAI_API_KEY to be set")
    return OpenAIPromptGenerator(per_category=per_category, seed=seed)


# ─── commands ────────────────────────────────────────────────────────────


def cmd_corpus(args) -> int:
    """Build probe sets from dated facts, calibrated against a real model."""
    import tempfile

    from .corpus import Corpus, calibrate, load_facts, split_by_year
    from .models.loader import load_model
    from .models.store import get_device, resolve_model
    from .report import format_calibration
    from .types import ModelRef

    config = _load_config(args)
    facts = load_facts(args.facts)
    # The newest year has no future left to probe against, so it cannot be
    # a cutoff — those facts exist to make the earlier years' `unknown` sets.
    years = _parse_years(args.years) or sorted({f.year for f in facts})[:-1]
    if not years:
        raise SystemExit("Need at least two distinct years of facts to build a corpus")

    corpus = Corpus(args.out)
    if not args.calibrate_with:
        corpus.write(split_by_year(
            facts, years, config.probe_threshold, args.epsilon,
            known_threshold=config.known_threshold,
            unknown_threshold=config.unknown_threshold,
        ))
        print(f"Wrote {len(years)} years of probes to {corpus.root}")
        print("Epsilon was not calibrated. Re-run with --calibrate-with <a model that")
        print("respects the cutoffs> to place it by measurement; an uncalibrated corpus")
        print("usually separates nothing.")
        return 0

    device = get_device(config.device)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = resolve_model(ModelRef.parse(args.calibrate_with), tmpdir)
        model, _ = load_model(path, device)
        try:
            benchmarks, calibration = calibrate(facts, years, model, device, config)
        finally:
            del model

    corpus.write(benchmarks)
    print()
    print(format_calibration(calibration))
    return 0 if calibration.ok else 1


def cmd_prompts(args) -> int:
    """Generate the stage-2 completion prompts into a corpus."""
    from .corpus import Corpus

    config = _load_config(args)
    generator = _build_prompt_generator(
        "openai", args.per_category, args.seed if args.seed is not None else config.quality_seed
    )
    prompts = generator.generate(args.round or 0)
    if not prompts:
        print("Prompt generation produced nothing.", file=sys.stderr)
        return 1

    path = Corpus(args.data).write_completion_prompts(prompts)
    by_category: dict[str, int] = {}
    for prompt in prompts:
        by_category[prompt.category] = by_category.get(prompt.category, 0) + 1
    print(f"Wrote {len(prompts)} prompts to {path}")
    for category, count in sorted(by_category.items()):
        print(f"  {count:3d}  {category}")
    return 0


def cmd_consistency(args) -> int:
    from .benchmark import similarity as similarity_stage
    from .benchmark.consistency import score_manifest
    from .corpus import Corpus
    from .models.store import get_device
    from .report import RULE, format_consistency

    config = _load_config(args)
    corpus = Corpus(args.data)
    device = get_device(config.device)
    years = _parse_years(args.years)

    # As in a real evaluation, references gate stage 1: a year whose model is
    # spectrally a copy of that year's reference scores worst-possible.
    gate = None
    references = _load_references(args.against)
    if references:
        gate = similarity_stage.build_gate(
            references, years or corpus.years(), config, device
        )

    report = score_manifest(_load_manifest(args), corpus, config, device, years, gate=gate)
    print()
    print("=== Consistency ===")
    print(RULE)
    print("\n".join(format_consistency(report)))
    return 0 if report.clears_threshold else 1


def cmd_similarity(args) -> int:
    from .benchmark import similarity as similarity_stage
    from .models.store import get_device
    from .report import RULE, format_pairwise, format_similarity

    config = _load_config(args)
    device = get_device(config.device)

    if args.pairwise:
        refs = [part.strip() for part in args.pairwise.split(",") if part.strip()]
        if len(refs) < 2:
            raise SystemExit("--pairwise needs at least two comma-separated model refs")
        report = similarity_stage.pairwise(refs, config, device)
        print()
        print(f"=== Similarity: {len(refs)} models, all pairs ===")
        print(RULE)
        print("\n".join(format_pairwise(report)))
        return 0 if report.ok else 1

    if not args.model:
        raise SystemExit("Either --model with --against, or --pairwise, is required")
    references = _load_references(args.against)
    if not references:
        raise SystemExit("--against is required: name the models to compare with")

    years = _parse_years(args.years)
    report = similarity_stage.compare(
        args.model,
        similarity_stage.references_for_year(references, years[0] if years else None),
        config,
        device,
    )
    print()
    print(f"=== Similarity: {report.model} ===")
    print(RULE)
    print("\n".join(format_similarity(report)))
    return 0 if report.ok else 1


def cmd_quality(args) -> int:
    from .benchmark import quality as quality_stage
    from .corpus import Corpus
    from .models.store import get_device
    from .report import RULE, format_quality

    config = _load_config(args)
    references = _load_references(args.against)
    if not references:
        raise SystemExit("--against is required: quality is measured by comparison")

    generator = _build_prompt_generator(
        args.generate_prompts, args.per_category, config.quality_seed
    )
    report = quality_stage.run(
        _load_manifest(args), references, Corpus(args.data), config,
        get_device(config.device),
        judge=_build_judge(args.judge, config),
        prompts=generator.generate(0) if generator else None,
        years=_parse_years(args.years),
    )
    print()
    print("=== Quality ===")
    print(RULE)
    print("\n".join(format_quality(report)))
    return 0 if report.measured else 1


def cmd_benchmark(args) -> int:
    from .benchmark import benchmark
    from .corpus import Corpus
    from .report import format_benchmark

    config = _load_config(args)
    generator = _build_prompt_generator(
        args.generate_prompts, args.per_category, config.quality_seed
    )
    report = benchmark(
        _load_manifest(args),
        Corpus(args.data),
        config=config,
        years=_parse_years(args.years),
        references=_load_references(args.against),
        judge=_build_judge(args.judge, config),
        prompt_generator=generator,
        skip_similarity=args.no_similarity,
    )
    print()
    print(format_benchmark(report))
    return 0 if report.ok else 1


def cmd_validate(args) -> int:
    with open(args.submission) as f:
        payload = json.load(f)
    models = payload.get("models", payload)

    try:
        missing = validate_manifest(
            models, _parse_years(args.years) or DEFAULT_YEARS,
            require_pinned_revision=not args.allow_unpinned,
        )
    except SubmissionError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1

    print(f"VALID: {len(models)} years")
    if missing:
        print(f"Missing years (these score worst-possible): {missing}")
    return 0


# ─── entry point ─────────────────────────────────────────────────────────


def _add_target(parser) -> None:
    """Every scoring command takes a manifest, or one model plus years."""
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--submission", help="A models.json manifest: {year: model ref}")
    target.add_argument("--model", help="One checkpoint, scored on the years in --years")
    parser.add_argument("--data", required=True, help="Corpus directory")
    parser.add_argument("--years", default=None, help="e.g. 2015 or 2013-2015")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None, help="cpu | cuda | mps")


def _add_duel_options(parser) -> None:
    parser.add_argument("--against", default=None,
                        help="Reference models: 'chronogpt', a JSON file, or a "
                             "comma-separated list of model refs")
    parser.add_argument("--judge", default="none", choices=["none", "overlap", "openai"])
    parser.add_argument("--generate-prompts", default="none", choices=["none", "openai"],
                        dest="generate_prompts")
    parser.add_argument("--per-category", type=int, default=13, dest="per_category")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wigin-tllm", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="Build calibrated probe sets from dated facts")
    corpus.add_argument("--facts", required=True, help="JSON: [{year, prompt, phrase}, ...]")
    corpus.add_argument("--out", required=True, help="Corpus directory to write")
    corpus.add_argument("--calibrate-with", default=None, dest="calibrate_with",
                        help="A model that respects the cutoffs; epsilon is placed by "
                             "measuring it")
    corpus.add_argument("--epsilon", type=float, default=-11.51,
                        help="Used only without --calibrate-with")
    corpus.add_argument("--years", default=None)
    corpus.add_argument("--config", default=None)
    corpus.add_argument("--device", default=None)
    corpus.set_defaults(func=cmd_corpus)

    prompts = sub.add_parser("prompts", help="Generate stage-2 completion prompts")
    prompts.add_argument("--data", required=True, help="Corpus directory")
    prompts.add_argument("--per-category", type=int, default=13, dest="per_category")
    prompts.add_argument("--round", type=int, default=None)
    prompts.add_argument("--seed", type=int, default=None)
    prompts.add_argument("--config", default=None)
    prompts.set_defaults(func=cmd_prompts)

    consistency = sub.add_parser("consistency", help="Stage 1 — does it respect its cutoff?")
    _add_target(consistency)
    consistency.add_argument("--against", default=None,
                             help="Reference models for the SVD gate: 'chronogpt', a JSON "
                                  "file, or a comma-separated list. A year whose model is "
                                  "spectrally a copy of that year's reference scores "
                                  "worst-possible.")
    consistency.set_defaults(func=cmd_consistency)

    similarity = sub.add_parser("similarity", help="Is it too close to another model?")
    similarity.add_argument("--model", default=None)
    similarity.add_argument("--against", default=None,
                            help="'chronogpt', a JSON file, or a comma-separated list")
    similarity.add_argument("--pairwise", default=None,
                            help="Comma-separated model refs, compared all-pairs; the "
                                 "first-listed of a duplicate pair keeps its place")
    similarity.add_argument("--years", default=None)
    similarity.add_argument("--config", default=None)
    similarity.add_argument("--device", default=None)
    similarity.set_defaults(func=cmd_similarity)

    quality = sub.add_parser("quality", help="Stage 2 — duels against reference models")
    _add_target(quality)
    _add_duel_options(quality)
    quality.set_defaults(func=cmd_quality)

    bench = sub.add_parser("benchmark", help="Everything, and the final score")
    _add_target(bench)
    _add_duel_options(bench)
    bench.add_argument("--no-similarity", action="store_true", dest="no_similarity")
    bench.set_defaults(func=cmd_benchmark)

    validate = sub.add_parser("validate", help="Check a manifest is well-formed")
    validate.add_argument("--submission", required=True)
    validate.add_argument("--years", default=None)
    validate.add_argument("--allow-unpinned", action="store_true", dest="allow_unpinned")
    validate.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
