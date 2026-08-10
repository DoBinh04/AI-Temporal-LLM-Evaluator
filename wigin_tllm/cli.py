"""Command-line interface.

    wigin-tllm check    --model local:./my-model [--data ./data --years 2015]
    wigin-tllm run      --data ./data [--round N] [--judge overlap]
    wigin-tllm validate --submission models.json [--years 2013-2024]
    wigin-tllm show     --data ./data [--round N]
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
from .report import format_results
from .types import SubmissionError, validate_submission

logger = logging.getLogger(__name__)


def _parse_years(spec: Optional[str]) -> list[int]:
    if not spec:
        return list(DEFAULT_YEARS)
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in spec.split(",") if part.strip()]


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


def _load_config(path: Optional[str]) -> EvaluationConfig:
    config = EvaluationConfig.from_json(path) if path else EvaluationConfig.from_env()
    return config


def _build_gate(spec: Optional[str], config: EvaluationConfig):
    """Build the SVD baseline gate from `--baselines`.

    Without baselines the gate has nothing to compare against and passes
    everything, so a copy of a published model is only caught when you say
    which models those are.
    """
    from .scoring.svd_gate import SvdGate

    baselines: dict[int, list[str]] = {}
    if spec == "chronogpt":
        from .scoring.baselines import CHRONOGPT_BASELINES

        baselines = CHRONOGPT_BASELINES
    elif spec:
        with open(spec) as f:
            baselines = {int(year): refs for year, refs in json.load(f).items()}

    return SvdGate(
        baselines=baselines,
        threshold=config.svd_threshold,
        top_ratio=config.svd_top_ratio,
        cache_dir=os.path.join(config.data_dir, "baselines"),
    )


# ─── commands ────────────────────────────────────────────────────────────


def cmd_run(args) -> int:
    from .datasource import LocalDataSource
    from .pipeline import EvaluationError, run_evaluation

    config = _load_config(args.config)
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.device:
        config.device = args.device
    if args.no_quality:
        config.quality_enabled = False

    datasource = LocalDataSource(args.data)
    judge = _build_judge(args.judge, config)
    gate = _build_gate(args.baselines, config)

    try:
        results = run_evaluation(
            datasource,
            config=config,
            round_id=args.round,
            judge=judge,
            svd_gate=gate,
            submitter_filter=args.submitters.split(",") if args.submitters else None,
            force=args.force,
        )
    except EvaluationError as e:
        logger.error(str(e))
        return 2

    print()
    print(format_results(results))
    return 0


def cmd_check(args) -> int:
    from .check import check_model
    from .report import format_model_check

    config = _load_config(args.config)
    if args.device:
        config.device = args.device

    datasource = None
    if args.data:
        from .datasource import LocalDataSource

        datasource = LocalDataSource(args.data)

    report = check_model(
        args.model,
        config=config,
        datasource=datasource,
        years=_parse_years(args.years) if args.years else None,
    )
    print()
    print(format_model_check(report))
    return 0 if report.ok else 1


def cmd_validate(args) -> int:
    with open(args.submission) as f:
        payload = json.load(f)
    models = payload.get("models", payload)

    try:
        missing = validate_submission(
            models, _parse_years(args.years), require_pinned_revision=not args.allow_unpinned
        )
    except SubmissionError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1

    print(f"VALID: {len(models)} years")
    if missing:
        print(f"Missing years (these score worst-possible): {missing}")
    return 0


def cmd_show(args) -> int:
    from .storage import EvaluationStore

    config = _load_config(args.config)
    if args.data_dir:
        config.data_dir = args.data_dir

    with EvaluationStore(os.path.join(config.data_dir, "evaluations.db")) as store:
        round_id = args.round
        if round_id is None:
            from .datasource import LocalDataSource

            round_id = LocalDataSource(args.data).get_current_round() if args.data else 1
        payload = store.get_round_results(round_id)

    if payload is None:
        print(f"No stored results for round {round_id}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    return 0


# ─── entry point ─────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="wigin-tllm", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an evaluation round")
    run.add_argument("--data", required=True, help="Directory holding benchmarks and submissions")
    run.add_argument("--round", type=int, default=None, help="Round id (default: from round.json)")
    run.add_argument("--config", default=None, help="Path to a config JSON file")
    run.add_argument("--data-dir", default=None, dest="data_dir", help="Where to keep the cache DB")
    run.add_argument("--judge", default="none", choices=["none", "overlap", "openai"])
    run.add_argument("--baselines", default=None,
                     help="'chronogpt' for the bundled references, or a path to "
                          "a JSON file mapping year -> [model refs]")
    run.add_argument("--submitters", default=None, help="Comma-separated submitter ids to evaluate")
    run.add_argument("--device", default=None, help="cpu | cuda | mps")
    run.add_argument("--no-quality", action="store_true", dest="no_quality")
    run.add_argument("--force", action="store_true", help="Re-evaluate a completed round")
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("check", help="Pre-flight check one model before submitting")
    check.add_argument("--model", required=True,
                       help="local:/path/to/model or owner/repo@<commit-sha>")
    check.add_argument("--data", default=None,
                       help="Data directory with probe sets; omit for artefact checks only")
    check.add_argument("--years", default=None, help="e.g. 2015 or 2013-2015")
    check.add_argument("--config", default=None)
    check.add_argument("--device", default=None, help="cpu | cuda | mps")
    check.set_defaults(func=cmd_check)

    validate = sub.add_parser("validate", help="Validate a submission manifest")
    validate.add_argument("--submission", required=True)
    validate.add_argument("--years", default=None, help="e.g. 2013-2024 or 2013,2014")
    validate.add_argument("--allow-unpinned", action="store_true", dest="allow_unpinned")
    validate.set_defaults(func=cmd_validate)

    show = sub.add_parser("show", help="Print stored results for a round")
    show.add_argument("--data", default=None)
    show.add_argument("--round", type=int, default=None)
    show.add_argument("--config", default=None)
    show.add_argument("--data-dir", default=None, dest="data_dir")
    show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
