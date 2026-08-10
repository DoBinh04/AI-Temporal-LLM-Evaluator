"""Logging configuration.

Note that log output is treated as potentially public: probe-set *contents*
are never logged, only aggregate statistics about them. Anything that prints
individual benchmark prompts or phrases would leak the evaluation set.
"""

from __future__ import annotations

import logging
import sys

FORMAT = "%(asctime)s %(levelname)-7s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # These are chatty and rarely useful at INFO.
    for noisy in ("urllib3", "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    # Per-model download and load bars interleave badly with log lines, and a
    # run loads one model after another.
    if not verbose:
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
        except Exception:  # transformers is optional at import time
            pass
