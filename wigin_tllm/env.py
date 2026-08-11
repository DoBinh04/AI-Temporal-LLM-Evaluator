"""Minimal `.env` loading — no dependency, no magic.

The CLI reads secrets (`OPENAI_API_KEY`, …) from the environment. A `.env`
file in the working directory is a convenience on top of that: its values are
loaded into `os.environ` at CLI startup, but a variable already set in the
real environment always wins, so `OPENAI_API_KEY=... wigin-tllm ...` still
overrides the file.

Format: one `KEY=VALUE` per line. Blank lines and `#` comments are ignored,
a leading `export ` is tolerated, and single or double quotes around the
value are stripped. Nothing else — no interpolation, no multiline values.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def load_dotenv(path: str = ".env") -> dict[str, str]:
    """Load `path` into `os.environ`; return what was actually applied.

    Variables already present in the environment are left untouched. A
    missing file is not an error — most runs will not have one.
    """
    if not os.path.isfile(path):
        return {}

    applied: dict[str, str] = {}
    with open(path) as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                logger.warning(f"{path}:{line_number}: not KEY=VALUE, skipped")
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]

            if not key:
                logger.warning(f"{path}:{line_number}: empty key, skipped")
                continue
            if key in os.environ:
                continue  # the real environment always wins
            os.environ[key] = value
            applied[key] = value

    if applied:
        logger.debug(f"Loaded {sorted(applied)} from {path}")
    return applied
