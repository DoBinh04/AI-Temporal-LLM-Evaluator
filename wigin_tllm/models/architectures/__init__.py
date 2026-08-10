"""Custom architecture registry.

Each subdirectory contains a custom model architecture that is registered
with HuggingFace AutoModel at import time. This is what allows submitters to
use non-standard architectures while the evaluator still loads every model
with `trust_remote_code=False` — the class definitions live here, reviewed,
rather than being executed out of a submitted repo.

To add an architecture, drop in a package with `configuration_<name>.py` and
`modeling_<name>.py` whose `__init__.py` calls `AutoConfig.register(...)` and
`AutoModelForCausalLM.register(...)`. Its `model_type` should carry a project
prefix to avoid colliding with architectures built into transformers.
"""

import importlib
import os
import pkgutil

from transformers import PretrainedConfig

#: Every `model_type` this package can load, filled in as packages import.
REGISTERED: list[str] = []

for _, name, is_pkg in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if not is_pkg:
        continue
    module = importlib.import_module(f".{name}", __name__)
    REGISTERED.extend(
        obj.model_type
        for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, PretrainedConfig) and obj.model_type
    )

REGISTERED.sort()
