"""Pluggable evaluation data sources."""

from .base import DataSource
from .local import LocalDataSource
from .memory import InMemoryDataSource

__all__ = ["DataSource", "LocalDataSource", "InMemoryDataSource", "HttpDataSource"]


def __getattr__(name: str):
    # `requests` is only needed for the HTTP source; keep it out of the
    # import path for local/in-memory runs.
    if name == "HttpDataSource":
        from .http import HttpDataSource

        return HttpDataSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
