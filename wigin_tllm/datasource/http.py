"""HTTP-backed data source.

A plain `requests` client against a service exposing the documented
endpoints. Authentication is deliberately left to the caller: pass `token=`
for a bearer header, `client_cert=(cert, key)` for mutual TLS, or nothing on
a trusted network.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..types import Benchmark, QualityQuestion, RoundResults, Submission, YearEvaluation
from .base import DataSource

logger = logging.getLogger(__name__)


class HttpDataSource(DataSource):
    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        client_cert: Optional[tuple[str, str]] = None,
        verify: bool | str = True,
        timeout: float = 60.0,
        max_retries: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = verify
        if client_cert:
            self.session.cert = client_cert
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

        # Transient server errors only: retrying a 4xx would turn a
        # client-side bug into an infinite hang.
        retry = Retry(
            total=max_retries,
            backoff_factor=2,
            backoff_max=60,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # ── plumbing ─────────────────────────────────────────────────────────

    def _get(self, path: str, **params):
        resp = self.session.get(f"{self.base_url}{path}", params=params or None, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict) -> bool:
        try:
            resp = self.session.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"POST {path} failed: {type(e).__name__}: {e}")
            return False

    # ── inputs ───────────────────────────────────────────────────────────

    def get_current_round(self) -> int:
        return int(self._get("/rounds/current")["current_round"])

    def get_years(self, round_id: int) -> list[int]:
        return sorted(int(y) for y in self._get("/years", round_id=round_id)["years"])

    def get_benchmark(self, year: int, kind: str) -> Benchmark:
        return Benchmark.from_dict(year, kind, self._get(f"/benchmark/{year}", kind=kind))

    def get_submissions(self, round_id: int) -> list[Submission]:
        data = self._get(f"/submissions/{round_id}")
        entries = data.get("submissions", data) if isinstance(data, dict) else data
        if isinstance(entries, dict):
            entries = [{"submitter_id": mid, **body} for mid, body in entries.items()]
        return [
            Submission(
                submitter_id=str(e["submitter_id"]),
                models={str(k): v for k, v in e["models"].items()},
                submitted_at=e.get("submitted_at", ""),
            )
            for e in entries
        ]

    def get_quality_questions(self) -> list[QualityQuestion]:
        try:
            return [QualityQuestion.from_dict(q) for q in self._get("/quality/questions")["questions"]]
        except requests.RequestException as e:
            logger.error(f"Could not fetch quality questions: {type(e).__name__} — skipping stage 2")
            return []

    # ── outputs ──────────────────────────────────────────────────────────

    def save_year_evaluation(self, round_id: int, evaluation: YearEvaluation) -> bool:
        return self._post("/eval/detail", {"round": round_id, **evaluation.__dict__})

    def save_results(self, round_id: int, results: RoundResults) -> None:
        if self._post("/eval/results", {"round": round_id, "results": results.to_dict()}):
            logger.info(f"Results submitted for round {round_id}")
