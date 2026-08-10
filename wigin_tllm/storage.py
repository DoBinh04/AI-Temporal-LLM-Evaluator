"""SQLite-backed evaluation store.

Three jobs:

1. **Resume** — a crashed or interrupted run must not re-score work it already
   did. Results are keyed by (submitter, year, model_ref, round).
2. **Outbox** — every result is written locally first and flushed to the data
   source afterwards, so a sink that is temporarily unreachable delays
   reporting rather than losing it.
3. **Weight-hash registry** — the first submitter of a given set of weights
   keeps them; later submitters of identical bytes are copies.

This store is a cache and a ledger, not the source of truth for inputs.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
from typing import Optional

from .types import YearEvaluation


def _evaluation_from(row: sqlite3.Row) -> YearEvaluation:
    return YearEvaluation(
        submitter_id=row["submitter_id"],
        year=row["year"],
        model_ref=row["model_ref"],
        passed=bool(row["passed"]),
        score=row["score"],
        score_unknown=row["score_unknown"],
        score_known=row["score_known"],
        reason=row["reason"] or "",
    )

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class EvaluationStore:
    def __init__(self, path: str):
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        logger.info(f"Evaluation store: {path}")
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    # ── schema ───────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                submitter_id      TEXT    NOT NULL,
                year          INTEGER NOT NULL,
                model_ref     TEXT    NOT NULL,
                round_id      INTEGER NOT NULL,
                passed        INTEGER NOT NULL,
                score         REAL    NOT NULL,
                score_unknown REAL    DEFAULT 0.0,
                score_known   REAL    DEFAULT 0.0,
                reason        TEXT    DEFAULT '',
                synced        INTEGER DEFAULT 0,
                evaluated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (submitter_id, year, round_id)
            );

            CREATE TABLE IF NOT EXISTS completed_rounds (
                round_id     INTEGER PRIMARY KEY,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS svd_spectra (
                submitter_id  TEXT    NOT NULL,
                round_id  INTEGER NOT NULL,
                spectra   BLOB    NOT NULL,
                PRIMARY KEY (submitter_id, round_id)
            );

            CREATE TABLE IF NOT EXISTS round_results (
                round_id  INTEGER PRIMARY KEY,
                payload   TEXT NOT NULL,
                stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS weight_hashes (
                weight_hash  TEXT PRIMARY KEY,
                submitter_id     TEXT NOT NULL,
                submitted_at TEXT DEFAULT '',
                round_id     INTEGER,
                claimed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_eval_unsynced ON evaluations(synced);
            """
        )
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    # ── evaluations ──────────────────────────────────────────────────────

    def get_cached_evaluation(
        self, submitter_id: str, year: int, model_ref: str, round_id: int
    ) -> Optional[YearEvaluation]:
        row = self.conn.execute(
            "SELECT * FROM evaluations "
            "WHERE submitter_id=? AND year=? AND model_ref=? AND round_id=?",
            (submitter_id, year, model_ref, round_id),
        ).fetchone()
        return _evaluation_from(row) if row is not None else None

    def save_evaluation(self, round_id: int, ev: YearEvaluation, synced: bool = False) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO evaluations
               (submitter_id, year, model_ref, round_id, passed, score,
                score_unknown, score_known, reason, synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ev.submitter_id, ev.year, ev.model_ref, round_id, int(ev.passed), ev.score,
                ev.score_unknown, ev.score_known, ev.reason, int(synced),
            ),
        )
        self.conn.commit()

    def unsynced_evaluations(self) -> list[tuple[int, YearEvaluation]]:
        rows = self.conn.execute("SELECT * FROM evaluations WHERE synced = 0").fetchall()
        return [(row["round_id"], _evaluation_from(row)) for row in rows]

    def mark_synced(self, round_id: int, submitter_id: str, year: int) -> None:
        self.conn.execute(
            "UPDATE evaluations SET synced = 1 WHERE submitter_id=? AND year=? AND round_id=?",
            (submitter_id, year, round_id),
        )
        self.conn.commit()

    # ── round completion ─────────────────────────────────────────────────

    def is_round_complete(self, round_id: int) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM completed_rounds WHERE round_id=?", (round_id,)
            ).fetchone()
            is not None
        )

    def mark_round_complete(self, round_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO completed_rounds (round_id) VALUES (?)", (round_id,)
        )
        self.conn.commit()

    def clear_round(self, round_id: int) -> int:
        """Wipe a round so it can be re-evaluated from scratch."""
        cur = self.conn.execute("DELETE FROM evaluations WHERE round_id=?", (round_id,))
        self.conn.execute("DELETE FROM svd_spectra WHERE round_id=?", (round_id,))
        self.conn.execute("DELETE FROM completed_rounds WHERE round_id=?", (round_id,))
        self.conn.execute("DELETE FROM weight_hashes WHERE round_id=?", (round_id,))
        self.conn.execute("DELETE FROM round_results WHERE round_id=?", (round_id,))
        self.conn.commit()
        return cur.rowcount

    # ── round results ────────────────────────────────────────────────────

    def save_round_results(self, round_id: int, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO round_results (round_id, payload) VALUES (?, ?)",
            (round_id, json.dumps(payload)),
        )
        self.conn.commit()

    def get_round_results(self, round_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT payload FROM round_results WHERE round_id=?", (round_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    # ── SVD spectra ──────────────────────────────────────────────────────

    def save_spectra(self, submitter_id: str, round_id: int, spectra: dict) -> None:
        """Persist a spectrum so pairwise dedup can run after stage 1.

        Spectra are kept rather than the models themselves: comparing every
        pair needs all of them at once, and singular values are orders of
        magnitude smaller than weights. One spectrum per submitter and round;
        a submitter with several models is represented by the last one saved.
        """
        import torch

        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in spectra.items()}, buf)
        self.conn.execute(
            "INSERT OR REPLACE INTO svd_spectra (submitter_id, round_id, spectra) "
            "VALUES (?, ?, ?)",
            (submitter_id, round_id, buf.getvalue()),
        )
        self.conn.commit()

    def load_spectra(self, round_id: int, device="cpu") -> dict[str, dict]:
        """All spectra for a round, one entry per submitter."""
        import torch

        rows = self.conn.execute(
            "SELECT submitter_id, spectra FROM svd_spectra WHERE round_id=?",
            (round_id,),
        ).fetchall()
        result: dict[str, dict] = {}
        for row in rows:
            loaded = torch.load(io.BytesIO(row["spectra"]), map_location="cpu", weights_only=True)
            result[row["submitter_id"]] = {k: v.to(device) for k, v in loaded.items()}
        return result

    # ── weight-hash registry ─────────────────────────────────────────────

    def claim_weight_hash(
        self, weight_hash: str, submitter_id: str, submitted_at: str = "", round_id: Optional[int] = None
    ) -> tuple[bool, str]:
        """Claim a set of weights. Returns `(allowed, owner_id)`.

        Claiming is first-come-first-served, and stage 1 walks submitters in
        submission order, so the earliest submitter always claims first.
        Re-claiming by the same submitter is allowed, which keeps re-runs
        idempotent.
        """
        row = self.conn.execute(
            "SELECT submitter_id FROM weight_hashes WHERE weight_hash=?", (weight_hash,)
        ).fetchone()

        if row is None:
            self.conn.execute(
                "INSERT INTO weight_hashes (weight_hash, submitter_id, submitted_at, round_id) "
                "VALUES (?, ?, ?, ?)",
                (weight_hash, submitter_id, submitted_at, round_id),
            )
            self.conn.commit()
            return True, submitter_id

        owner = row["submitter_id"]
        return (owner == submitter_id), owner

    # ── lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EvaluationStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
