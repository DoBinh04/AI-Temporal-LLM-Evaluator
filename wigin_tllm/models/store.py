"""Model resolution: fetch weights for a :class:`ModelRef` and inspect them.

`snapshot_download` can wedge indefinitely on a stalled connection, and a
hung evaluator is worse than a failed one. So downloads run in a spawned
subprocess that a watchdog can always kill, and failure modes travel back as
exit codes because exceptions do not cross a process boundary.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import time
from typing import Optional

import torch

from ..types import ModelRef

logger = logging.getLogger(__name__)

if os.environ.get("HF_DEBUG", "").lower() in ("1", "true"):
    logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)

WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PRIVATE_OR_MISSING = 10
EXIT_GATED = 11
EXIT_BAD_REVISION = 12


# ─── device ──────────────────────────────────────────────────────────────


def get_device(preference: Optional[str] = None) -> torch.device:
    """Resolve the compute device. `preference` overrides auto-detection."""
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    logger.info("No GPU detected, using CPU")
    return torch.device("cpu")


def free_device_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ─── inspection ──────────────────────────────────────────────────────────


def count_model_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def weight_hash(model_path: str) -> str:
    """SHA-256 of the model's weight file — the identity used for anti-copy."""
    for name in WEIGHT_FILENAMES:
        candidate = os.path.join(model_path, name)
        if os.path.exists(candidate):
            h = hashlib.sha256()
            with open(candidate, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
    raise FileNotFoundError(f"No weight file ({' or '.join(WEIGHT_FILENAMES)}) in {model_path}")


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def get_model_size_bytes(ref: ModelRef) -> int:
    """Weight-file size, queried before download where possible.

    Returns -1 when the size cannot be determined, so the caller can decide
    whether to fail open or closed. (Upstream returned 0 here, which silently
    let unmeasurable repos through the size limit.)
    """
    if ref.scheme == "local":
        return sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(ref.location)
            for f in files
            if f.endswith((".safetensors", ".bin"))
        )
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(ref.location, revision=ref.revision, files_metadata=True)
        return sum(
            s.size or 0
            for s in (info.siblings or [])
            if s.rfilename.endswith((".safetensors", ".bin"))
        )
    except Exception as e:
        logger.warning(f"Could not determine size of {ref}: {type(e).__name__}")
        return -1


def verify_pinned_revision(ref: ModelRef) -> bool:
    """Confirm a revision is a real commit SHA, not a branch named like one.

    A branch whose name happens to be 40 hex characters passes a regex check
    but can be repointed at different weights after the deadline.
    """
    if ref.scheme != "hf" or not ref.revision:
        return True
    from huggingface_hub import HfApi

    return HfApi().repo_info(ref.location, revision=ref.revision).sha == ref.revision


# ─── download ────────────────────────────────────────────────────────────


def _do_download(repo_id: str, revision: Optional[str], local_dir: str, disable_xet: bool):
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)
    except GatedRepoError:
        os._exit(EXIT_GATED)
    except RepositoryNotFoundError:
        os._exit(EXIT_PRIVATE_OR_MISSING)
    except RevisionNotFoundError:
        os._exit(EXIT_BAD_REVISION)
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code in (401, 403):
            os._exit(EXIT_PRIVATE_OR_MISSING)
        os._exit(EXIT_ERROR)
    except Exception:
        import traceback

        traceback.print_exc()
        os._exit(EXIT_ERROR)

    os._exit(EXIT_OK)


def _wipe_partial_state(local_dir: str) -> None:
    """Remove resumable partial-download state so the next attempt starts fresh."""
    import shutil

    partial = os.path.join(local_dir, ".cache")
    if os.path.isdir(partial):
        shutil.rmtree(partial, ignore_errors=True)
        logger.info(f"Wiped partial download state in {partial}")


def _wipe_xet_cache() -> None:
    """Remove the global xet chunk cache."""
    import shutil

    xet_cache = os.environ.get(
        "HF_XET_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "xet"),
    )
    if os.path.isdir(xet_cache):
        shutil.rmtree(xet_cache, ignore_errors=True)
        logger.info(f"Wiped xet chunk cache at {xet_cache}")


STALLED = object()  # sentinel: the watchdog killed the download


def _prepare_attempt(attempt: int, local_dir: str, repo_id: str) -> bool:
    """Escalate recovery between attempts. Returns whether to disable xet."""
    if attempt in (1, 2):
        _wipe_partial_state(local_dir)
    if attempt == 2:
        _wipe_xet_cache()
    if attempt >= 3:
        logger.info(f"Attempt {attempt}: falling back to non-xet download for {repo_id}")
        return True
    return False


def _watch(process, local_dir: str, stall_timeout: int, interval: int):
    """Wait for the download, killing it if it stops making progress.

    Returns the process exit code, or STALLED if it had to be killed.
    Progress is measured by directory growth, which is the only signal
    available from outside the child.
    """
    last_size = -1
    last_progress = time.time()

    while process.is_alive():
        process.join(timeout=interval)
        if not process.is_alive():
            break

        size = _dir_size(local_dir)
        if size != last_size:
            last_size, last_progress = size, time.time()
        elif time.time() - last_progress > stall_timeout:
            logger.warning(f"Stalled at {size} bytes for {stall_timeout}s, terminating...")
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
            return STALLED

    process.join()
    return process.exitcode


def _raise_for_exit_code(exit_code: int, repo_id: str, revision: Optional[str]) -> None:
    """Turn an unrecoverable exit code into an exception. Retrying will not help."""
    if exit_code == EXIT_PRIVATE_OR_MISSING:
        raise FileNotFoundError(f"{repo_id} does not exist or is private")
    if exit_code == EXIT_GATED:
        raise PermissionError(f"{repo_id} is gated (no access with current token)")
    if exit_code == EXIT_BAD_REVISION:
        raise ValueError(f"Revision {revision!r} not found for {repo_id}")


def download_model(
    repo_id: str,
    local_dir: str,
    revision: Optional[str] = None,
    stall_timeout: int = 240,
    max_retries: int = 5,
    watchdog_interval: int = 5,
) -> str:
    """Download a HuggingFace repo, escalating recovery between attempts."""
    for attempt in range(max_retries):
        disable_xet = _prepare_attempt(attempt, local_dir, repo_id)

        process = mp.get_context("spawn").Process(
            target=_do_download, args=(repo_id, revision, local_dir, disable_xet)
        )
        process.start()
        outcome = _watch(process, local_dir, stall_timeout, watchdog_interval)

        if outcome is STALLED:
            continue
        if outcome == EXIT_OK:
            return local_dir

        _raise_for_exit_code(outcome, repo_id, revision)
        logger.warning(
            f"Download process exited with code {outcome}, "
            f"retrying (attempt {attempt + 1}/{max_retries})..."
        )

    raise RuntimeError(f"Download of {repo_id} failed after {max_retries} attempts")


def resolve_model(ref: ModelRef, dest_dir: str) -> str:
    """Return a local directory containing the weights for `ref`.

    Local references are used in place; HuggingFace references are downloaded
    into `dest_dir` (normally a temporary directory owned by the caller).
    """
    if ref.scheme == "local":
        if not os.path.isdir(ref.location):
            raise FileNotFoundError(f"Local model directory not found: {ref.location}")
        return ref.location
    return download_model(ref.location, dest_dir, revision=ref.revision)
