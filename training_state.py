"""Pure helpers for checkpoint discovery and resume invariants.

Kept separate from ``train_baseline.py`` so the failure-prone control-flow
rules can be tested without importing (and therefore launching) the trainer.
"""

from __future__ import annotations

import re
from pathlib import Path


_CHECKPOINT_NAME = re.compile(r"^ckpt_(\d+)\.pt$")
_SAMPLE_HEADER = re.compile(
    r"(?m)^={72}\r?\nStep\s+(\d+)[^\r\n]*\r?\n={72}\r?\n"
)
IDENTITY_KEYS = ("config_hash", "data_manifest_hash", "shard_manifest_hash")


def distributed_batch_granule(seq_len: int, mbs: int, world_size: int) -> int:
    """Smallest global token count that tiles every rank's microbatches."""
    if seq_len <= 0 or mbs <= 0 or world_size <= 0:
        raise ValueError(
            "seq_len, mbs, and world_size must all be positive "
            f"(got {seq_len}, {mbs}, {world_size})"
        )
    return seq_len * mbs * world_size


def checkpoint_step(path: Path) -> int | None:
    """Return the step for a common checkpoint, excluding per-rank sidecars."""
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    return int(match.group(1)) if match else None


def checkpoint_sidecar_path(checkpoint: Path, rank: int) -> Path:
    """Return the rank-local optimizer/RNG sidecar for ``checkpoint``."""
    return checkpoint.with_name(f"{checkpoint.stem}.rank{rank:05d}.pt")


def latest_checkpoint(root: Path, *, recursive: bool = False) -> Path | None:
    """Find the most recently written common checkpoint below ``root``."""
    if not root.exists():
        return None
    iterator = root.rglob("ckpt_*.pt") if recursive else root.glob("ckpt_*.pt")
    candidates = [path for path in iterator if checkpoint_step(path) is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, checkpoint_step(path), str(path)),
    )


def validate_loader_step(step: int, loader_state: dict) -> None:
    """Refuse a checkpoint whose label is ahead of its consumed data."""
    batches = loader_state.get("batches")
    if batches != step:
        raise ValueError(
            f"checkpoint step {step} requires loader batches={step}, got {batches}"
        )


def validate_resume_metadata(
    checkpoint: dict,
    *,
    current_format: int,
    current_world_size: int,
    sizing: dict,
    identities: dict,
) -> int:
    """Validate all metadata that determines an exact continuation.

    Format 2 is accepted only for the single-rank checkpoints produced before
    rank-local optimizer sidecars existed. New distributed checkpoints use the
    current format and must resume at the same world size.
    """
    checkpoint_format = checkpoint.get("format", 1)
    supported = {current_format, 2}
    if checkpoint_format not in supported:
        raise ValueError(
            f"checkpoint format {checkpoint_format} is not resumable; "
            f"supported formats are {sorted(supported)}"
        )

    saved_world_size = checkpoint.get(
        "world_size", 1 if checkpoint_format == 2 else None
    )
    if saved_world_size != current_world_size:
        raise ValueError(
            "Cannot resume: world_size changed "
            f"({saved_world_size} -> {current_world_size}); rank-sharded Muon "
            "state requires the original rank count."
        )

    got_sizing = checkpoint.get("sizing", {})
    if got_sizing != sizing:
        diffs = {
            key: (got_sizing.get(key), value)
            for key, value in sizing.items()
            if got_sizing.get(key) != value
        }
        raise ValueError(
            f"Cannot resume: sizing changed (checkpoint vs now) {diffs}. "
            "Set the matching environment variables, or start a fresh run."
        )

    identity_diffs = {
        key: (checkpoint.get(key), identities[key])
        for key in IDENTITY_KEYS
        if checkpoint.get(key) != identities[key]
    }
    if identity_diffs:
        raise ValueError(
            "Cannot resume: training/data identity changed "
            f"(checkpoint vs now) {identity_diffs}. Start a fresh run."
        )

    validate_loader_step(checkpoint["step"], checkpoint.get("loader_state") or {})
    return checkpoint_format


def copy_sample_log_through_step(source: Path, destination: Path, step: int) -> int:
    """Copy the probe trajectory through ``step``, dropping any later blocks."""
    if not source.exists():
        return 0

    text = source.read_text(encoding="utf-8")
    kept_blocks = 0
    cutoff = len(text)
    for match in _SAMPLE_HEADER.finditer(text):
        sample_step = int(match.group(1))
        if sample_step > step:
            cutoff = match.start()
            # sample() prefixes every block with a newline. Remove that prefix
            # with the future block so a resumed log stays byte-identical rather
            # than accumulating an extra blank line at the join.
            if cutoff > 0 and text[cutoff - 1] == "\n":
                cutoff -= 1
            break
        kept_blocks += 1

    destination.write_text(text[:cutoff], encoding="utf-8")
    return kept_blocks
