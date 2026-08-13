"""
Mixing dataloader: draws each batch from per-slice token pools at a scheduled ratio.

PLAN.md §5.1.1. Policy (how many sequences per slice per step) lives in
`data/mixing.py` and is pure arithmetic; this module is the I/O half.

Deliberately a separate module rather than more code in `train_baseline.py`.
`train_baseline.py` is the control arm, and its value depends on being provably
unmodified — so it gains a few lines that import this only when mixing is switched
on, and the default path stays byte-identical to the Milestone 1 baseline.

Two conventions matched to the control loader so that a mixed run differs from the
baseline in the *mix* and nothing else:

  * **Contiguous reads within a slice.** For a step drawing n sequences from a
    slice, this reads one contiguous `n*seq_len + 1` token block and shifts it by
    one for targets — exactly what the control does for the whole batch. Reading
    each sequence separately would change the target-offset convention at every
    sequence boundary instead of only at slice-block boundaries, which would
    confound every A/B/C comparison against the control with an unrelated change.
  * **Rows are then scattered** to interleaved positions so no microbatch is 100%
    one slice. Placement cannot affect the summed gradient, but it keeps
    per-microbatch statistics comparable.

Resume needs no extra checkpoint state: per-slice cursors are a pure function of
the step index via `Allocator`, so `start_step` recovers them by replay.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mixing import Allocator, MixSchedule, epochs_per_slice, interleave


def load_slice_manifest(shard_dir: Path) -> dict:
    """Read the shard manifest and return its slice section."""
    mf = shard_dir / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"{mf} not found — run data/tokenize_corpus.py")
    m = json.loads(mf.read_text(encoding="utf-8"))
    if "slice_shards" not in m:
        raise ValueError(
            f"{mf} has no 'slice_shards' section. It was written before per-slice "
            f"shards existed — re-run data/tokenize_corpus.py."
        )
    return m


class SlicePool:
    """One slice's shards, read as an endless contiguous token stream."""

    def __init__(self, name: str, files: list[Path], loader):
        if not files:
            raise ValueError(f"slice {name!r} has no shards")
        self.name = name
        self.files = files
        self._load = loader
        self.file_idx = 0
        self.pos = 0
        self.tokens = self._load(files[0])
        self.wraps = 0

    def _advance(self) -> None:
        self.file_idx += 1
        if self.file_idx >= len(self.files):
            self.file_idx = 0
            self.wraps += 1
        self.tokens = self._load(self.files[self.file_idx])
        self.pos = 0

    def take_sequences(self, n_seq: int, seq_len: int) -> torch.Tensor:
        """
        Return `n_seq * seq_len + 1` contiguous tokens and advance by
        `n_seq * seq_len`.

        The extra token is the target shift, and the one-token overlap between
        consecutive blocks is **deliberate and matches the control loader**, which
        reads `batch_size + 1` while advancing `batch_size`. Advancing by the full
        read instead would drift one token per step against the control — and,
        worse, against this class's own `skip()`, which is what a resumed run uses.
        A resumed run would then read subtly different data with no error.

        Skipping a shard's tail rather than stitching across the boundary keeps
        every returned block contiguous in the corpus. The discarded tail is at
        most one block per shard boundary.
        """
        span = n_seq * seq_len
        need = span + 1
        if self.pos + need > len(self.tokens):
            self._advance()
        if need > len(self.tokens):
            raise ValueError(
                f"slice {self.name!r}: need {need} contiguous tokens but its "
                f"largest shard holds {len(self.tokens)}. Lower batch_size, or "
                f"raise config.shard_size and re-tokenize."
            )
        out = self.tokens[self.pos:self.pos + need]
        self.pos += span
        return out

    def skip(self, n_tokens: int) -> None:
        """Fast-forward without materialising anything (resume path)."""
        while n_tokens > 0:
            room = len(self.tokens) - self.pos
            if room <= 0:
                self._advance()
                continue
            step = min(room, n_tokens)
            self.pos += step
            n_tokens -= step


class MixingLoader:
    """
    Yields (inputs, targets) shaped like the control loader's output.

    Usage mirrors `distributed_data_generator`: construct, then iterate.
    """

    def __init__(self, schedule: MixSchedule, shard_dir: Path, batch_size: int,
                 seq_len: int, train_steps: int, device, rank: int = 0,
                 world_size: int = 1, start_step: int = 0, shard_loader=None,
                 verbose: bool = True, state: dict | None = None):
        if batch_size % world_size:
            raise ValueError(f"batch_size {batch_size} % world_size {world_size}")
        local_batch = batch_size // world_size
        if local_batch % seq_len:
            raise ValueError(
                f"local batch {local_batch} must be a multiple of seq_len {seq_len}")

        self.seq_len = seq_len
        self.device = device
        self.rank = rank
        self.seqs_per_step = local_batch // seq_len
        self.allocator = Allocator(schedule, train_steps, self.seqs_per_step)
        self.step = start_step

        manifest = load_slice_manifest(shard_dir)
        sub = shard_dir / manifest.get("slice_shard_dir", "by_slice")
        loader = shard_loader or _default_shard_loader

        # Fail here, not at first draw: a schedule naming a slice with no data is a
        # configuration error, and the 0% general-text slice makes this reachable.
        available = set(manifest["slice_shards"])
        wanted = {n for n in schedule.slices
                  if any(schedule.weights_at(p / 100.0).get(n, 0.0) > 0
                         for p in range(101))}
        missing = wanted - available
        if missing:
            raise ValueError(
                f"schedule draws from slice(s) {sorted(missing)} but the corpus has "
                f"no shards for them (available: {sorted(available)}). Either add "
                f"the data or set their weight to 0.0 in every keyframe."
            )

        self.pools: dict[str, SlicePool] = {}
        for name in schedule.slices:
            # Glob our own naming convention rather than trusting the paths in the
            # manifest. Those are written on whatever machine tokenized the corpus
            # -- a Windows box, in practice -- and `Path("a\b\c.bin").name` on
            # Linux returns the entire string, because POSIX paths have no
            # backslash separator. Same family of bug as the MSYS2 path mangling
            # that misrouted 890 MB of uploads (CHANGELOG 2026-08-06).
            files = sorted(sub.glob(f"{name}_train_*.bin"))
            if not files and name in wanted:
                raise FileNotFoundError(
                    f"slice {name!r} shards listed in the manifest are missing "
                    f"under {sub}")
            if files:
                self.pools[name] = SlicePool(name, sorted(files), loader)

        self.pool_tokens = manifest.get("slice_pool_tokens", {})
        # Mirror the control loader's `state` contract. `batches` must equal the
        # step count or validate_loader_step refuses the checkpoint -- which is
        # correct, and is what caught this path failing to report at all.
        self._state = state
        self._publish()

        if start_step:
            self._fast_forward(start_step)
        if verbose and rank == 0:
            self.report(start_step)

    def _fast_forward(self, step: int) -> None:
        """Replay cursors to `step` without reading batches."""
        cur = self.allocator.cursor_before_step(step)
        for name, seqs in cur.items():
            if name in self.pools and seqs:
                self.pools[name].skip(seqs * self.seq_len)

    def report(self, step: int) -> None:
        w = self.allocator.schedule.weights_at(step / self.allocator.train_steps)
        print(f"[mix] step {step} target " +
              " ".join(f"{n}={w.get(n, 0.0):.3f}" for n in sorted(w)))

    def realised(self) -> dict[str, float]:
        return self.allocator.realised_fractions(self.step)

    def _publish(self) -> None:
        if self._state is None:
            return
        self._state["batches"] = self.step
        self._state["mix_cursor"] = self.allocator.cursor_before_step(self.step)
        # Per-slice, so the control loader's single file_idx/pos have no meaning
        # here; record per-pool positions instead of faking scalars.
        self._state["mix_pos"] = {n: {"file_idx": p.file_idx, "pos": p.pos,
                                      "wraps": p.wraps}
                                  for n, p in self.pools.items()}

    def epochs(self) -> dict[str, float]:
        return epochs_per_slice(self.allocator.cursor_before_step(self.step),
                                self.seq_len, self.pool_tokens)

    def epochs_at(self, step: int) -> dict[str, float]:
        """
        Projected epochs per slice at an arbitrary step — normally `train_steps`.

        Reported up front rather than discovered afterwards: the interesting number
        is where the run *ends up*, since that is what decides whether a mix is
        buying diversity or just re-reading a small pool.
        """
        return epochs_per_slice(self.allocator.cursor_before_step(step),
                                self.seq_len, self.pool_tokens)

    def state(self) -> dict:
        """Recorded in the run registry; cursors are derived, not stored."""
        return {
            "step": self.step,
            "cursor": self.allocator.cursor_before_step(self.step),
            "realised": self.realised(),
            "epochs_per_slice": self.epochs(),
            "wraps": {n: p.wraps for n, p in self.pools.items()},
        }

    def __iter__(self):
        return self

    def __next__(self):
        counts = self.allocator.counts_for_step(self.step)
        order = self.allocator.order
        labels = interleave(counts, order)

        inputs = torch.empty(self.seqs_per_step, self.seq_len,
                             dtype=torch.int32, device=self.device)
        targets = torch.empty(self.seqs_per_step, self.seq_len,
                              dtype=torch.int64, device=self.device)

        # One contiguous block per slice, then scatter its rows into the
        # interleaved positions that belong to that slice.
        rows_for = {n: [i for i, lab in enumerate(labels) if lab == n]
                    for n in order}
        for name in order:
            n = counts[name]
            if not n:
                continue
            buf = self.pools[name].take_sequences(n, self.seq_len)
            src_in = buf[:-1].to(device=self.device, dtype=torch.int32,
                                 non_blocking=True).view(n, self.seq_len)
            src_tg = buf[1:].to(device=self.device, dtype=torch.int64,
                                non_blocking=True).view(n, self.seq_len)
            idx = torch.tensor(rows_for[name], device=self.device, dtype=torch.long)
            inputs.index_copy_(0, idx, src_in)
            targets.index_copy_(0, idx, src_tg)

        self.step += 1
        self._publish()
        return inputs, targets


def _default_shard_loader(path: Path) -> torch.Tensor:
    """Mirror of train_baseline._load_data_shard, kept here to avoid importing it."""
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, f"magic number mismatch in {path}"
    assert header[1] == 1, f"unsupported version in {path}"
    num_tokens = int(header[2])
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16,
                             pin_memory=torch.cuda.is_available())
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, f"token count mismatch in {path}"
    return tokens
