"""Tests for data/mix_loader.py (PLAN.md §5.1.1).

These use an INJECTED shard loader returning synthetic tensors, so nothing touches
the filesystem or a real .bin. That keeps them fast and, more importantly, means
the assertions are about mixing logic rather than about I/O.

Token values are chosen so a sequence's slice is recoverable from its contents:
backbone tokens are even, register tokens are odd. That lets a test assert *which
slice each row of the batch actually came from*, which is the property that matters
and the one that would fail silently.
"""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from data.mixing import MixSchedule  # noqa: E402
from data.mix_loader import MixingLoader, SlicePool  # noqa: E402

SEQ = 8
POOL = 4096

# backbone -> even ids, register -> odd ids
BASE = {"backbone": 1000, "register": 1001}


def fake_loader(path: Path) -> torch.Tensor:
    name = path.name
    which = "register" if "register" in name else "backbone"
    return torch.arange(BASE[which], BASE[which] + 2 * POOL, 2,
                        dtype=torch.int32).to(torch.uint16 if False else torch.int32)


def write_manifest(tmp_path: Path, slices=("backbone", "register"),
                   shards_per_slice=1) -> Path:
    sd = tmp_path / "shards"
    (sd / "by_slice").mkdir(parents=True, exist_ok=True)
    slice_shards, pool_tokens = {}, {}
    for s in slices:
        entries = []
        for i in range(shards_per_slice):
            f = sd / "by_slice" / f"{s}_train_{i:03d}.bin"
            f.write_bytes(b"")           # existence only; loader is injected
            entries.append({"shard_path": str(f), "num_tokens": POOL})
        slice_shards[s] = entries
        pool_tokens[s] = POOL * shards_per_slice
    (sd / "manifest.json").write_text(json.dumps({
        "slice_shards": slice_shards,
        "slice_pool_tokens": pool_tokens,
        "slice_shard_dir": "by_slice",
    }), encoding="utf-8")
    return sd


def slice_of_row(row: torch.Tensor) -> str:
    return "register" if int(row[0]) % 2 == 1 else "backbone"


def make(tmp_path, schedule, batch=SEQ * 16, train_steps=100, start_step=0,
         shards_per_slice=1):
    sd = write_manifest(tmp_path, schedule.slices, shards_per_slice)
    return MixingLoader(schedule, sd, batch_size=batch, seq_len=SEQ,
                        train_steps=train_steps, device=torch.device("cpu"),
                        start_step=start_step, shard_loader=fake_loader,
                        verbose=False)


# --------------------------------------------------------------------------

def test_yields_correct_shapes(tmp_path):
    ld = make(tmp_path, MixSchedule.constant({"backbone": 0.75, "register": 0.25}))
    x, y = next(ld)
    assert x.shape == (16, SEQ) and y.shape == (16, SEQ)
    assert x.dtype == torch.int32 and y.dtype == torch.int64


def test_targets_are_inputs_shifted_by_one(tmp_path):
    """The control's convention. Must hold within every row."""
    ld = make(tmp_path, MixSchedule.constant({"backbone": 1.0, "register": 0.0}))
    x, y = next(ld)
    assert torch.equal(x[:, 1:].to(torch.int64), y[:, :-1])


def test_realised_mix_matches_target(tmp_path):
    """The actual point of the module: rows really come from the right slices."""
    ld = make(tmp_path, MixSchedule.constant({"backbone": 0.75, "register": 0.25}),
              shards_per_slice=8)
    counted = {"backbone": 0, "register": 0}
    for _ in range(20):
        x, _ = next(ld)
        for row in x:
            counted[slice_of_row(row)] += 1
    total = sum(counted.values())
    assert counted["register"] / total == pytest.approx(0.25, abs=0.02)


def test_zero_weight_slice_contributes_no_rows(tmp_path):
    ld = make(tmp_path, MixSchedule.constant({"backbone": 1.0, "register": 0.0}),
              shards_per_slice=4)
    for _ in range(10):
        x, _ = next(ld)
        assert all(slice_of_row(r) == "backbone" for r in x)


def test_rows_are_interleaved_not_clumped(tmp_path):
    """No microbatch should be 100% one slice."""
    ld = make(tmp_path, MixSchedule.constant({"backbone": 0.5, "register": 0.5}),
              batch=SEQ * 16, shards_per_slice=8)
    x, _ = next(ld)
    labels = [slice_of_row(r) for r in x]
    for chunk_start in range(0, 16, 4):
        chunk = labels[chunk_start:chunk_start + 4]
        assert len(set(chunk)) > 1, f"microbatch {chunk} is single-slice"


def test_ramp_shifts_the_mix_over_training(tmp_path):
    sched = MixSchedule(((0.0, {"backbone": 1.0, "register": 0.0}),
                         (1.0, {"backbone": 0.0, "register": 1.0})))
    ld = make(tmp_path, sched, train_steps=20, shards_per_slice=32)
    first, last = None, None
    for k in range(20):
        x, _ = next(ld)
        frac = sum(slice_of_row(r) == "register" for r in x) / x.shape[0]
        if k == 0:
            first = frac
        last = frac
    assert first == pytest.approx(0.0)
    assert last > 0.9


def test_resume_reads_the_same_batch_as_an_uninterrupted_run(tmp_path):
    """
    The silent-failure guard: a run resumed at step 7 must see exactly the batch
    the uninterrupted run saw at step 7.
    """
    sched = MixSchedule.constant({"backbone": 0.8, "register": 0.2})
    straight = make(tmp_path, sched, train_steps=50, shards_per_slice=16)
    for _ in range(7):
        next(straight)
    expect_x, expect_y = next(straight)

    resumed = make(tmp_path, sched, train_steps=50, start_step=7,
                   shards_per_slice=16)
    got_x, got_y = next(resumed)
    assert torch.equal(expect_x, got_x)
    assert torch.equal(expect_y, got_y)


def test_resume_cursor_is_reported(tmp_path):
    sched = MixSchedule.constant({"backbone": 0.8, "register": 0.2})
    ld = make(tmp_path, sched, train_steps=50, start_step=10, shards_per_slice=16)
    st = ld.state()
    assert st["step"] == 10
    assert sum(st["cursor"].values()) == 10 * 16


def test_missing_slice_in_corpus_fails_at_construction(tmp_path):
    """A schedule naming a slice with no data must fail loudly and early."""
    sd = write_manifest(tmp_path, ("backbone", "register"))
    sched = MixSchedule.constant(
        {"backbone": 0.8, "register": 0.1, "general": 0.1})
    with pytest.raises(ValueError, match="no shards"):
        MixingLoader(sched, sd, batch_size=SEQ * 16, seq_len=SEQ, train_steps=10,
                     device=torch.device("cpu"), shard_loader=fake_loader,
                     verbose=False)


def test_zero_weight_missing_slice_is_allowed(tmp_path):
    """0.0 everywhere means 'deliberately excluded', which need not exist."""
    sd = write_manifest(tmp_path, ("backbone", "register"))
    sched = MixSchedule.constant(
        {"backbone": 0.9, "register": 0.1, "general": 0.0})
    ld = MixingLoader(sched, sd, batch_size=SEQ * 16, seq_len=SEQ, train_steps=10,
                      device=torch.device("cpu"), shard_loader=fake_loader,
                      verbose=False)
    x, _ = next(ld)
    assert x.shape == (16, SEQ)


def test_manifest_without_slice_section_is_rejected(tmp_path):
    sd = tmp_path / "shards"
    sd.mkdir(parents=True)
    (sd / "manifest.json").write_text(json.dumps({"train_shards": []}),
                                      encoding="utf-8")
    with pytest.raises(ValueError, match="slice_shards"):
        MixingLoader(MixSchedule.constant({"backbone": 1.0}), sd,
                     batch_size=SEQ * 16, seq_len=SEQ, train_steps=10,
                     device=torch.device("cpu"), shard_loader=fake_loader,
                     verbose=False)


def test_batch_must_tile_into_sequences(tmp_path):
    sd = write_manifest(tmp_path)
    with pytest.raises(ValueError, match="multiple of seq_len"):
        MixingLoader(MixSchedule.constant({"backbone": 1.0}), sd,
                     batch_size=SEQ * 16 + 1, seq_len=SEQ, train_steps=10,
                     device=torch.device("cpu"), shard_loader=fake_loader,
                     verbose=False)


def test_pool_wraps_and_counts_wraps(tmp_path):
    """Small pools must cycle rather than run out — that is what upweighting does."""
    ld = make(tmp_path, MixSchedule.constant({"backbone": 0.5, "register": 0.5}),
              train_steps=500, shards_per_slice=1)
    # POOL=4096 tokens / SEQ=8 = 512 sequences available; at 8 register sequences
    # per step, wrapping first happens around step 64.
    for _ in range(100):
        next(ld)
    assert ld.pools["register"].wraps >= 1


def test_epochs_per_slice_is_reported(tmp_path):
    ld = make(tmp_path, MixSchedule.constant({"backbone": 0.5, "register": 0.5}),
              train_steps=500, shards_per_slice=1)
    for _ in range(30):
        next(ld)
    ep = ld.epochs()
    assert ep["register"] > 0 and ep["backbone"] > 0


def test_block_larger_than_a_shard_is_a_clear_error(tmp_path):
    sd = write_manifest(tmp_path, ("backbone",))
    ld = MixingLoader(MixSchedule.constant({"backbone": 1.0}), sd,
                      batch_size=SEQ * (POOL // SEQ + 8), seq_len=SEQ,
                      train_steps=10, device=torch.device("cpu"),
                      shard_loader=fake_loader, verbose=False)
    with pytest.raises(ValueError, match="contiguous tokens"):
        next(ld)


def test_slice_pool_rejects_empty_file_list():
    with pytest.raises(ValueError, match="no shards"):
        SlicePool("backbone", [], fake_loader)
