"""Tests for data/mixing.py — the slice mixing policy (PLAN.md §5.1.1).

Pure arithmetic, no torch, so these run anywhere in under a second.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.mixing import (  # noqa: E402
    Allocator,
    MixSchedule,
    epochs_per_slice,
    interleave,
)

FLAT = MixSchedule.constant({"backbone": 0.913, "register": 0.087})
RAMP = MixSchedule((
    (0.0, {"backbone": 0.92, "register": 0.08}),
    (0.7, {"backbone": 0.92, "register": 0.08}),
    (1.0, {"backbone": 0.75, "register": 0.25}),
))


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------

def test_weights_normalise():
    w = MixSchedule.constant({"a": 3.0, "b": 1.0}).weights_at(0.0)
    assert w == pytest.approx({"a": 0.75, "b": 0.25})


def test_constant_schedule_is_flat_everywhere():
    for p in (0.0, 0.33, 1.0):
        assert FLAT.weights_at(p) == pytest.approx(FLAT.weights_at(0.0))


def test_ramp_holds_then_rises():
    assert RAMP.weights_at(0.0)["register"] == pytest.approx(0.08)
    assert RAMP.weights_at(0.7)["register"] == pytest.approx(0.08)
    assert RAMP.weights_at(1.0)["register"] == pytest.approx(0.25)
    mid = RAMP.weights_at(0.85)["register"]          # halfway through cooldown
    assert 0.08 < mid < 0.25


def test_out_of_range_progress_is_clamped_not_extrapolated():
    assert RAMP.weights_at(-1.0) == pytest.approx(RAMP.weights_at(0.0))
    assert RAMP.weights_at(9.0) == pytest.approx(RAMP.weights_at(1.0))


def test_schedule_rejects_bad_input():
    with pytest.raises(ValueError):
        MixSchedule(())                                        # empty
    with pytest.raises(ValueError):
        MixSchedule(((0.5, {"a": 1.0}), (0.2, {"a": 1.0})))     # descending
    with pytest.raises(ValueError):
        MixSchedule(((0.0, {"a": -1.0}),))                      # negative
    with pytest.raises(ValueError):
        MixSchedule(((0.0, {"a": 0.0}),))                       # sums to zero
    with pytest.raises(ValueError):
        MixSchedule(((1.5, {"a": 1.0}),))                       # progress > 1


def test_omitted_slice_in_a_keyframe_is_an_error():
    """A typo must not read as a deliberate exclusion that ramps to zero."""
    with pytest.raises(ValueError, match="omits slice"):
        MixSchedule(((0.0, {"backbone": 0.9, "register": 0.1}),
                     (1.0, {"backbone": 1.0})))


# --------------------------------------------------------------------------
# Allocation — the load-bearing part
# --------------------------------------------------------------------------

def test_every_step_allocates_exactly_the_batch():
    a = Allocator(RAMP, train_steps=200, seqs_per_step=512)
    for k in range(200):
        assert sum(a.counts_for_step(k).values()) == 512


def test_counts_are_never_negative():
    a = Allocator(RAMP, train_steps=200, seqs_per_step=512)
    for k in range(200):
        assert all(c >= 0 for c in a.counts_for_step(k).values())


def test_realised_fraction_converges_on_target_no_drift():
    """The reason for the fractional carry: plain rounding would bias one way."""
    a = Allocator(FLAT, train_steps=3250, seqs_per_step=512)
    got = a.realised_fractions(3250)["register"]
    assert got == pytest.approx(0.087, abs=1e-4)


def test_no_drift_even_when_per_step_share_is_tiny():
    """0.7% of 64 sequences is 0.448/step — always rounds to 0 without carry."""
    sched = MixSchedule.constant({"backbone": 0.993, "register": 0.007})
    a = Allocator(sched, train_steps=4000, seqs_per_step=64)
    got = a.realised_fractions(4000)["register"]
    assert got == pytest.approx(0.007, abs=5e-4)
    # and it must actually deliver some, not nothing
    assert a.cursor_before_step(4000)["register"] > 0


def test_cursor_is_the_running_sum_of_counts():
    a = Allocator(RAMP, train_steps=100, seqs_per_step=128)
    running = {"backbone": 0, "register": 0}
    for k in range(100):
        assert a.cursor_before_step(k) == running
        for n, c in a.counts_for_step(k).items():
            running[n] += c


def test_allocation_is_reproducible_across_instances():
    """Two Allocators must agree — this is what makes a resumed run exact."""
    a = Allocator(RAMP, train_steps=500, seqs_per_step=256)
    b = Allocator(RAMP, train_steps=500, seqs_per_step=256)
    assert [a.counts_for_step(k) for k in range(500)] == \
           [b.counts_for_step(k) for k in range(500)]


def test_resume_recovers_the_same_cursor_by_replay():
    """
    The resume contract: a fresh Allocator asked for the cursor at step 900 must
    match one that walked there step by step. If this breaks, a resumed run reads
    different data with no error — the silent failure mode.
    """
    walked = Allocator(RAMP, train_steps=1000, seqs_per_step=512)
    for k in range(900):
        walked.counts_for_step(k)
    fresh = Allocator(RAMP, train_steps=1000, seqs_per_step=512)
    assert fresh.cursor_before_step(900) == walked.cursor_before_step(900)


def test_ramp_actually_shifts_the_mix_late():
    a = Allocator(RAMP, train_steps=1000, seqs_per_step=512)
    early = sum(a.counts_for_step(k)["register"] for k in range(0, 100))
    late = sum(a.counts_for_step(k)["register"] for k in range(900, 1000))
    assert late > 2 * early


def test_stop_after_does_not_change_the_mix():
    """Progress is keyed to train_steps, mirroring the LR schedule."""
    full = Allocator(RAMP, train_steps=3250, seqs_per_step=512)
    stopped = Allocator(RAMP, train_steps=3250, seqs_per_step=512)
    assert [full.counts_for_step(k) for k in range(50)] == \
           [stopped.counts_for_step(k) for k in range(50)]


def test_zero_weight_slice_is_never_drawn():
    """A 0% general-text slice must contribute nothing, not one stray sequence."""
    sched = MixSchedule.constant(
        {"backbone": 0.9, "register": 0.1, "general": 0.0})
    a = Allocator(sched, train_steps=500, seqs_per_step=512)
    assert a.cursor_before_step(500)["general"] == 0


def test_allocator_rejects_bad_sizing():
    with pytest.raises(ValueError):
        Allocator(FLAT, train_steps=0, seqs_per_step=512)
    with pytest.raises(ValueError):
        Allocator(FLAT, train_steps=10, seqs_per_step=0)


# --------------------------------------------------------------------------
# Interleaving and reporting
# --------------------------------------------------------------------------

def test_interleave_preserves_counts():
    counts = {"backbone": 470, "register": 42}
    got = interleave(counts, ("backbone", "register"))
    assert len(got) == 512
    assert got.count("register") == 42
    assert got.count("backbone") == 470


def test_interleave_spreads_rather_than_clumping():
    """Every microbatch should see roughly the step's mixture."""
    counts = {"backbone": 448, "register": 64}   # 12.5%
    labels = interleave(counts, ("backbone", "register"))
    chunks = [labels[i:i + 64] for i in range(0, 512, 64)]
    per_chunk = [c.count("register") for c in chunks]
    # perfectly spread would be 8 per chunk of 64; allow slack, forbid clumping
    assert max(per_chunk) - min(per_chunk) <= 2
    assert all(p > 0 for p in per_chunk)


def test_interleave_handles_single_slice():
    assert interleave({"backbone": 4}, ("backbone",)) == ["backbone"] * 4


def test_epochs_per_slice_surfaces_repetition():
    """Upweighting a small pool buys repetition; the number must be visible."""
    got = epochs_per_slice(
        cursor={"backbone": 3_000_000, "register": 1_000_000},
        seq_len=1024,
        pool_tokens={"backbone": 422_000_000, "register": 40_000_000},
    )
    assert got["backbone"] == pytest.approx(3_000_000 * 1024 / 422_000_000)
    assert got["register"] > 20          # ~25 epochs — far past the ~4-epoch point
    assert got["backbone"] < 8


def test_epochs_per_slice_flags_an_empty_pool():
    got = epochs_per_slice({"general": 10}, 1024, {"general": 0})
    assert got["general"] == float("inf")
