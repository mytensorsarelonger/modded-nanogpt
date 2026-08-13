"""
Slice mixing: how many sequences each slice contributes to each training step.

PLAN.md §5.1.1. This module is pure arithmetic — no torch, no I/O — because the
mixing *policy* is the part that has to be provably correct and reproducible. The
dataloader that consumes it lives in train_baseline.py.

Two design decisions worth stating, both chosen for the resume story:

**Schedules, not ratios.** §4.3 wants register material upweighted during cooldown,
and Biderman et al. 2026 note later data has larger influence on final behaviour.
"15% register" and "8% rising to 25% through cooldown" are different interventions.
A `MixSchedule` is piecewise-linear keyframes over training progress, so a constant
ratio is just a single keyframe.

**No RNG.** A random draw per sequence would need its generator state checkpointed
and restored to make a resumed run read the same data. Instead allocation is a
deterministic function of the step index, so a resumed run recovers its exact
position by replaying scalar arithmetic — which composes with the existing
`start_batch` fast-forward instead of fighting it. Replay is O(steps) over a few
thousand steps of float math: free.

Driftless by construction: per-step integer counts come from a fractional carry per
slice, so the long-run realised fraction equals the target rather than drifting from
repeated rounding in the same direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MixSchedule:
    """
    Target slice weights as a function of training progress in [0, 1].

    `keyframes` is [(progress, {slice: weight}), ...] with ascending progress.
    Weights are renormalised to sum to 1, so they can be written as percentages,
    ratios, or anything proportional. Between keyframes weights interpolate
    linearly; outside the first/last they are held flat.

        # constant 8.7% register — the corpus as it stands
        MixSchedule([(0.0, {"backbone": 0.913, "register": 0.087})])

        # §4.3: flat, then ramp register through the cooldown 30%
        MixSchedule([(0.0, {"backbone": 0.92, "register": 0.08}),
                     (0.7, {"backbone": 0.92, "register": 0.08}),
                     (1.0, {"backbone": 0.75, "register": 0.25})])
    """

    keyframes: tuple[tuple[float, dict[str, float]], ...]

    def __post_init__(self):
        if not self.keyframes:
            raise ValueError("MixSchedule needs at least one keyframe")
        progs = [p for p, _ in self.keyframes]
        if any(p < 0.0 or p > 1.0 for p in progs):
            raise ValueError(f"keyframe progress must lie in [0, 1], got {progs}")
        if any(b <= a for a, b in zip(progs, progs[1:])):
            raise ValueError(f"keyframe progress must strictly ascend, got {progs}")
        names = {n for _, w in self.keyframes for n in w}
        for p, w in self.keyframes:
            if any(v < 0 for v in w.values()):
                raise ValueError(f"negative weight at progress {p}: {w}")
            if sum(w.values()) <= 0:
                raise ValueError(f"weights sum to zero at progress {p}")
            missing = names - set(w)
            if missing:
                # Silent zero-fill would make a typo look like a deliberate
                # exclusion, and the ramp would then quietly interpolate to zero.
                raise ValueError(
                    f"keyframe at progress {p} omits slice(s) {sorted(missing)}; "
                    f"list every slice in every keyframe, using 0.0 to exclude one"
                )

    @property
    def slices(self) -> tuple[str, ...]:
        """Slice names in a stable order — allocation tie-breaks depend on it."""
        return tuple(sorted({n for _, w in self.keyframes for n in w}))

    @classmethod
    def constant(cls, weights: dict[str, float]) -> "MixSchedule":
        return cls(keyframes=((0.0, dict(weights)),))

    def weights_at(self, progress: float) -> dict[str, float]:
        """Normalised weights at a given training progress."""
        progress = min(max(progress, 0.0), 1.0)
        kf = self.keyframes
        if progress <= kf[0][0]:
            raw = dict(kf[0][1])
        elif progress >= kf[-1][0]:
            raw = dict(kf[-1][1])
        else:
            for (p0, w0), (p1, w1) in zip(kf, kf[1:]):
                if p0 <= progress <= p1:
                    t = 0.0 if p1 == p0 else (progress - p0) / (p1 - p0)
                    raw = {n: w0[n] + t * (w1[n] - w0[n]) for n in w0}
                    break
        total = sum(raw.values())
        return {n: v / total for n, v in raw.items()}


def allocate_step(weights: dict[str, float], seqs_per_step: int,
                  carry: dict[str, float], order: Sequence[str]) -> dict[str, int]:
    """
    Split one step's sequences across slices, consuming and updating `carry`.

    Largest-remainder within the step, with the remainder carried into the next
    step. That combination is what makes the realised fraction converge on the
    target instead of drifting: plain rounding of `w * N` every step biases the
    same direction forever when `w * N` sits just below .5.

    Mutates `carry` in place. Returns counts summing exactly to `seqs_per_step`.
    """
    desired = {n: weights.get(n, 0.0) * seqs_per_step + carry.get(n, 0.0)
               for n in order}
    counts = {n: int(desired[n] // 1) for n in order}
    counts = {n: max(0, c) for n, c in counts.items()}
    short = seqs_per_step - sum(counts.values())
    if short > 0:
        # Hand the leftovers to the largest fractional parts. Ties break on the
        # fixed slice order so the result cannot depend on dict iteration order.
        ranked = sorted(order, key=lambda n: (-(desired[n] - counts[n]), n))
        for n in ranked[:short]:
            counts[n] += 1
    elif short < 0:
        ranked = sorted(order, key=lambda n: (desired[n] - counts[n], n))
        for n in ranked[: -short]:
            if counts[n] > 0:
                counts[n] -= 1
    for n in order:
        carry[n] = desired[n] - counts[n]
    return counts


class Allocator:
    """
    Deterministic per-step slice allocation, replayable from step 0.

    `counts_for_step(k)` is what step k draws; `cursor_before_step(k)` is how many
    sequences each slice has yielded before step k, which is exactly what a
    resuming run needs to position its per-slice cursors.
    """

    def __init__(self, schedule: MixSchedule, train_steps: int, seqs_per_step: int):
        if train_steps <= 0:
            raise ValueError("train_steps must be positive")
        if seqs_per_step <= 0:
            raise ValueError("seqs_per_step must be positive")
        self.schedule = schedule
        self.train_steps = train_steps
        self.seqs_per_step = seqs_per_step
        self.order = schedule.slices
        self._counts: list[dict[str, int]] = []
        self._cursor: list[dict[str, int]] = [{n: 0 for n in self.order}]
        self._carry: dict[str, float] = {n: 0.0 for n in self.order}

    def _extend_to(self, step: int) -> None:
        while len(self._counts) <= step:
            k = len(self._counts)
            # Progress uses train_steps, NOT the early-exit step, so STOP_AFTER
            # cannot change the mix a given step sees. Same reasoning as the LR
            # schedule: train_steps defines the trajectory's shape.
            w = self.schedule.weights_at(k / self.train_steps)
            c = allocate_step(w, self.seqs_per_step, self._carry, self.order)
            self._counts.append(c)
            prev = self._cursor[-1]
            self._cursor.append({n: prev[n] + c[n] for n in self.order})

    def counts_for_step(self, step: int) -> dict[str, int]:
        self._extend_to(step)
        return dict(self._counts[step])

    def cursor_before_step(self, step: int) -> dict[str, int]:
        self._extend_to(max(step - 1, 0))
        return dict(self._cursor[step])

    def realised_fractions(self, through_step: int) -> dict[str, float]:
        """Actual fraction each slice supplied through `through_step`."""
        cur = self.cursor_before_step(through_step)
        total = sum(cur.values())
        if total == 0:
            return {n: 0.0 for n in self.order}
        return {n: cur[n] / total for n in self.order}


def interleave(counts: dict[str, int], order: Sequence[str]) -> list[str]:
    """
    Spread a step's slice labels across its sequence positions.

    Placement does not affect the summed gradient, but microbatch accumulation
    means a contiguous block would make some microbatches 100% one slice. Spread
    them so every microbatch sees roughly the step's mixture — cheaper to reason
    about, and it keeps per-microbatch activation statistics comparable.

    Bresenham-style: deterministic, no RNG, low discrepancy.
    """
    total = sum(counts.values())
    out: list[str] = []
    acc = {n: 0.0 for n in order}
    remaining = dict(counts)
    for _ in range(total):
        best, best_score = None, None
        for n in order:
            if remaining[n] <= 0:
                continue
            acc[n] += counts[n] / total
            score = (-acc[n], n)
            if best_score is None or score < best_score:
                best, best_score = n, score
        for n in order:
            if remaining[n] > 0 and n != best:
                acc[n] -= counts[n] / total
        out.append(best)
        remaining[best] -= 1
        acc[best] -= 1.0
    return out


def epochs_per_slice(cursor: dict[str, int], seq_len: int,
                     pool_tokens: dict[str, int]) -> dict[str, float]:
    """
    How many times over each slice's pool has been read.

    Worth surfacing rather than hiding: upweighting a small slice buys
    *repetition*, not diversity. Taking register from 8.7% to 25% out of a 40 M
    token pool means many epochs on the same 371 books while the backbone is still
    on its third — see PLAN.md §5.6 on the ~4-epoch point. Past some weight the
    answer is more register tokens, not a bigger multiplier.
    """
    out = {}
    for n, seqs in cursor.items():
        pool = pool_tokens.get(n, 0)
        out[n] = (seqs * seq_len / pool) if pool > 0 else float("inf")
    return out


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------
#
# Referred to by name from the command line so a run's mix is recorded as a word
# in the registry rather than an inline blob nobody can compare across runs.
#
# NOTE on §4.2's A/B/C: the plan describes them as backbone(classical+general)
# vs register(+scripture). The corpus currently has no general-text and no
# scripture slice, so these are register-vs-backbone only. Running them today
# tests the register ratio, NOT the mix §4.2 actually specifies — and `mix-c` in
# particular reads a 40 M-token register pool at 60%, which is repetition, not
# diversity. Check the epochs_per_slice line before believing any of them.
PRESETS: dict[str, MixSchedule] = {
    # The corpus as it stands. Useful as the loader's own control: it should
    # reproduce the natural mix, so any difference from the Milestone 1 baseline
    # is attributable to the loader rather than to the mix.
    "corpus": MixSchedule.constant({"backbone": 0.913, "register": 0.087}),

    # §4.3: hold, then lift register through the cooldown 30%.
    "cooldown-ramp": MixSchedule((
        (0.0, {"backbone": 0.913, "register": 0.087}),
        (0.7, {"backbone": 0.913, "register": 0.087}),
        (1.0, {"backbone": 0.75, "register": 0.25}),
    )),

    # §4.2 mix ablations, static.
    "mix-a": MixSchedule.constant({"backbone": 0.80, "register": 0.20}),
    "mix-b": MixSchedule.constant({"backbone": 0.60, "register": 0.40}),
    "mix-c": MixSchedule.constant({"backbone": 0.40, "register": 0.60}),
}


def parse_schedule(spec: str) -> MixSchedule:
    """
    Resolve a `--mix` value: a preset name, or inline JSON for one-offs.

    JSON forms:
        {"backbone": 0.9, "register": 0.1}                       # constant
        [[0.0, {"backbone": 0.9, "register": 0.1}], [1.0, {...}]]  # keyframes
    """
    spec = spec.strip()
    if spec in PRESETS:
        return PRESETS[spec]
    if not spec.startswith(("{", "[")):
        raise ValueError(
            f"unknown mix {spec!r}; known presets: {', '.join(sorted(PRESETS))}, "
            f"or pass inline JSON"
        )
    import json
    obj = json.loads(spec)
    if isinstance(obj, dict):
        return MixSchedule.constant({str(k): float(v) for k, v in obj.items()})
    return MixSchedule(tuple(
        (float(p), {str(k): float(v) for k, v in w.items()}) for p, w in obj))
