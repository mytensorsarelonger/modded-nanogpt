"""
evals/register_extract.py — activation extraction for the steering
pre-registration (k3-lab paper/steering/prereg-R1-R2.md, §3.1 artifacts 1
and 3).

The §7.2 harness (evals/exemplar.py) scores *behavior* — perplexity and
loss gaps. The steering work needs the *representations* behind those
scores, from the same checkpoints, under the same EOT-anchored windows:

1. Per-context layer means h̄_ℓ(x): for each of the eval set's 7 texts,
   block ℓ's output at the passage-token positions of the window, averaged
   over positions. modal_app.register_extract appends one row per
   (run, step) to runs/<run_id>/register_acts.jsonl; the k3-lab dispatcher
   recomputes v_ℓ, u_ℓ, ρ_ℓ and the E6 controls locally from those rows
   (pre-registration rule 10: this repo ships forward-pass artifacts only).

2. θ_A covariance: the centered residual-stream covariance over 32 fixed
   off-target corpus windows (lexicographically-first gutenberg shard,
   first 32×1024 tokens — deterministic, no RNG), per layer, denominator
   N−1, accumulated in float64 via the TWO-PASS centered form. The one-pass
   E[xxᵀ] − μμᵀ form is forbidden here on purpose: its catastrophic
   cancellation sits exactly in the tail the E6 random-vector gate reads
   (prereg §2.3 item 5). Saved as register_covariance_<step>.npz,
   loadable with np.load(path, allow_pickle=False).

Layer convention (pre-registered, prereg §2.2): h_ℓ = the output of
baseline_model.GPT.blocks[ℓ], ℓ ∈ 0..num_layers−1. The EOT anchor position
(0) is excluded from every mean and from the covariance (first-token
outliers; prereg §2.1 / §2.3 item 5).

steering direction v_ℓ (prereg §2.2, Δh = αd with d = v̂_ℓ, α = ‖v_ℓ‖).

K-B/03 (steered re-scoring, OTM and continuations — prereg §3.1 item 2,
§2.2, §2.4): modal_app.register_extract's steering phase loads each θ_B
checkpoint once, computes v_ℓ per layer from the SAME layer_means values
written to register_acts.jsonl (bit-exact: float64 numpy on the identical
python lists, so the dispatcher's local recomputation is bit-identical),
and writes runs/<θ_B run>/steer_rescore.json (72 rows per regime: 12
layers × {v, neg_v, zero, rand 1..3}) plus steer_continuations.jsonl (72
rows per regime: 12 layers × {v, neg_v, zero} × 2 pair targets). The
steering hook is capture_block_outputs' seam: a forward hook on
blocks[ℓ] whose output becomes output + delta — E5's constant gate, every
position, no conditioning. v_ℓ, u_ℓ, ρ_ℓ, ℓ* and the verdict itself are
the k3-lab dispatcher's job (prereg rule 10); this repo ships forward
passes only.

Two deliberate paths, both recorded as findings, both left as they are:
- evaluate()/score_passage applies the softcap a SECOND time after
  forward_logits already caps once (exemplar.py:194-195). Δḡ_steer must
  be computed on the same path as the existing register_eval rows, so the
  steered re-score inherits it (a recorded finding for a later train).
- OTM uses forward_logits' single-capped logits — the model's own
  distribution — NOT the harness's second cap; the prereg §2.4 KL is a
  property of the steered model, not of the scoring harness.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from evals.exemplar import (
    EOT_TOKEN,
    Passage,
    _load_shard_tokens,
    load_passages,
    split_pairs,
)

LAYER_CONVENTION = "h_l = output of block l, l in 0..11"

# OTM constants, prereg §2.4 / §3.1 item 2.
OTM_ALPHAS = (0.5, 1.0, 2.0)
OTM_KL_DIRECTION = "unsteered||steered"


@contextlib.contextmanager
def steer_hook(model, layer: int, delta: Tensor):
    """Steer block `layer`: output + delta.to(out.dtype) at EVERY position.

    E5's constant gate (prereg §2.2): one forward hook on model.blocks[layer]
    that adds `delta` at every position — no position, token or gate
    conditioning. The hook is ALWAYS removed on exit: a leaked steering
    hook would silently steer every later forward pass (score, covariance,
    continuation). delta may be float32/float64; it is cast to the block's
    bf16 output dtype inside the hook, exactly like the capture path.
    """
    block = model.blocks[layer]
    handle = block.register_forward_hook(
        lambda _module, _inputs, out: out + delta.to(out.dtype)
    )
    try:
        yield handle
    finally:
        handle.remove()


def rand_direction(seed: int, dim: int = 768) -> np.ndarray:
    """Unit-norm random direction for the norm-matched controls (prereg §2.2).

    seed = 20261009 + 1000·j + ℓ (j ∈ {1,2,3}, ℓ = layer). Deterministic
    (default_rng), unit norm, float64. The caller scales by ‖v_ℓ‖ so the
    control is norm-matched to v_ℓ exactly.
    """
    r = np.random.default_rng(seed).standard_normal(dim)
    return (r / np.linalg.norm(r)).astype(np.float64)


def otm_values(
    model, windows: Tensor, layer: int, v: np.ndarray, alphas, device,
    batch: int = 4,
) -> list[float]:
    """OTM(α) per prereg §2.4 (equations.md:200-202), one value per alpha.

    delta = α·v (v = the FULL v_ℓ, not the unit direction). For each window:
    logp_0 = log_softmax(forward_logits(x), -1) unhooked; logp_α the same
    under the hook; KL_t = Σ_vocab exp(logp_0)·(logp_0 − logp_α) at each
    position t; positions 1..W−1 kept (position 0 dropped, as for Σ);
    OTM(α) = mean over kept positions, then over windows. forward_logits'
    single-capped logits — the model's own distribution — NOT the harness's
    second softcap. B×1024×50304 float32 is ~206 MB per window, so batch
    ≤ 4 windows per forward (default 4).
    """
    windows_dev = windows.to(device)
    n_windows, window = windows_dev.shape
    v_t = torch.tensor(np.asarray(v, dtype=np.float64), dtype=torch.float32,
                        device=device)
    alphas = [float(a) for a in alphas]
    totals = [0.0] * len(alphas)
    counts = [0] * len(alphas)
    # batches outer, alphas inner: the unsteered logp_0 pass runs once per
    # chunk and is reused by every α (α = 0 skips forward passes entirely —
    # OTM(0) = 0 is exact by construction, prereg §2.4).
    for start in range(0, n_windows, batch):
        chunk = windows_dev[start : start + batch]
        with torch.no_grad():
            logp0 = torch.log_softmax(model.forward_logits(chunk), -1)
            for i, alpha in enumerate(alphas):
                if alpha == 0:
                    continue
                delta = (alpha * v_t).to(torch.bfloat16)
                with steer_hook(model, layer, delta):
                    logpa = torch.log_softmax(model.forward_logits(chunk), -1)
                # KL_t = Σ_vocab exp(logp_0)·(logp_0 − logp_α); keep
                # positions 1..W−1 (position 0 dropped, as for Σ)
                kl = (logp0.exp() * (logp0 - logpa)).sum(-1)[:, 1:]
                totals[i] += float(kl.sum())
                counts[i] += kl.numel()
    values = []
    for i, alpha in enumerate(alphas):
        if alpha == 0:
            values.append(0.0)
        else:
            if counts[i] == 0:
                raise ValueError("otm_values: no kept positions")
            values.append(totals[i] / counts[i])
    return values


def greedy_continuation(model, prefix_ids: list[int], n_new: int,
                        device) -> list[int]:
    """`n_new` greedy (temperature 0) steps from prefix_ids (no KV cache).

    Each step: argmax(forward_logits(seq)[0, -1]) on the full sequence;
    the sequence is re-forwarded whole every step — deterministic and
    steering-hook compatible (the caller installs/removes the hook). The
    returned ids are ints, plain python, appendable to the prefix.
    """
    seq = torch.tensor([list(prefix_ids)], dtype=torch.int64, device=device)
    new_ids: list[int] = []
    with torch.no_grad():
        for _ in range(n_new):
            logits = model.forward_logits(seq)[0, -1]
            next_id = int(torch.argmax(logits).item())
            new_ids.append(next_id)
            seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.int64,
                                                device=device)], dim=1)
    return new_ids


def direction_delta(direction: str, v: np.ndarray, v_norm: float) -> np.ndarray:
    """The delta for one contract direction, all against the same v_ℓ (K-B/03).

    "v" → +v_ℓ, "neg_v" → −v_ℓ, "zero" → zeros (the hook is still installed
    by the caller), "rand:<seed>" → rand_direction(seed)·‖v_ℓ‖ (norm-matched,
    prereg §2.2). float64 in, float64 out; the hook casts to the block dtype.
    """
    v = np.asarray(v, dtype=np.float64)
    if direction == "v":
        return v
    if direction == "neg_v":
        return -v
    if direction == "zero":
        return np.zeros_like(v)
    if direction.startswith("rand:"):
        seed = int(direction.split(":", 1)[1])
        return rand_direction(seed, dim=v.shape[0]) * float(v_norm)
    raise ValueError(f"unknown direction {direction!r}")


def steer_row(
    model, enc, *,
    regime: str,
    run_id: str,
    step: int,
    theta_A: dict,
    layer: int,
    direction: str,
    alpha: float,
    v: np.ndarray,
    v_norm: float,
    exemplar_file,
    craft_file,
    seq_len: int,
    device,
    windows: Tensor | None = None,
    window_source: str | None = None,
    otm_batch: int = 4,
    otm: list[float] | None = None,
) -> dict:
    """One steer_rescore.json row (prereg §3.1 item 2). Forward passes only.

    Direction deltas (all against the same v_ℓ): "v" → +v (alpha 1.0),
    "neg_v" → −v (alpha 1.0), "zero" → zeros (alpha 0.0; hook STILL
    installed), "rand:<seed>" → rand_direction(seed)·‖v‖ (alpha 1.0,
    norm-matched). Each row's evaluate() runs under the hook on the same
    second-softcap path as the existing register_eval rows. Direction "v"
    additionally carries the OTM(α) sweep: computed here from `windows`
    unless the caller passes precomputed `otm` values (the modal_app phase
    does that, so it can time the OTM passes separately from the re-scores).
    Key set exactly as the k3-lab dispatcher's validate_steer expects
    (check_kb.py): regime, run_id, step, theta_A, layer, direction, alpha,
    v_norm, mean_loss_gap_steered, contrast_gaps, otm (null off "v").
    """
    from evals.exemplar import evaluate

    delta = torch.tensor(direction_delta(direction, v, v_norm),
                         dtype=torch.float32, device=device)
    with steer_hook(model, layer, delta):
        result = evaluate(model, enc, exemplar_file=exemplar_file,
                          craft_file=craft_file, seq_len=seq_len,
                          device=device)
    otm_dict = None
    if direction == "v":
        if otm is None:
            if windows is None or window_source is None:
                raise ValueError("direction 'v' requires windows/window_source")
            otm = otm_values(model, windows, layer, v, OTM_ALPHAS, device,
                             batch=otm_batch)
        otm_dict = {
            "alphas": list(OTM_ALPHAS),
            "values": [float(x) for x in otm],
            "kl_direction": OTM_KL_DIRECTION,
            "off_target_windows": int(windows.shape[0]),
            "positions_per_window": int(windows.shape[1] - 1),
            "window_source": window_source,
        }
    return {
        "regime": regime,
        "run_id": run_id,
        "step": step,
        "theta_A": {"run_id": theta_A["run_id"], "step": theta_A["step"]},
        "layer": int(layer),
        "direction": direction,
        "alpha": float(alpha),
        "v_norm": float(v_norm),
        "mean_loss_gap_steered": float(result["mean_loss_gap"]),
        "contrast_gaps": result["contrast_gaps"],
        "otm": otm_dict,
    }


def capture_block_outputs(model, inputs: Tensor) -> list[Tensor]:
    """h_ℓ for every ℓ in one forward pass.

    Installs a forward hook on each model.blocks[ℓ], runs
    model.forward_logits(inputs) (the no-grad eval path, and the same path
    a K-B/03 steering hook will act on), and returns one detached (B, T, D)
    tensor per block in block order. Every hook is removed in a finally —
    a leaked hook would silently double-capture on the next call.
    """
    outputs: list[Tensor] = []
    handles = [
        block.register_forward_hook(
            lambda _module, _inputs, out: outputs.append(out.detach())
        )
        for block in model.blocks
    ]
    try:
        model.forward_logits(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def context_layer_means(
    model, enc, passage: Passage, seq_len: int, device
) -> tuple[int, Tensor]:
    """(T_p, float32 (num_layers, D)) for one passage.

    Window = [EOT] + ids (length T_p + 1), the same EOT-anchored
    single-window shape the §7.2 harness scores. h̄_ℓ = block ℓ's output at
    positions 1..T_p — the passage tokens, with the EOT anchor at position
    0 excluded (prereg §2.1) — averaged over those positions in float32
    (the blocks run bf16; callers store float64).
    """
    ids = enc.encode(passage.text, allowed_special=set())
    n = len(ids)
    if n == 0:
        raise ValueError(f"passage {passage.id!r} tokenized to zero tokens")
    if n + 1 > seq_len:
        raise ValueError(
            f"passage {passage.id!r}: {n} tokens + 1 EOT anchor > seq_len "
            f"{seq_len}; the steering context set C requires single-window "
            "passages (prereg §2.1)"
        )
    stream = torch.tensor([EOT_TOKEN] + ids, dtype=torch.int64, device=device)
    with torch.no_grad():
        outputs = capture_block_outputs(model, stream.unsqueeze(0))
    means = torch.stack([
        out[0, 1 : n + 1].to(torch.float32).mean(dim=0) for out in outputs
    ])
    return n, means


def acts_contexts(
    model, enc, exemplar_file, craft_file, seq_len: int, device
) -> list[dict]:
    """The context objects of prereg §2.1 / §3.1 item 1, in order.

    For each contrast pair (split_pairs order): target, then wrong; then
    the craft passages in file order. Roles: "target" | "wrong" | "craft".
    The exemplar file must hold pairs only — C is exactly the pairs plus
    the craft passages, and a single would add a role the pre-registration
    does not define.
    """
    singles, pairs = split_pairs(load_passages(Path(exemplar_file)))
    if singles:
        raise ValueError(
            f"{exemplar_file}: {len(singles)} unpaired passage(s) "
            f"({', '.join(p.id for p in singles)}); the steering context "
            "set C is exactly the contrast pairs plus the craft passages "
            "(prereg §2.1)"
        )
    craft = load_passages(Path(craft_file))
    ordered = [
        (role, passage)
        for target, wrong in pairs
        for role, passage in (("target", target), ("wrong", wrong))
    ]
    ordered += [("craft", passage) for passage in craft]

    contexts: list[dict] = []
    for role, passage in ordered:
        tokens, means = context_layer_means(model, enc, passage, seq_len, device)
        contexts.append({
            "id": passage.id,
            "role": role,
            "tokens": tokens,
            "layer_means": {
                str(layer): means[layer].double().tolist()
                for layer in range(means.size(0))
            },
        })
    return contexts


def offtarget_windows(
    shard_glob: str, n_windows: int = 32, window: int = 1024, root=None
) -> tuple[Tensor, str]:
    """The fixed off-target windows of prereg §2.4.

    The first n_windows × window tokens of the lexicographically-first
    matching shard, as an int64 (n_windows, window) tensor — 32 consecutive
    non-overlapping windows, deterministic, no RNG, no EOT added. `root`
    defaults to cwd (the repo root in the eval container, where the shards
    Volume is mounted at data/shards) — the same contract as the
    contamination guard's.
    """
    base = Path(root) if root is not None else Path.cwd()
    shard_files = sorted(base.glob(shard_glob))
    if not shard_files:
        raise FileNotFoundError(
            f"no shards match {shard_glob!r} (cwd={Path.cwd()})"
        )
    shard = shard_files[0]
    need = n_windows * window
    tokens = _load_shard_tokens(shard)
    if len(tokens) < need:
        raise ValueError(
            f"{shard.name}: {len(tokens)} tokens < {n_windows}x{window} = {need}"
        )
    windows = tokens[:need].view(n_windows, window)
    source = f"{shard.name}:tokens[0:{need}]:{n_windows}x{window}"
    return windows, source


def layer_covariance(
    model, windows: Tensor, device, batch: int = 8, exclude_position: int | None = 0
) -> dict:
    """Centered per-layer covariance of h_ℓ over the windows' rows.

    Every position of each window except `exclude_position` contributes as
    an individual row — no per-window averaging (prereg §2.3 item 5):
    N = n_windows × (window − 1) rows per layer. Two passes in float64:
    sums → mean, then centered products, denominator N − 1. Returns
    {"covariance": (L, D, D) float64, "mean": (L, D) float64, "count": N}.
    """
    n_windows, window = windows.shape
    if exclude_position is None:
        keep = torch.arange(window)
    else:
        keep = torch.tensor([t for t in range(window) if t != exclude_position])
    n_rows = n_windows * keep.numel()
    if n_rows < 2:
        raise ValueError(f"covariance needs >= 2 rows, got {n_rows}")

    def _rows(outputs: list[Tensor], layer: int) -> Tensor:
        out = outputs[layer]
        rows = out.index_select(1, keep.to(out.device))
        return rows.reshape(-1, rows.size(-1)).to(torch.float64)

    # pass 1 — means
    sums: list[Tensor] | None = None
    for start in range(0, n_windows, batch):
        chunk = windows[start : start + batch].to(device)
        with torch.no_grad():
            outputs = capture_block_outputs(model, chunk)
        if sums is None:
            sums = [
                torch.zeros(out.size(-1), dtype=torch.float64, device=out.device)
                for out in outputs
            ]
        for layer in range(len(outputs)):
            sums[layer] += _rows(outputs, layer).sum(dim=0)
    assert sums is not None
    means = [s / n_rows for s in sums]

    # pass 2 — centered products (NOT E[xxᵀ] − μμᵀ; see module docstring)
    covs = [
        torch.zeros((m.numel(), m.numel()), dtype=torch.float64, device=m.device)
        for m in means
    ]
    for start in range(0, n_windows, batch):
        chunk = windows[start : start + batch].to(device)
        with torch.no_grad():
            outputs = capture_block_outputs(model, chunk)
        for layer in range(len(outputs)):
            centered = _rows(outputs, layer) - means[layer]
            covs[layer] += centered.T @ centered

    denom = n_rows - 1   # sample covariance, prereg §3.1 item 3
    return {
        "covariance": (torch.stack(covs) / denom).cpu().numpy(),
        "mean": torch.stack(means).cpu().numpy(),
        "count": n_rows,
    }


def save_covariance(
    path,
    cov: dict,
    *,
    step: int,
    window_tokens: int,
    excluded_position: int,
    window_source: str,
) -> None:
    """Write one register_covariance_<step>.npz per prereg §3.1 item 3.

    Exact keys, float64 arrays, int64 0-d scalars, 0-d unicode
    window_source. Uncompressed np.savez, no pickle — the file must load
    with np.load(path, allow_pickle=False).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        covariance=np.asarray(cov["covariance"], dtype=np.float64),
        mean=np.asarray(cov["mean"], dtype=np.float64),
        count=np.int64(cov["count"]),
        step=np.int64(step),
        window_tokens=np.int64(window_tokens),
        excluded_position=np.int64(excluded_position),
        window_source=np.str_(window_source),
    )
