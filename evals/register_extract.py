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

K-B/03 (steered re-scoring and the OTM sweep) extends
modal_app.register_extract's checkpoint loop; capture_block_outputs is the
seam — a steering hook on blocks[ℓ] composes with the same capture path.
"""

from __future__ import annotations

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
