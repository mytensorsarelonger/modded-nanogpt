"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import torch
import torch._dynamo
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config, config_hash, manifest_hash, validate_against_shards
import probes
from training_state import (
    checkpoint_sidecar_path,
    completion_metadata,
    copy_sample_log_through_step,
    distributed_batch_granule,
    effective_training_overrides,
    latest_checkpoint,
    validate_loader_step,
    validate_resume_metadata,
)


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        # pin_memory requires CUDA; harmless to skip on CPU.
        tokens = torch.empty(num_tokens, dtype=torch.uint16,
                             pin_memory=torch.cuda.is_available())
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024,
                               start_batch: int = 0, state: dict | None = None):
    """
    Yields (inputs, targets). If `state` is a dict, the generator writes its
    position into it every step as {"file_idx", "pos", "batches"} so a checkpoint
    can record exactly where in the corpus it was.

    `start_batch` fast-forwards that many batches on resume. This matters: at 3.7
    epochs a resume that restarts the stream from shard 0 re-shows the model the
    beginning of the corpus and silently changes the data distribution it sees,
    which would make a resumed curve non-comparable to an uninterrupted one.
    Fast-forwarding only advances offsets and loads shards -- it does no compute.
    """
    files = sorted(Path.cwd().glob(filename_pattern))
    assert len(files) > 0, f"No shards matched {filename_pattern!r} (cwd={Path.cwd()})"
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    # view(-1, seq_len) below requires the per-rank batch to tile exactly.
    assert local_batch_size % seq_len == 0, (
        f"local_batch_size {local_batch_size} must be a multiple of seq_len {seq_len}"
    )
    # Explicit shard index rather than an iterator: cycling past the end and
    # fast-forwarding on resume are both one line with an index, and were both
    # error-prone with a bare iterator.
    file_idx, batches = 0, 0
    tokens, pos = _load_data_shard(files[0]), 0

    def advance():
        """Move to the next shard, wrapping. Returns nothing; mutates closure."""
        nonlocal file_idx, tokens, pos
        # The corpus is smaller than a full run's token budget (see
        # epochs_over_corpus in the run record), so wrapping is expected and
        # deliberate rather than an error.
        file_idx = (file_idx + 1) % len(files)
        tokens, pos = _load_data_shard(files[file_idx]), 0

    # Replay position arithmetic on resume. No GPU work, no yields; only shard
    # loads, so it costs one read per shard boundary crossed.
    for _ in range(start_batch):
        if pos + batch_size + 1 >= len(tokens):
            advance()
        pos += batch_size
        batches += 1

    while True:
        if pos + batch_size + 1 >= len(tokens):
            advance()
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        pos += batch_size
        batches += 1
        if state is not None:
            state["file_idx"], state["pos"], state["batches"] = file_idx, pos, batches
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128, num_heads=None):
        super().__init__()
        inferred_heads = dim // head_dim
        self.num_heads = inferred_heads if num_heads is None else num_heads
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        assert hdim == dim, (
            f"num_heads*head_dim must equal model_dim ({self.num_heads}*"
            f"{self.head_dim} != {dim})"
        )
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, head_dim=128, num_heads=None):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim=head_dim, num_heads=num_heads)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 head_dim: int = 128, num_heads: int | None = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            Block(model_dim, head_dim=head_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

    @torch.no_grad()
    def forward_logits(self, inputs: Tensor):
        """Forward pass returning logits (for sampling)."""
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return logits


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# MUON_COMPILE=0 leaves this function eager while the model stays compiled.
# Diagnostic for a real, reproduced failure: with the whole thing compiled on
# torch 2.10.0+cu128 / sm89, weights go non-finite on the SECOND optimizer update
# (val is clean after one update, NaN after two) on a cold inductor cache. This
# function mutates `grad` in place via grad.lerp_ and runs Newton-Schulz in bf16,
# which makes it the prime suspect. Compiling the model is worth ~3.4x, so
# excluding only this is far better than COMPILE=0 if it is sufficient.
MUON_COMPILE = os.environ.get("MUON_COMPILE", "1") == "1"


@(torch.compile if MUON_COMPILE else (lambda f: f))
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#                Setup                 #
########################################

# --- Device and mode ---------------------------------------------------------
# One script, no source rewriting. SMOKE=1 shrinks the run; the device is
# detected rather than patched. This replaces smoke_test.py's string-replacement
# approach, which silently rotted whenever a patched line was edited (its
# `val_tokens = 20 * 524288` replace became a no-op and went unnoticed).
SMOKE = os.environ.get("SMOKE", "0") == "1"
CUDA = torch.cuda.is_available()

# torchrun sets these; default them so the script runs standalone too.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

if CUDA:
    device = torch.device("cuda", LOCAL_RANK)
    torch.cuda.set_device(device)
    # NCCL is Linux-only (PLAN.md §2); gloo everywhere keeps one code path.
    dist.init_process_group(backend=config.dist_backend, device_id=device)
else:
    device = torch.device("cpu")
    dist.init_process_group(backend="gloo")

dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

def device_name() -> str:
    return torch.cuda.get_device_name(device) if CUDA else "CPU"

# torch.compile needs Triton on CUDA and MSVC on Windows. Off by default under
# SMOKE, since inductor buys nothing for a ten-step correctness check.
COMPILE = os.environ.get("COMPILE", "0" if SMOKE else "1") == "1"

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a", encoding="utf-8") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {device_name()} with world_size {dist.get_world_size()}")
print0("="*100)

seq_len = config.seq_len
# SMOKE shrinks the batch so a correctness pass fits in CPU RAM. Overridable
# individually so a mid-size debug run on the 4090 needs no code change either.
batch_size = int(os.environ.get("BATCH_SIZE", 64 * 1024 if SMOKE else config.batch_size))
mbs = int(os.environ.get("MBS", 1 if SMOKE and not CUDA else config.mbs))
_granule = distributed_batch_granule(seq_len, mbs, dist.get_world_size())
assert batch_size % _granule == 0, (
    f"batch_size {batch_size} must be a multiple of "
    f"seq_len*mbs*world_size = {_granule}"
)

# Val size is derived from what is actually on disk, never hardcoded. A literal
# here goes stale the moment the corpus changes size, and the failure mode is an
# unhelpful StopIteration from the dataloader rather than a clear message.
# Requirements: <= the tokens the val shards hold (with one spare for the
# input/target offset), and a multiple of seq_len * mbs so that view(-1, seq_len)
# tiles exactly and the microbatch loop divides evenly.
_shard_meta = validate_against_shards()
_available = _shard_meta["total_val_tokens"] - 1
# SMOKE evaluates a token slice, not the whole val set: the point is to prove the
# code path runs, and a full val pass on CPU is thousands of forward passes.
_requested = int(os.environ.get(
    "VAL_TOKENS", 64 * _granule if SMOKE else config.val_tokens))
val_tokens = min(_requested, (_available // _granule) * _granule)
assert val_tokens > 0, (
    f"val shards hold only {_shard_meta['total_val_tokens']} tokens, need at least "
    f"{_granule}. Add more val books or lower mbs/seq_len."
)
if val_tokens < _requested:
    print0(f"NOTE: val_tokens capped at {val_tokens:,} by shard contents "
           f"({_shard_meta['total_val_tokens']:,} available; asked for "
           f"{_requested:,})", console=True)
print0(f"val_tokens={val_tokens:,}  batch_size={batch_size:,}  mbs={mbs}", console=True)

val_inputs, val_targets = next(distributed_data_generator("data/shards/gutenberg_val_*.bin", val_tokens))

# Seed BEFORE constructing the model. Parameter init draws from the global RNG,
# so without this every run starts from different weights: two identical 6-step
# runs were measured 0.0148 apart in val_loss, ~1000x the ~1e-5 discrepancy a
# checkpoint resume introduces. That noise floor would sit underneath every
# Phase 0.5 ablation, so a real architecture effect smaller than it would be
# indistinguishable from the random draw. Override with INIT_SEED to measure the
# noise floor deliberately (same config, different seed).
_init_seed = int(os.environ.get("INIT_SEED", config.init_seed))
torch.manual_seed(_init_seed)
if CUDA:
    torch.cuda.manual_seed_all(_init_seed)
print0(f"init_seed={_init_seed}", console=True)

model = GPT(vocab_size=config.vocab_size, num_layers=config.num_layers,
            model_dim=config.model_dim, head_dim=config.head_dim,
            num_heads=config.num_heads).to(device)
if COMPILE:
    model.compile(dynamic=False)
else:
    # Also neuters the @torch.compile on muon_update, which would otherwise
    # invoke inductor at the first optimizer step.
    torch._dynamo.config.disable = True
    print0("torch.compile disabled (set COMPILE=1 to enable)", console=True)

# Tokenizer for sampling
enc = tiktoken.get_encoding(config.tokenizer_name)


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for _ in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = int(os.environ.get("TRAIN_STEPS", 10 if SMOKE else config.train_steps))
    val_every = int(os.environ.get("VAL_EVERY", 0))
    stop_after = int(os.environ.get("STOP_AFTER", 0))
    sample_every = int(os.environ.get("SAMPLE_EVERY", config.sample_every))
    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY", config.checkpoint_every))
    adamw_fused = os.environ.get("ADAMW_FUSED", "1") == "1"

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    optimizer1 = AdamW([
                        dict(params=[model.embed.weight], lr=config.adamw_lr_embed),
                        dict(params=[model.proj.weight], lr=config.adamw_lr_head),
                        dict(params=[p for p in model.parameters() if p.ndim < 2],
                             lr=config.adamw_lr_scalars)],
                       betas=config.adamw_betas, eps=config.adamw_eps,
                       weight_decay=config.adamw_wd, fused=adamw_fused)
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=config.muon_lr, weight_decay=config.muon_wd)
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta

    # --- Run registry ---
    import hashlib, json, subprocess
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    run_id = str(uuid.uuid4())
    short_tag = f"baseline_{dist.get_world_size()}gpu"
    # GIT_SHA / GIT_DIRTY env override first: a cloud container has no .git, so
    # without this every remote run records git_sha="unknown" and §4.0.1's
    # "which code" column is worthless for exactly the runs that cost money.
    git_sha = os.environ.get("GIT_SHA")
    git_dirty = os.environ.get("GIT_DIRTY")
    git_dirty = None if git_dirty is None else git_dirty not in ("0", "", "false")
    if git_sha is None or git_dirty is None:
        # Narrow except: a bare one also swallows KeyboardInterrupt and SystemExit.
        try:
            git_sha = git_sha or subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            if git_dirty is None:
                git_dirty = subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    stderr=subprocess.DEVNULL).decode().strip() != ""
        except (OSError, subprocess.SubprocessError):
            git_sha = git_sha or "unknown"
            git_dirty = True if git_dirty is None else git_dirty
    # Hash the WHOLE config dataclass, not a hand-listed subset: a subset means
    # two runs differing only in an unlisted field (muon_lr, betas, val_frac)
    # collide on the same config_hash, which is the one thing the hash exists to
    # prevent. Same for the manifest — hash the file, not a summary of it.
    base_cfg_hash = config_hash()[:16]
    effective_overrides = effective_training_overrides(
        batch_size=batch_size,
        mbs=mbs,
        seq_len=seq_len,
        train_steps=train_steps,
        val_tokens=val_tokens,
        sample_every=sample_every,
        checkpoint_every=checkpoint_every,
        val_every=val_every,
        compile_model=COMPILE,
        adamw_fused=adamw_fused,
        muon_compile=MUON_COMPILE,
        init_seed=_init_seed,
        world_size=dist.get_world_size(),
        # A checkpoint resumed under a different framework/CUDA build is not a
        # bit-exact continuation even when every model hyperparameter matches.
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
    )
    cfg_hash = config_hash(effective_overrides)[:16]
    data_hash = (manifest_hash() or "unknown")[:16]
    # The shard manifest pins tokenizer, vocab, eot, val-split method AND the
    # tier-1 filter thresholds, so one hash covers the whole data recipe.
    _sm_path = config.shard_dir / "manifest.json"
    shard_hash = hashlib.sha256(_sm_path.read_bytes()).hexdigest()[:16]

    run_entry = {
        "run_id": run_id,
        "short_tag": short_tag,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "config_hash": cfg_hash,
        "base_config_hash": base_cfg_hash,
        "data_manifest_hash": data_hash,
        "shard_manifest_hash": shard_hash,
        "tokenizer": config.tokenizer_name,
        "vocab_size": config.vocab_size,
        "num_layers": config.num_layers,
        "model_dim": config.model_dim,
        "mbs": mbs,
        "batch_size": batch_size,
        "val_tokens": val_tokens,
        "train_steps": train_steps,
        "train_tokens_available": _shard_meta["total_train_tokens"],
        "planned_epochs_over_corpus": round(
            train_steps * batch_size / max(_shard_meta["total_train_tokens"], 1), 2),
        "hardware": device_name(),
        "world_size": dist.get_world_size(),
        "compile": COMPILE,
        "adamw_fused": adamw_fused,
        "muon_compile": MUON_COMPILE,
        "init_seed": _init_seed,
        "val_every": val_every,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    print0(f"Run ID: {run_id}")
    print0(f"Config hash:        {cfg_hash}")
    print0(f"Data manifest hash: {data_hash}")
    print0(f"Shard recipe hash:  {shard_hash}")
    print0(f"Planned epochs over corpus: {run_entry['planned_epochs_over_corpus']}")

    # --- Optional W&B (a view; runs/index.jsonl stays the source of truth) ----
    # Enabled by WANDB=1. Deliberately soft: an unavailable or misconfigured
    # tracker must never take down a paid training run, so every call is guarded
    # and failure downgrades to a warning and disables further logging.
    # run_entry is passed as the W&B config so a chart maps 1:1 back to an
    # index.jsonl record via config_hash.
    class _Tracker:
        def __init__(self):
            self.run = None
            if os.environ.get("WANDB", "0") != "1" or dist.get_rank() != 0:
                return
            try:
                import wandb
                self.wandb = wandb
                self.run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "k3mini"),
                    name=f"{short_tag}-{run_id[:8]}",
                    id=run_id, resume="allow",   # a resumed run continues one chart
                    config=run_entry,
                    tags=[short_tag, f"cfg:{cfg_hash}", f"data:{data_hash}"],
                )
                print0(f"W&B: logging to {self.run.url}", console=True)
            except Exception as e:
                print0(f"W&B disabled ({type(e).__name__}: {e})", console=True)
                self.run = None

        def log(self, data: dict, step: int):
            if self.run is None:
                return
            try:
                self.wandb.log(data, step=step)
            except Exception as e:
                print0(f"W&B log failed, disabling ({e})", console=True)
                self.run = None

        def finish(self, summary: dict | None = None):
            if self.run is None:
                return
            try:
                if summary:
                    self.run.summary.update(summary)
                self.wandb.finish()
            except Exception:
                pass

    tracker = _Tracker()

    # --- Checkpointing ---
    ckpt_dir = runs_dir / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    CKPT_FORMAT = 3   # format 3 adds rank-local Muon/RNG sidecars

    def save_checkpoint(model, step, optimizers, loader_state):
        """
        Save everything needed to continue the run identically.

        Weights and optimizer state are the obvious part. The rest is what makes a
        resumed curve comparable to an uninterrupted one:
          - `loader_state` so the stream continues where it stopped instead of
            replaying the corpus from shard 0 (at 3.7 epochs that would change the
            data distribution the model sees),
          - RNG state, so post-resume stochasticity continues rather than repeats,
          - the sizing that determined the schedule, so a resume under a different
            batch_size or train_steps is caught rather than silently mis-scheduled.
        Written to a temp file and renamed, so an interrupted write cannot leave a
        truncated checkpoint that fails to load later.
        """
        validate_loader_step(step, loader_state)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # The loader advances globally by batch_size, so every rank must report
        # the same position. Catch a divergent rank before writing a checkpoint
        # set that could never be resumed faithfully.
        loader_states = [None] * world_size
        dist.all_gather_object(loader_states, dict(loader_state))
        if any(rank_state != loader_states[0] for rank_state in loader_states[1:]):
            raise RuntimeError(f"loader state differs across ranks: {loader_states}")

        ckpt_path = ckpt_dir / f"ckpt_{step:05d}.pt"
        rank_path = checkpoint_sidecar_path(ckpt_path, rank)
        rank_tmp_path = rank_path.with_suffix(".pt.tmp")
        rank_payload = {
            "format": CKPT_FORMAT,
            "step": step,
            "rank": rank,
            "world_size": world_size,
            # AdamW state is replicated; Muon momentum is intentionally sharded
            # by parameter owner and therefore must be saved by every rank.
            "sharded_optimizer_states": [optimizers[1].state_dict()],
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if CUDA else None,
        }
        torch.save(rank_payload, rank_tmp_path)
        rank_tmp_path.replace(rank_path)

        # Publish the common file only after every sidecar is durable. Auto
        # discovery ignores orphaned sidecars from interrupted writes.
        dist.barrier()
        if rank == 0:
            tmp_path = ckpt_path.with_suffix(".pt.tmp")
            payload = {
                "format": CKPT_FORMAT,
                "step": step,
                "world_size": world_size,
                "model_state_dict": model.state_dict(),
                "replicated_optimizer_states": [optimizers[0].state_dict()],
                "loader_state": dict(loader_state),
                "sizing": {"batch_size": batch_size, "mbs": mbs,
                           "seq_len": seq_len, "train_steps": train_steps,
                           "val_tokens": val_tokens},
                "config_hash": cfg_hash,
                "data_manifest_hash": data_hash,
                "shard_manifest_hash": shard_hash,
            }
            torch.save(payload, tmp_path)
            tmp_path.replace(ckpt_path)
            print0(f"Saved checkpoint to {ckpt_path} ({world_size} rank state file(s))")
        dist.barrier()

    def load_checkpoint(model, optimizers, path: Path) -> dict:
        """Restore in place and return the metadata needed to continue."""
        ck = torch.load(path, map_location=device, weights_only=True)
        want = {"batch_size": batch_size, "mbs": mbs, "seq_len": seq_len,
                "train_steps": train_steps, "val_tokens": val_tokens}
        checkpoint_format = ck.get("format", 1)
        expected_cfg_hash = base_cfg_hash if checkpoint_format == 2 else cfg_hash
        identities = {"config_hash": expected_cfg_hash,
                      "data_manifest_hash": data_hash,
                      "shard_manifest_hash": shard_hash}
        fmt = validate_resume_metadata(
            ck, current_format=CKPT_FORMAT,
            current_world_size=dist.get_world_size(), sizing=want,
            identities=identities,
        )
        model.load_state_dict(ck["model_state_dict"])

        if fmt == 2:
            # Legacy format 2 is safe only at world_size=1, enforced above.
            if len(ck.get("optimizer_states", [])) != len(optimizers):
                raise ValueError("format-2 checkpoint has incomplete optimizer state")
            for opt, state in zip(optimizers, ck["optimizer_states"]):
                opt.load_state_dict(state)
            cpu_rng_state = ck["cpu_rng_state"]
            cuda_rng_state = ck.get("cuda_rng_state")
        else:
            replicated = ck.get("replicated_optimizer_states", [])
            if len(replicated) != 1:
                raise ValueError("checkpoint has incomplete replicated optimizer state")
            optimizers[0].load_state_dict(replicated[0])

            rank = dist.get_rank()
            rank_path = checkpoint_sidecar_path(path, rank)
            if not rank_path.exists():
                raise FileNotFoundError(
                    f"rank {rank} state file missing for {path}: {rank_path}"
                )
            rank_ck = torch.load(rank_path, map_location=device, weights_only=True)
            expected_rank_meta = {
                "format": CKPT_FORMAT, "step": ck["step"], "rank": rank,
                "world_size": dist.get_world_size(),
            }
            rank_meta = {key: rank_ck.get(key) for key in expected_rank_meta}
            if rank_meta != expected_rank_meta:
                raise ValueError(
                    f"rank state metadata mismatch ({rank_meta} != "
                    f"{expected_rank_meta})"
                )
            sharded = rank_ck.get("sharded_optimizer_states", [])
            if len(sharded) != 1:
                raise ValueError("rank state has incomplete sharded optimizer state")
            optimizers[1].load_state_dict(sharded[0])
            cpu_rng_state = rank_ck["cpu_rng_state"]
            cuda_rng_state = rank_ck.get("cuda_rng_state")

        torch.set_rng_state(cpu_rng_state.cpu().to(torch.uint8))
        if CUDA and cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state.cpu().to(torch.uint8))
        return {"step": ck["step"], "loader_state": ck.get("loader_state") or {}}

    # --- Sampling ---
    sample_log_path = ckpt_dir / "samples.log"

    # vocab_size is padded to a multiple of 128 (50304) but the tokenizer only
    # defines 50257 ids. The model can and does sample into the padding, which
    # tiktoken cannot decode ("Invalid token for decoding: 50274"). Mask the pad
    # region out of the distribution rather than catching the error downstream --
    # sampling a token that means nothing is a bug, not a display problem.
    n_real_tokens = enc.n_vocab

    def _generate(model, enc, prompt: str, temperature: float, max_new_tokens: int,
                  gen: torch.Generator) -> str:
        """Sample one continuation. Returns the new text only, not the prompt."""
        input_ids = torch.tensor(enc.encode(prompt), dtype=torch.int32,
                                 device=device).unsqueeze(0)
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = model.forward_logits(generated[:, -config.seq_len:])
            logits = logits[0, -1] / temperature
            logits[n_real_tokens:] = float("-inf")   # never sample padding
            probs = F.softmax(logits, dim=-1)
            # A diverged model makes probs non-finite, and torch.multinomial then
            # fails with a device-side assert whose traceback points here rather
            # than at the divergence. One cheap check per token converts an opaque
            # CUDA abort -- potentially hours into a paid run -- into a diagnosis.
            if not torch.isfinite(probs).all():
                return ("<NON-FINITE LOGITS: model has diverged; "
                        "sampling aborted at "
                        f"{generated.size(1) - input_ids.size(1)} new tokens>")
            next_token = torch.multinomial(probs, num_samples=1, generator=gen)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        # Decode only the new tokens rather than char-slicing the decoded string by
        # len(prompt): that slice is correct only while the prompt round-trips
        # through BPE exactly, which is not guaranteed.
        return enc.decode(generated[0, input_ids.size(1):].tolist())

    def sample(model, enc, step, include_heldout: bool = False):
        """
        Run the probe suite and append it to samples.log (PLAN.md §7.1).

        Every prompt is drawn at every temperature in probes.TEMPERATURES, and the
        generator is re-seeded to the SAME value per (prompt, temperature) at every
        checkpoint. That makes step 500 and step 2500 differ only by the model,
        which is the entire point of the instrument. It also means drawing samples
        never advances global RNG, so adding or removing the dump cannot perturb
        the training trajectory.
        """
        if dist.get_rank() != 0:
            return
        pairs = probes.routine_prompts()
        if include_heldout:
            pairs = pairs + probes.heldout_prompts()

        model.eval()
        # encoding is mandatory, not cosmetic: a byte-BPE vocab emits arbitrary
        # Unicode, especially from a randomly-initialised model, and the Windows
        # cp1252 default turns the primary instrument into a crash.
        with open(sample_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*72}\nStep {step}"
                    f"{'  (+ held-out)' if include_heldout else ''}\n{'='*72}\n")
            for family, prompt in pairs:
                for temp in probes.TEMPERATURES:
                    gen = torch.Generator(device=device)
                    gen.manual_seed(config.sample_seed)
                    text = _generate(model, enc, prompt, temp,
                                     probes.MAX_NEW_TOKENS, gen)
                    f.write(f"\n--- {family} | T={temp} ---\n"
                            f"[PROMPT] {prompt}\n[CONTINUATION] {text}\n")
        print0(f"Probe suite at step {step}: "
               f"{len(pairs) * len(probes.TEMPERATURES)} generations -> "
               f"{sample_log_path.name}", console=True)
        model.train()

    ########################################
    #        Training and Validation       #
    ########################################

    # --- Resume -------------------------------------------------------------
    # RESUME=auto  -> newest checkpoint under RESUME_DIR, or runs/ by default
    # RESUME=<path> -> that checkpoint
    # unset/0      -> fresh run
    # Broadcast parameters BEFORE loading, so every rank starts from rank 0's
    # init and then every rank loads the same checkpoint independently.
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)

    start_step, loader_state = 0, {}
    _resume = os.environ.get("RESUME", "0")
    if _resume not in ("0", ""):
        resume_dir = Path(os.environ.get("RESUME_DIR", runs_dir))
        ck_path = (latest_checkpoint(resume_dir, recursive=True)
                   if _resume == "auto" else Path(_resume))
        if ck_path is None or not ck_path.exists():
            raise FileNotFoundError(
                f"RESUME={_resume} but no checkpoint found "
                f"(looked in {resume_dir} / at {ck_path})"
            )
        meta = load_checkpoint(model, optimizers, ck_path)
        start_step = meta["step"]
        loader_state = meta["loader_state"]
        if dist.get_rank() == 0:
            prior_sample_log = ck_path.parent / "samples.log"
            copied = copy_sample_log_through_step(
                prior_sample_log, sample_log_path, start_step
            )
            print0(
                f"Copied {copied} prior sample block(s) through step "
                f"{start_step} from {prior_sample_log}",
                console=True,
            )
        dist.barrier()
        run_entry["resumed_from"] = str(ck_path)
        run_entry["resume_step"] = start_step
        print0(f"Resumed from {ck_path} at step {start_step} "
               f"(loader batches={loader_state.get('batches', 0)})", console=True)

    loader_pos = dict(loader_state)
    # MIX selects the slice-mixing loader (PLAN.md §5.1.1). Unset is the default
    # and takes the original single-stream path, byte-identical to the Milestone 1
    # baseline — this file is the control arm, so mixing has to be opt-in or the
    # control stops being one.
    mix_spec = os.environ.get("MIX", "").strip()
    mix_loader = None
    if mix_spec:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
        from mixing import parse_schedule            # noqa: E402
        from mix_loader import MixingLoader          # noqa: E402
        schedule = parse_schedule(mix_spec)
        mix_loader = MixingLoader(
            schedule, config.shard_dir, batch_size=batch_size, seq_len=seq_len,
            train_steps=train_steps, device=device, rank=dist.get_rank(),
            world_size=dist.get_world_size(), start_step=start_step,
            state=loader_pos)
        train_loader = mix_loader
        ep = mix_loader.epochs()
        print0(f"[mix] spec={mix_spec} slices={list(schedule.slices)}", console=True)
        # Upweighting a small slice buys repetition, not diversity. Surface it:
        # register at 60% out of a 40 M-token pool is many epochs on 371 books.
        print0("[mix] projected epochs/slice at end: " + " ".join(
            f"{n}={v:.1f}" for n, v in sorted(mix_loader.epochs_at(train_steps).items())
        ), console=True)
        over = [n for n, v in mix_loader.epochs_at(train_steps).items() if v > 4.0]
        if over:
            print0(f"[mix] WARNING: {', '.join(over)} exceed ~4 epochs, past the "
                   f"point where repetition stops paying (PLAN.md §5.6). More "
                   f"tokens for that slice, not a bigger weight.", console=True)
    else:
        train_loader = distributed_data_generator(
            "data/shards/gutenberg_train_*.bin", batch_size,
            seq_len=seq_len, start_batch=loader_state.get("batches", 0),
            state=loader_pos)

    # start the clock
    training_time = 0
    last_val_step = start_step
    final_val_loss = None
    final_val_step = None
    completed_step = start_step
    # Device-side accumulators: adding a detached scalar is a trivial GPU op with
    # no host sync, unlike .item() every step.
    train_loss_accum = torch.zeros((), device=device)
    train_tokens_seen = 0
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(start_step, train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        # VAL_EVERY forces a fixed cadence. The stock schedule gives only two
        # points in a 100-step run (step 0 and the end), which cannot localise a
        # divergence. Validation is under no_grad and consumes no RNG, so
        # changing its cadence cannot alter the training trajectory.
        val_step_freq = (val_every if val_every
                         else (125 if step / train_steps < 0.9 else 25))
        if step == train_steps or step % val_step_freq == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            # Guard on start_step, not 0: after a resume the first validation has
            # step == last_val_step == start_step, which divides by zero.
            step_avg = (time_since_last_val / (step - last_val_step)
                        if step > last_val_step else float("nan"))
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            final_val_loss = float(val_loss)
            final_val_step = step
            _tl = (float(train_loss_accum / train_tokens_seen)
                   if train_tokens_seen > 0 else float("nan"))
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} "
                   f"train_loss:{_tl:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            # Train loss is accumulated as a device tensor and only read here, so
            # the per-step path stays sync-free (the reference discards the loss
            # entirely precisely to avoid a sync every step).
            metrics = {"val/loss": float(val_loss), "perf/step_ms": 1000 * step_avg,
                       "perf/train_time_s": training_time}
            if train_tokens_seen > 0:
                metrics["train/loss"] = float(train_loss_accum / train_tokens_seen)
                metrics["train/lr_mul"] = optimizers[-1].param_groups[0]["lr"] / \
                    optimizers[-1].param_groups[0]["initial_lr"]
            metrics["train/tokens"] = step * batch_size
            tracker.log(metrics, step=step)
            train_loss_accum = torch.zeros((), device=device)
            train_tokens_seen = 0
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        # Probe suite and checkpoints on INDEPENDENT cadences, deliberately
        # outside the validation block above. Nested inside it they only fired on
        # steps that were also validation steps, so sample_every=100 against
        # val_step_freq=125 silently did nothing — the cadence you set was not the
        # cadence you got. Both clocks are stopped, so this time is not counted as
        # training time.
        # STOP_AFTER exits early WITHOUT altering train_steps. That distinction is
        # the whole point: train_steps sets the LR schedule and is checked by the
        # resume guard, so shortening it would change the trajectory being tested
        # and be rejected as a sizing mismatch. This lets a resume from a
        # long-run checkpoint be validated against the original run's own
        # val_loss at an intermediate step, for minutes of GPU time instead of
        # hours. Diagnostic only; a real run leaves it unset.
        if stop_after and step >= stop_after:
            # If STOP_AFTER is not itself a validation step, include the training
            # time since the last validation without pretending the older loss
            # was measured at this step.
            if step > last_val_step:
                dist.barrier()
                training_time += time.perf_counter() - t0
            print0(f"STOP_AFTER={stop_after} reached at step {step}; exiting early "
                   f"(train_steps={train_steps} unchanged, schedule unaffected)",
                   console=True)
            break

        is_last = step == train_steps
        if is_last:
            # At loop entry `step` optimizer updates and loader batches have
            # completed. Saving here makes the label, weights, optimizer, and
            # data position describe the same state. A resume from an already
            # complete final checkpoint only copies the prior trajectory and
            # does not duplicate the final sample block.
            if step > start_step:
                sample(model, enc, step, include_heldout=True)
                save_checkpoint(model, step, optimizers, loader_pos)
                dist.barrier()
            break

        if step > start_step:
            # Sampling and checkpointing are not training, so their cost must not
            # land in training_time. But the previous version reset t0 here on
            # EVERY step, which meant `time_since_last_val` at the next validation
            # measured a single step instead of the whole window -- understating
            # step_avg and train_time_s by roughly the window length, and writing
            # that wrong wall-clock into the run registry. Shift t0 forward by the
            # excluded duration instead: overhead is removed, window start is kept.
            _extra_t0 = time.perf_counter()
            did_extra = False
            if step % sample_every == 0:
                sample(model, enc, step)
                did_extra = True
            if step % checkpoint_every == 0:
                save_checkpoint(model, step, optimizers, loader_pos)
                did_extra = True
            if did_extra:
                dist.barrier()
                t0 += time.perf_counter() - _extra_t0

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            _loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
            _loss.backward()
            # Instrumentation only -- does not touch the update. The model returns
            # a SUM over tokens, so dividing by tokens later gives mean per-token.
            train_loss_accum += _loss.detach()
            train_tokens_seen += targets[i*mbs:(i+1)*mbs].numel()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        completed_step = step + 1
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

# Update run index BEFORE tearing down the process group -- dist.get_rank() and
# print0 both need it, so writing the record afterwards crashed every run at the
# last line, after all the work was done.
_rank = dist.get_rank()
if _rank == 0:
    # A registry entry without the outcome is only half of PLAN.md §4.0.1: the
    # point is to look up which config produced which curve.
    run_entry.update(completion_metadata(
        completed_step=completed_step,
        train_steps=train_steps,
        batch_size=batch_size,
        train_tokens_available=_shard_meta["total_train_tokens"],
        final_val_loss=final_val_loss,
        final_val_step=final_val_step,
    ))
    run_entry["train_time_s"] = round(training_time, 2)
    run_entry["stop_after_requested"] = stop_after or None
    run_entry["sample_log"] = str(sample_log_path)
    # The direct test of resume correctness: an interrupted run and an
    # uninterrupted one must finish at the SAME corpus position. Loss agreement
    # implies it, but this states it outright and survives in the registry.
    run_entry["final_loader_state"] = dict(loader_pos)
    # A mixed run must be distinguishable from a control run in the registry, or
    # the ablation table cannot tell them apart later.
    run_entry["mix"] = mix_spec or None
    if mix_loader is not None:
        run_entry["mix_state"] = mix_loader.state()
    run_entry["start_step"] = start_step
    run_entry["logfile"] = logfile
    # Per-run record FIRST, at a path only this run writes. index.jsonl is a
    # single append-only file on a shared Volume, and concurrent runs clobber each
    # other: three 1000-step seed runs each finished exit=0 and committed, and only
    # ONE row survived, because a Volume commit is a whole-file snapshot of that
    # container's view. Losing an ablation row silently is worse than losing a run,
    # because the table still looks complete. So run.json is the source of truth
    # and index.jsonl is a best-effort convenience rebuilt by `fetch`.
    record_path = ckpt_dir / "run.json"
    tmp_record = record_path.with_suffix(".json.tmp")
    tmp_record.write_text(json.dumps(run_entry, indent=2, default=str),
                          encoding="utf-8")
    tmp_record.replace(record_path)
    print0(f"Run record written to {record_path}", console=True)

    index_path = runs_dir / "index.jsonl"
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_entry, allow_nan=False) + "\n")
    print0(f"Run index updated at {index_path}", console=True)
    # Mirror the outcome into the W&B summary so the two records agree, then close
    # the run. Finishing before destroy_process_group keeps print0 usable.
    tracker.finish({k: run_entry[k] for k in
                    ("final_val_loss", "train_time_s", "steps_completed",
                     "epochs_over_corpus") if k in run_entry})

dist.destroy_process_group()
