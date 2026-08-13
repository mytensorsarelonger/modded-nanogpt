"""
K3-mini Phase 0 configuration.

Single source of truth for model size and hyperparameters.
Everything reads from this file — see PLAN.md §4.0.1. If you find yourself
hardcoding a value that lives here, that is the bug.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json

# Repo root
ROOT = Path(__file__).parent

@dataclass
class Config:
    # --- Model ---
    # Phase 0 uses the GPT-2 tokenizer (50257) padded to a multiple of 128.
    # This becomes 65536 at Milestone 8, when our own 64K tokenizer is trained
    # (65536 is both a multiple of 128 and the uint16 shard ceiling).
    # It must agree with data/shards/manifest.json — see validate_against_shards().
    vocab_size: int = 50304
    num_layers: int = 12
    model_dim: int = 768
    head_dim: int = 128
    num_heads: int = 6            # 768 / 128

    # --- Tokenizer ---
    tokenizer_name: str = "gpt2"
    eot_token: int = 50256        # tiktoken gpt2 <|endoftext|>, used as doc separator
    uint16_ceiling: int = 65536   # hard limit of the shard payload dtype

    # --- Training ---
    batch_size: int = 8 * 64 * 1024   # 524288 tokens/step
    mbs: int = 8                    # microbatch for 24GB card (was 64)
    seq_len: int = 1024
    train_steps: int = 3250
    val_tokens: int = 20 * 524288

    # --- Optimizer ---
    adamw_lr_embed: float = 0.7
    adamw_lr_head: float = 0.004
    adamw_lr_scalars: float = 0.015
    # NOTE: the annotation is load-bearing. Without it this is a plain class
    # attribute, not a dataclass field, so it silently vanishes from asdict()
    # and therefore from config_hash().
    adamw_betas: tuple[float, float] = (0.8, 0.95)
    adamw_eps: float = 1e-10
    adamw_wd: float = 0.001
    muon_lr: float = 0.025
    muon_wd: float = 0.05

    # --- Data ---
    data_dir: Path = ROOT / "data"
    shard_dir: Path = ROOT / "data" / "shards"
    gutenberg_raw: Path = ROOT / "data" / "gutenberg" / "raw"
    gutenberg_processed: Path = ROOT / "data" / "gutenberg" / "processed"
    tokenizer_path: Path = ROOT / "data" / "tokenizer"
    manifest_path: Path = ROOT / "data" / "manifest.jsonl"

    # Validation split. Held out as WHOLE BOOKS, never a tail slice of the
    # concatenated stream (PLAN.md §5.5.3, §7.3.2). Once chosen the set is
    # pinned to val_books_path so it does not drift as the corpus grows.
    val_frac: float = 0.01
    # A val set of one book is a val set of one author, which is the defect the
    # whole-book holdout exists to avoid. Both constraints must be satisfied, so
    # book count dominates on a small corpus and val_frac dominates on a large one.
    min_val_books: int = 8
    val_split_seed: int = 1930
    val_books_path: Path = ROOT / "data" / "val_books.json"
    shard_size: int = 100_000_000   # tokens per shard

    # --- Runs ---
    runs_dir: Path = ROOT / "runs"
    index_path: Path = ROOT / "runs" / "index.jsonl"

    # --- Sampling (PLAN.md §7.1: the primary instrument) ---
    # 250, not 2000: a 3250-step run sampled every 2000 steps yields TWO reads,
    # which is not a trajectory. The prompts, temperatures and length live in
    # probes.py; only the cadence and the seed are tuning knobs.
    sample_every: int = 250
    # Matched to sample_every, not sparser. ~1.4 GB x 13 per run is trivial on a
    # Volume, and the trajectory is research data: Biderman et al. 2023a found
    # intermediate checkpoints of one run predict final memorization better than
    # smaller fully-trained models do. See PLAN.md §4.0.1.
    checkpoint_every: int = 250
    # Seeds parameter initialisation. Fixed so two runs of the same config are
    # comparable: unseeded, two identical 6-step runs measured 0.0148 apart in
    # val_loss, which would sit as a noise floor under every Phase 0.5 ablation.
    init_seed: int = 1904
    # Fixed seed, reused at EVERY checkpoint on purpose. Identical sampling
    # randomness at step 500 and step 2500 means any difference you read is a
    # difference in the model, not in the dice. An unseeded sampler folds noise
    # into exactly the comparison the instrument exists to make. Distinct from
    # init_seed: this one only seeds the probe suite's generator.
    sample_seed: int = 1899
    # Prompts, temperatures and length live in probes.py — see §7.1.

    # --- Backend ---
    # NCCL is Linux-only, so gloo is the portable default. On Linux multi-GPU set
    # DIST_BACKEND=nccl. Note this participates in config_hash(), so changing it
    # changes the hash — which is correct: it is part of the run's identity.
    dist_backend: str = os.environ.get("DIST_BACKEND", "gloo")

    def __post_init__(self):
        # Both values are passed into the control model. Refuse a configuration
        # that would silently change the projection width or leave one of these
        # hashed architecture fields decorative.
        if self.num_heads * self.head_dim != self.model_dim:
            raise ValueError(
                "num_heads * head_dim must equal model_dim "
                f"({self.num_heads} * {self.head_dim} != {self.model_dim})"
            )

config = Config()


def config_hash(effective_overrides: dict | None = None) -> str:
    """Stable hash of the effective config, for the run registry.

    Runtime sizing flags are resolved by the trainer after this dataclass is
    constructed. Passing them here makes two env-overridden runs hash
    differently instead of recording the same nominal defaults.
    """
    d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()}
    if effective_overrides:
        d.update({
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in effective_overrides.items()
        })
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def manifest_hash() -> str | None:
    """
    Hash of the data manifest — the field PLAN.md §4.0.1 calls out as the one
    people skip and regret. It is what proves a mix comparison was actually a
    mix comparison. Returns None if the manifest does not exist yet.
    """
    if not config.manifest_path.exists():
        return None
    h = hashlib.sha256()
    with open(config.manifest_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_against_shards() -> dict:
    """
    Assert the config agrees with what was actually written to disk.
    Catches the failure mode where config.py drifts from the shards and
    becomes decorative. Call this at the top of any training script.
    """
    shard_manifest = config.shard_dir / "manifest.json"
    if not shard_manifest.exists():
        raise FileNotFoundError(
            f"{shard_manifest} not found — run data/tokenize_corpus.py first"
        )
    with open(shard_manifest) as f:
        sm = json.load(f)

    if sm["vocab_size"] != config.vocab_size:
        raise ValueError(
            f"vocab_size mismatch: config={config.vocab_size} "
            f"shards={sm['vocab_size']}. The shards win — fix config.py."
        )
    if sm["tokenizer"] != config.tokenizer_name:
        raise ValueError(
            f"tokenizer mismatch: config={config.tokenizer_name} shards={sm['tokenizer']}"
        )
    if sm.get("eot_token") != config.eot_token:
        raise ValueError(
            f"eot_token mismatch: config={config.eot_token} shards={sm.get('eot_token')}"
        )
    return sm


if __name__ == "__main__":
    print(f"config_hash:   {config_hash()}")
    print(f"manifest_hash: {manifest_hash()}")
    print(f"vocab_size:    {config.vocab_size}  tokenizer: {config.tokenizer_name}")
    print(f"adamw_betas in asdict: {'adamw_betas' in asdict(config)}")
    try:
        sm = validate_against_shards()
        print(f"shards OK: {sm['total_train_tokens']:,} train / "
              f"{sm['total_val_tokens']:,} val tokens")
    except (FileNotFoundError, ValueError) as e:
        print(f"shard validation: {e}")
