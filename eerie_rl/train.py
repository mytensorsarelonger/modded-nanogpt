"""Eerie RL — the LoRA RL training loop.

Uses the Tinker cookbook's RL training loop with the ``EerieDataset``
from ``eerie_rl.env``.  Config: smallest Qwen (Qwen3-8B), LoRA rank 32,
``group_size=4``, ``groups_per_batch=18``, ``max_steps=5``, learning
rate 1e-5 (LoRA wants ~10x lower than full-FT).

**Before running: report the config and stop. James decides whether
to spend credits** (house rule 5 in ``synthetic/AGENTS.md``, and the
tinker-rl skill's always-on rule).  When approved, run it, then re-run
the smoke test against the trained sampler and report before/after.

Usage (from the synthetic venv)::

    cd ~/git/synthetic
    export $(grep -v '^#' .env | xargs)
    .venv/Scripts/python.exe ~/git/modded-nanogpt/eerie_rl/train.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from synthetic (has the Tinker key)
for p in [
    Path(__file__).resolve().parents[1] / ".env",
    Path.home() / "git/synthetic/.env",
]:
    if p.exists():
        load_dotenv(p)
        break

import chz
from tinker_cookbook import model_info
from tinker_cookbook.rl import train as rl_train

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eerie_rl.case import load_cases_from_corpus
from eerie_rl.env import cases_to_json, EerieDatasetBuilder


@chz.chz
class EerieRLConfig:
    """Config for the eerie RL training run."""

    log_path: str = "/tmp/eerie-rl"
    model_name: str = "Qwen/Qwen3-8B"
    base_url: str | None = None

    # Data
    n_cases: int = 18
    seed: int = 1899

    # Training
    group_size: int = 4       # rollouts per case (GRPO groups)
    groups_per_batch: int = 18  # cases per training batch
    learning_rate: float = 1e-5  # LoRA wants ~10x lower than full-FT
    lora_rank: int = 32
    max_steps: int = 5       # short — the point is end-to-end, not a result
    max_tokens: int = 512
    temperature: float = 1.0

    # Checkpointing
    save_every: int = 5
    eval_every: int = 0  # disabled — no separate eval set yet

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None


def cli_main(cfg: EerieRLConfig) -> None:
    # 1. Load cases from the K3-mini register corpus
    corpus_root = Path.home() / "git/modded-nanogpt"
    manifest = corpus_root / "data/manifest.jsonl"
    raw_dir = corpus_root / "data/gutenberg/raw"

    if not manifest.exists():
        print(f"ERROR: {manifest} not found")
        sys.exit(1)

    cases = load_cases_from_corpus(
        manifest, raw_dir, n_cases=cfg.n_cases, seed=cfg.seed
    )
    print(f"Loaded {len(cases)} cases from the register corpus")

    cases_json = cases_to_json(cases)

    # 2. Resolve renderer
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)

    # 3. Build the dataset builder
    dataset_builder = EerieDatasetBuilder(
        batch_size=cfg.groups_per_batch,
        group_size=cfg.group_size,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        cases_json=cases_json,
    )

    # 4. Build the training config
    train_config = rl_train.Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        recipe_name="eerie_rl",
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        learning_rate=cfg.learning_rate,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        max_steps=cfg.max_steps,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        base_url=cfg.base_url,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )

    # 5. Report and stop for approval
    print(f"\n--- RL Training Config ---")
    print(f"  Model:          {cfg.model_name}")
    print(f"  Renderer:       {renderer_name}")
    print(f"  LoRA rank:      {cfg.lora_rank}")
    print(f"  Group size:     {cfg.group_size}")
    print(f"  Groups/batch:   {cfg.groups_per_batch}")
    print(f"  Max steps:      {cfg.max_steps}")
    print(f"  Learning rate:  {cfg.learning_rate}")
    print(f"  Max tokens:     {cfg.max_tokens}")
    print(f"  Temperature:    {cfg.temperature}")
    print(f"  Cases:           {len(cases)}")
    print(f"  Log path:       {cfg.log_path}")
    print(f"\n  This will take {cfg.max_steps} gradient steps on {cfg.model_name}.")
    print(f"  Approve to proceed (James's call).")

    # 6. Run
    asyncio.run(rl_train.main(train_config))


if __name__ == "__main__":
    cli_main(chz.entrypoint(EerieRLConfig))
