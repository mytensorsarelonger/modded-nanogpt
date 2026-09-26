"""Eerie RL — Tinker LoRA RL for the weird & eerie register.

Generalizes the synthetic repo's Case→Env→Tinker pattern (one case = one
verified-reward episode) into a literary domain.  The corpus is the K3-mini
register slice (372 public-domain weird & eerie books from Gutenberg); the
reward is mechanically checkable — n-gram overlap against held-out ground
truth — so no judge is in the loop and the mode-collapse failure mode from
RL_STRATEGY.md §1 is structurally avoided.

Architecture mirrors ``foundermath/tinker_env.py``:

    Case ─► Env ─► initial_observation() ─► sample ─► step() ─► reward
                  ↑                                          │
                  └── score_text() ─────────────────────────┘

One reward function, used by both the env's ``step`` and any offline scorer,
so the smoke test and the RL loop see identical numbers.
"""
