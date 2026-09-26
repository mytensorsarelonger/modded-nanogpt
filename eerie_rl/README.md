# eerie_rl — Tinker LoRA RL for the Weird & Eerie Register

Self-contained Tinker RL package that generalizes the synthetic repo's
Case→Env→Tinker pattern into a literary domain. The corpus is the K3-mini
register slice (372 public-domain weird & eerie books from Gutenberg); the
reward is mechanically checkable — n-gram and word-overlap against held-out
ground truth — so no judge is in the loop and the mode-collapse failure mode
from RL_STRATEGY.md §1 is structurally avoided.

## Files

| File | Role |
|---|---|
| `case.py` | `Case`, `Task`, `score_text`, `oracle_reply`, case generation from the K3-mini corpus. The reward lives here: n-gram precision/recall/F1 + word-overlap Jaccard, blended 50/50. Pure, no I/O. |
| `env.py` | `EerieEnv` (Tinker `Env`), `EerieGroupBuilder` (GRPO), `EerieDataset`, `EerieDatasetBuilder` (chz). One Case → one Env → one step → reward. |
| `smoke.py` | Sampling-only baseline: loads 18 cases, samples the base model, scores, prints per-case breakdown. Cheap — run without asking. |
| `train.py` | The LoRA RL training loop (5 steps, Qwen3-8B, LoRA rank 32). **Needs James's approval before running** (house rule 5). |

## How it maps to the synthetic repo

| synthetic | eerie_rl | What changed |
|---|---|---|
| `foundermath/case.py` | `case.py` | `Case` is now a literary passage + task type + target, not a math item + answer + answer_kind. No numeric grader — the reward is text-similarity. |
| `foundermath/checks.py` | `case.py` (`score_text`) | `all_hard_pass` → `score_text`. Binary 0/1 → float [0,1] because literary restoration is graded. Same "one scoring path" contract: the env and any offline scorer call the same function. |
| `foundermath/grade.py` | `case.py` (`ngram_*`, `word_overlap`) | Deterministic graders keyed by `answer_kind` → deterministic text-similarity functions keyed by `Task`. |
| `foundermath/tinker_env.py` | `env.py` | `FounderMathEnv` → `EerieEnv`. Same `Env`/`EnvGroupBuilder`/`RLDataset`/`RLDatasetBuilder` shapes from the cookbook. System line now asks for literary continuation/restoration/infilling, not `Answer: <value>`. |

## Reward design (RL_STRATEGY.md §5)

Three task types, all verifiable and judge-free:

* **Continuation** — given the first 60% of a passage, produce the rest. Reward = 50% n-gram precision + 50% word overlap.
* **Restoration** — a passage is degraded (semicolons→periods, archaic words→plain, sentence starts lowercased); the model restores the original. Reward = 50% n-gram recall + 50% word overlap.
* **Infilling** — the middle third of a passage is deleted; the model fills it. Reward = 50% n-gram F1 + 50% word overlap.

The n-gram component is the ceiling (exact reconstruction); the word-overlap component is the learnable gradient (diction/register match without requiring the exact sequence). Both are mechanically checkable.

## Running

The Tinker key lives in `~/git/synthetic/.env`. The Tinker stack lives in the synthetic venv.

```bash
# Smoke test (cheap, no training):
cd ~/git/synthetic
export $(grep -v '^#' .env | xargs)
.venv/Scripts/python.exe ~/git/modded-nanogpt/eerie_rl/smoke.py

# RL training (needs approval):
.venv/Scripts/python.exe ~/git/modded-nanogpt/eerie_rl/train.py
```

## Baseline (verified)

Qwen3-8B, 18 cases from 9 books, temp=0:
- Avg reward: 0.052
- Avg n-gram precision: 0.007
- Avg n-gram recall: 0.007
- Avg word overlap: 0.096
- Continuation (n=8): 0.038
- Infilling (n=7): 0.048
- Restoration (n=3): 0.096

The near-zero n-gram component is expected — the base model cannot guess the exact held-out continuation. The word-overlap component (~0.10) gives the RL signal: "use the same diction." RL pushes both toward 1.0; the word-overlap is the learnable gradient and the n-gram is the ceiling.
