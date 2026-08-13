# RL Strategy for K3-mini — Phase 1 Implications

> Handoff from the 2026-08-13 RL-strategy conversation, versioned here verbatim so
> the plan it modifies sits next to it. PLAN.md was amended the same day to carry
> its Phase 1 requirements (see CHANGELOG 2026-08-13). Post-training **execution**
> remains deferred until a Phase 1 base model exists.

**Date:** 2026-08-13
**Context:** Conversation covering LLM RL developments (early 2025 → Aug 2026) and
their implications for the K3-mini project (modded-nanogpt fork, register-restricted
Gutenberg corpus, ~1.2B active / ~5.5B total MoE target, KDA:MLA architecture per
Kimi K3). Phase 1 (pretraining) is in progress; post-training was deliberately
deferred. This doc updates what Phase 1 must now carry and what the eventual
post-training recipe looks like.

## 1. Core strategic conclusions (decisions made in conversation)

1. **The target is a register (distribution) AND a deliberation process (policy).**
   These need different machinery:
   - Register → likelihood training (pretrain/midtrain/SFT). Distribution matching,
     not reward maximization.
   - Deliberation → eventually RL-shaped methods, but only after the deliberation
     distribution exists in the base model.
2. **Do NOT run online RL against an aesthetic/judge reward as the primary
   objective.** It is the most reliable path to mode collapse — the model finds a
   few judge-favored outputs and hammers them; register texture (the entire value
   proposition) dies. Entropy-collapse literature = this phenomenon.
3. **RL-ability is determined at pretrain/midtrain time** (the big 2025–26 lesson;
   cf. Qwen-vs-Llama RL-ability, OctoThinker midtraining rehabilitation). Since we
   control the corpus, design for RL before pretraining ends.
4. **The user's asymmetry is the key resource:** can write the target by hand,
   cannot explain how. Judge >> edit >> write. Post-training should spend
   recognition/editing ability, not require articulation.

## 2. Phase 1 (pretraining mix) — new/confirmed requirements

The mix now has **three jobs, not one**:

1. **Install the register** (existing plan, unchanged).
2. **Install deliberation-shaped text** for later RL to amplify. There is no
   natural "author thinks aloud, then writes" corpus. Nearest in-register
   analogues: prefaces, craft essays, criticism, writers' letters about writing
   (e.g., Poe's "Philosophy of Composition"). Thin, but it's the seed crystal.
   Include it.
3. **Keep a general/instruction-shaped substrate slice** so the model is steerable
   at all (already in plan; do not cut in pursuit of purity).

**Open design decision** (prototype both, decide during Phase 1): thinking channel
**in-register vs. plain functional English**.

- Plain-English thinking (R1-Zero stance: scaffolding, only outcome rewarded) is
  pragmatic and opens an escape hatch — synthetic deliberation traces can be
  manufactured WITHOUT violating the "don't rephrase register data" rule, since
  traces are a separate channel, not corpus. Risk: bleed-through into output
  register.
- In-register deliberation (model reasons like a 19th-c. critic) is more
  alien/on-thesis, much harder to bootstrap.

## 3. Measurement regime (start now, during pretraining)

- **Hand-written exemplars are the eval set.** Track their perplexity under every
  checkpoint and every mix ablation.
- **Contrast pairs:** target writing vs. superficially-similar-but-wrong writing
  (wrong in the way the user can feel but not explain). Track the loss gap — it
  should widen if the mix works. This is the Schulman-style "known curve" for the
  actual goal.
- Track perplexity on **craft-essay/deliberation-style text separately** (job 2
  above).
- Later: **restoration-task accuracy at temperature** (see §5).
- **Always report pass@k of the base model as baseline** before claiming any RL
  gain (guards against the format-unlocking illusion; cf. spurious-rewards results
  on Qwen).

## 4. Post-training ladder (ordered by distance pulled from base distribution)

1. **Pretraining mix** ← currently here
2. **Midtrain/anneal** on hand exemplars + nearest neighbors
3. **Expert iteration / rejection sampling with the user as judge** (sample n →
   user picks best → SFT on picks → repeat). Probably the ceiling of what's needed
   at this scale. Reward model = the user; unhackable, on-distribution.
4. **Contrastive pairs from user edits** (DPO-style: model output vs. user-edited
   version). Edits are dense signal for the inexplicable process.
5. **Online RL against a learned reward** — only if 1–4 leave a gap; heavy KL
   anchor to base; constant diversity monitoring.

**STaR-style bridge for deliberation:** sample with thinking → user selects outputs
that land → SFT on winners INCLUDING their traces. Trains the deliberation process
via recognition alone. Offline/iterative loops (rejection sampling + SFT) may beat
online RL entirely at this budget (cf. PCL-Reasoner offline-RL results).

## 5. Verifiable rewards in-domain (build these; they rescue us from judge-reward collapse)

Ground-truth-anchored literary tasks — all require the register, none can be hacked
against a reward model:

- **Continuation:** deliberate, write next N tokens of a real passage; reward vs.
  held-out real continuation (perplexity/similarity).
- **Restoration:** degrade a real passage (modernize/flatten/delete clauses) →
  model restores; reward = distance to original.
- **Infilling:** delete middle, reward reconstruction.
- **Hard constraints:** meter, rhyme, form — mechanically checkable
  (Countdown-shaped).

These double as the in-domain calibration ladder (Schulman "nuts and bolts" style):
restoration = our Pendulum, continuation = our Pong. Run RL here first, establish
known curves, only then approach the real objective with: outcome-only reward
(user-as-judge + contrastive-perplexity signal + constraint checks), thinking
unrewarded and free, KL anchor to base.

## 6. RL field updates (early 2025 → Aug 2026) — condensed

- **Algorithms:** GRPO-variant wars ended in a shrug. Consensus recipe:
  DAPO-flavored GRPO (clip-higher, dynamic sampling, token-level loss, overlong
  filtering, no KL penalty for verifiable rewards). Dr. GRPO bias fixes
  (length/std normalization) explain much of pathological length inflation.
  GSPO = sequence-level importance ratios. Algorithm choice matters far less than
  data, entropy management, infra.
- **Entropy is THE diagnostic.** Watch policy entropy the way Schulman watched
  reward curves. Known-good curve shapes exist. Tools: clip-higher, high-entropy
  ("forking") token emphasis (~20% of tokens carry the signal), dynamic sampling
  discarding all-correct/all-wrong groups.
- **RLVR debate resolved into nuance:** RL mostly sharpens what base can sample
  (pass@k argument), but prolonged RL (ProRL) can find new modes; many reported
  gains were format-unlocking (spurious rewards on Qwen).
- **Train/inference numerics mismatch is a first-class bug** — rollout engine vs.
  trainer compute different logprobs → silent off-policyness → instability.
  Elevated risk for our stack (chunkwise KDA kernels via `fla` vs. training path;
  MoE compounds it). Action: log the sampler-vs-trainer logprob gap from day one;
  fix via truncated importance sampling or bitwise-matched kernels.
- **On-policy distillation is now a standard pipeline stage** (K3, DeepSeek V4: RL
  specialists → multi-teacher on-policy distillation into one model; no separate
  RLHF/DPO stage). ~10x cheaper than RL when a teacher exists. Note: our register
  model has no natural teacher — mostly N/A, but worth remembering.
- **K3 post-training pipeline (our eventual template):** SFT cold start →
  domain-expert RL at multiple reasoning-effort levels (token-budget multiplier
  curriculum) → consolidation. Length control must be designed in, not bolted on.
- **Agentic RL / environments became the field's center of mass** (K3 AgentEnv
  microVM sandboxes, snapshot/fork, partial rollouts, scaffold randomization).
  Mostly out of scope for us; steal partial rollouts for long-tail generation
  lengths even in single-turn RL.
- **Rubric rewards / reasoning LLM judges matured** for non-verifiable domains —
  relevant since our end goal is non-verifiable, but subordinate to §5's
  verifiable tasks.
- **Spectral inheritance finding:** RL leaves core weight scales ~unchanged,
  shifts singular frames — supports "RL elicits, pretraining creates" → effort
  belongs in the corpus.
- **Calibration ladder, 2026 edition:** single-token bandit → Countdown/TinyZero
  curve reproduction → GSM8K with pass@k vs pass@1 → entropy-collapse stress
  test. Ours adds the §5 in-domain rungs.

## 7. Immediate action items

1. Add craft-essay/prefaces/letters-on-writing slice to the corpus (job 2).
2. Build the exemplar-perplexity + contrast-pair loss-gap eval harness; wire into
   checkpoint eval loop.
3. Prototype both thinking-channel formats (in-register vs. plain) on a small run.
4. Add sampler-vs-trainer logprob-gap logging to the training stack (KDA/MoE
   mismatch risk).
5. Draft the restoration + continuation task generators from the existing corpus
   (they need no new data).
6. Keep the general-text slice; do not cut it.

## 8. Standing prohibitions

- No online RL against an aesthetic judge as primary objective (mode collapse).
- No rephrasing of register data (synthetic deliberation traces OK only as a
  separate channel).
- No RL-gain claims without base-model pass@k baseline.
- Post-training work remains deferred until Phase 1 base model exists — but
  Phase 1 now explicitly carries jobs 1–3 in §2.
