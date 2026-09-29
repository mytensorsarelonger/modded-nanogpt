# MIX-LITERATURE.md — reading notes grounding the Phase 1 mix design

Written 2026-09-29 from the four core papers, extracted and verified against the
source texts (not summaries). Purpose: answer "are we doing this the right way"
for §4.2's mix ablations with measured facts instead of heuristics, and name the
design decisions the corpus/mix phase actually needs to make. Referenced from
PLAN.md §4.2 and §5.1.1. This file holds the *evidence*; the *decisions* stay
in PLAN.md.

The one-sentence synthesis: **the field's answer to "which mixture" is a
small-scale response-surface sweep with a fitted predictive model, and K3-mini
already has the unusual advantage — the §7.2 harness supplies the objective
function the literature had to replace with benchmark scores.**

---

## 1. Data Mixing Laws (Ye et al., arXiv 2403.16952, ICLR 2025)

**Verified claims:**

- Domain-`i` validation loss is predictable from mixture proportions as
  `L_i(r) = c_i + k_i · exp(Σ_j t_ij · r_j)` (their Eqn. 7). The exponential-
  of-linear form (M4) won their function search on coefficient economy
  (M+2 params) with accuracy ≈ the richer M1 form (MAE 0.005–0.031 vs random
  0.077–0.889 across GitHub/Books3/Pile-CC).
- `t_ij < 0` means training domain `j` *helps* validation domain `i` — the
  cross-domain interaction matrix is the actual scientific object. This is the
  formal version of the question the §7.2 gap asks informally.
- Their pipeline: small runs (70M–410M models, 30B tokens, 24–32 mixtures) →
  step-scaling `L(S)=E₁+B/S` → size-scaling `L(N)=E₂+A/N` → mixture-law fit →
  predict the target scale. Result: a 1B model on 100B RedPajama tokens at the
  predicted mixture matches the default mixture trained **48% longer**.
- Continual-training extension predicts the *critical proportion* that avoids
  catastrophic forgetting — the formal tool for the cooldown-ramp idea
  (§5.1.1) and for any "register into the cooldown" schedule.
- Mixture-sampling craft matters: their Alg. 2 uses double-diminishing
  grids and *down-samples* mixtures containing a zero proportion (zeros
  dominate naive grids). Fitting uses AdaBoost on the exponential form for
  stability.

**For K3-mini:** the §7.2 row of every checkpoint is already a domain-loss
vector in exactly their sense (exemplar/craft/mundane). Fitting their form to
~24–32 sampled mixtures at 1000 steps is a direct port; the harness was the
missing precondition, and it now exists.

## 2. Scaling Laws for Mixture Pretraining Under Data Constraints
(Sedova, Seto, Schluter, Ablin — Apple/ITU, arXiv 2605.12715, May 2026)

**Verified claims:**

- 2,000+ runs, 101M–805M GPT-2-style models, targets constrained at
  10M–1B tokens (German/French/Swahili via FineWeb2, OpenWebMath, top-k%
  DCLM quality subsets), generic = English web, mixed at the sample level —
  structurally our setting (scarce register/craft + abundant backbone/general).
- **Headline: in mixtures, scarce target data tolerates 15–20 repetitions at
  optimum**, vs the ≤4-epoch rule from single-source training. Mechanism:
  the generic stream keeps supplying fresh gradients and regularizes.
- Overfitting onset is governed by the repetition factor `r = h·D_total/D_target`
  **alone** — the same `r` triggers degradation regardless of how it is
  reached (weight × pool size are exchangeable).
- Optimal `r` grows with compute budget and target pool size; larger models
  overfit faster but still achieve lower best-loss.
- The cost is real and measured: at high target weight `h`, English benchmark
  average drops (101M: 43.7 → 39.7; 539M: 51.3 → 48.6). There is no free
  register — the tradeoff curve exists; the mixture law predicts where on it
  you are.
- 6-parameter repetition-aware law, R² = 0.96 under `r·h` weighting; fits at
  small scales and extrapolates across model sizes.
- Limitations they state: no joint HP tuning, GPT-2 family only, ≤939M models.

**For K3-mini — this may overturn a standing repo assumption.** At the
3250-step budget (1.704B tokens):

| mix | register weight | register epochs `r` | single-source rule (≤4) | mixture rule (15–20) |
|-----|-----------------|--------------------|-------------------------|----------------------|
| corpus/M1 | 0.087 | 3.7 | OK | far under-optimal |
| mix-a | 0.20 | 8.5 | over | under |
| mix-b | 0.40 | 16.9 | way over | **inside the optimal window** |
| mix-c | 0.60 | 25.4 | — | past optimum |

`mixing.py`'s comment ("mix-c at 60% is repetition, not diversity") and the
~4-epoch warning in the trainer encode the Muennighoff single-source rule.
The Apple result predicts **mix-b's register repetition is near-optimal at
this budget**, and that the M1-style 8.7% *under-exposes* the register. That
is a testable prediction, not a certainty: their targets are languages/math
with perplexity objectives; ours is a register discrimination objective
nobody has measured under repetition. The §7.2 harness can measure it —
see §5 below.

## 3. RegMix (Liu et al., arXiv 2407.01492, ICLR 2025)

**Verified claims:**

- 512 proxy models (1M params, 1B tokens each) on Dirichlet-sampled mixtures;
  **linear ridge regression** mixture→performance; predicted best mixture
  trained at 1B/25B beat all 64 candidate mixtures; beats human selection up
  to 7B/100B; matches DoReMi at **10% of its compute**.
- Rank-invariance assumption: the *ranking* of mixtures is preserved across
  scales even when absolute performance is not — that is why 1M-param proxies
  can steer a 1B target.
- Findings that argue for the automatic approach: mixtures matter up to
  14.6 points (Lambada); web corpora, not "high-quality" Wikipedia, correlate
  most strongly with downstream performance; **domain interactions contradict
  common sense** — the reason hand-designed A/B/C corners are a weaker design
  than a sampled sweep.

**For K3-mini:** the template for the sweep's mechanics, with one substitution:
the regression label is the §7.2 objective (gap + craft PPL constraint), not
downstream accuracy. Their result also justifies short proxies (their proxies
saw 1B tokens; a 1000-step run here sees 524M — coarser, but rank-invariance
is the claim being leaned on, and the seed band is validated at that horizon).

## 4. To Repeat or Not To Repeat (Muennighoff et al., arXiv 2305.13230, NeurIPS 2023)

**Verified claims:**

- The source of the ≤4-epoch rule: up to ~4 epochs of repetition is nearly as
  valuable as unique data; meaningful gains to ~16; ends at ~40. **Applies to
  single-source training** (all tokens repeated) — the Apple paper's point of
  departure.
- Multi-epoch degradation depends on dataset size, model params, training
  objectives; **dataset quality and FLOPs are NOT significant factors** —
  "the register pool is high-quality" is not a defense against over-repetition.
- Dropout is the one regularization that works (needs careful tuning at
  scale); MoE enables cheap dense-HP tuning (a curiosity for Milestone 2,
  not Phase 1).

**For K3-mini:** this is the ceiling the repo's current thresholds encode.
Read it *with* paper 2, whose mixture regime is ours.

---

## 5. What this means for the Phase 1 design (the actual recommendations)

**R1 — Replace hand-picked A/B/C with a sampled response-surface sweep.**
24–32 mixtures (Data-Mixing-Laws' double-diminishing grid with zero
down-sampling, or RegMix's Dirichlet), 1000 steps each (~$2.5–3/run on A100,
~$60–90 total — same order as the planned A/B/C budget), each auto-labeled by
the §7.2 harness. Fit the exponential-of-linear law (per objective) or ridge
(their robust fallback). Predict, then verify: top-1 predicted mixture, plus
the A/B/C corners as anchors, at the full 3250 steps. This is §4.2's
"ablation comparison table" with the response surface as its evidence.

**R2 — Test the repetition window on the register axis.** The Apple paper
predicts register r ≈ 15–20 is optimal at this budget, i.e. register weights
in the 0.35–0.47 range — mix-b's neighborhood — and that 8.7% under-exposes.
The repo's current thresholds encode the single-source rule. The sweep in R1
covers this axis at no extra cost; the exemplar-PPL curve over checkpoints
will show the repetition frontier (improve-then-degrade) if it exists.
Do NOT edit the ~4-epoch warning out of the code before the sweep measures
it — the warning is a hypothesis now, and it is cheaply testable.

**R3 — The objective is the owner's, and it is now formal.** Maximize the
§7.2 contrast-pair gap subject to: craft PPL stays near its backbone-trend
line, and the mundane probe does not collapse (word-diversity floor). Fitting
the response surface per-target (gap, craft PPL, mundane fluency) reproduces
the paper's "balance of capabilities" analysis with our own axes.

**R4 — Precondition unchanged:** the sweep's labels are only meaningful once
`register_exemplars.txt` carries the owner's writing. Sweep *after* the
exemplars, or accept that a re-sweep is mandatory. The craft-essay slice
(§5.2.3) should land before or with the sweep so the surface is fit over the
full five-slice taxonomy rather than refit later.

**R5 — Known-transfer caveats, stated plainly:** (a) perplexity-objective
findings may not transfer to a discrimination objective — R2 is a hypothesis
test, not an assumption; (b) the Apple paper is recent (May 2026), one group,
≤939M models, no joint HP tuning; (c) RegMix's rank-invariance is assumed,
not proven, at our scale; the A/B/C-corner anchor runs in R1 double as its
check. None of these caveats weaken R1 — a sampled sweep with verified
predictions dominates hand-picked corners under every one of them.

## Secondary reading (skim-tier, for breadth)

- DoReMi (arXiv 2305.10429) — DRO-based mixture search; optimizes robustness
  across domains, which is the *opposite* of sacrificing robustness for a
  register — know it, probably not our tool.
- Albalak et al., "A Survey on Data Selection for Language Models"
  (arXiv 2402.16827) — the map of the field.
- DataComp-LM / DCLM (arXiv 2406.11794) — the filtering half of curation;
  model-based filtering is the lever that matters at scale.
- QuRating (arXiv 2402.09739) — classifier-rated selection; the lineage of
  our EDU slice (PLAN.md §5.1.2's audit note).
- Organize the Web / domains-enhance-curation (arXiv 2502.10341) — the case
  that domain structure beats sample-level filtering; supports the slice
  taxonomy the repo already chose.
