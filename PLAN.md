# Project Plan — A Literary Model in the Weird & Eerie Register

**Status:** Milestones 0 and 1 complete. Baseline trained end to end — 3,250 steps,
462 M-token corpus, val_loss 2.80559, 3.07 h on a rented A100. Prose is competent
Victorian pastiche and **not yet eerie**; the samples say the register share (8.7%)
and the missing general-text slice (0%) are the binding constraints. Next: the
mixing dataloader, then Milestone 2 (`kda_mini.py`).
**Last updated:** 2026-08-08 (see [CHANGELOG.md](CHANGELOG.md) for the run log)
**Owner:** (you)

---

### Changes in this revision

1. **§3.1 is now verified line-by-line against the primary K3 report** (`k3_tech_report.pdf`,
   25pp, on disk). Every constant survived: `g_min = −5`, β₁ = 4, β₂ = 25, `N_s = 2`,
   3:1 + final MLA, cosine + 1% warmup + wd 0.1, N ≈ 8 blocks. Nothing had to be
   corrected. Several *additional* load-bearing details that the plan had omitted are
   now recorded (§3.1, marked ⊕).
2. **§3.4's "known trap" is confirmed.** Report Eq. 11 and §2.3 both state plainly that
   shared experts process full-width `x`; only the routed path goes through `W↓`. The
   circulating explainer is wrong.
3. **Phase 0 now names a specific file** (§3.5, §4.0):
   `records/track_3_optimization/train_gpt_simple.py`. Not `train_gpt.py` at HEAD —
   see Appendix C for why that would have been a costly mistake.
4. **The blueprint's actual architecture is ReLU², not SwiGLU, and has RoPE on every
   layer.** This changes the Phase 0.5 swap order (§4.1).
5. **Corrected data format** (§4.0, §5.5). The shards need a 1024-byte header; the
   earlier "nanoGPT raw uint16 memmap" description was wrong and would have failed an
   assert on first load.
6. **Hardware reality** (§2). The Windows laptop is a 4070 Laptop / 8 GB and is a
   data-pipeline machine only.
7. **Run registry (§4.0.1) and probe diversity (§7, §5.3)** — two process borrowings
   from Poolside's "Model Factory" write-up. Everything else in that post is
   post-training/RL industrialization and is deliberately out of scope.

---

## 1. Thesis

Train a language model from scratch whose distinctive quality is *voice* rather than
breadth: a model fluent in the weird and eerie registers, grounded in classical
literature, and unsheltered from the strange and difficult textual traditions
(Talmud, Midrash, hermetica, mystical and anthropological literature) that most
general-purpose corpora sand down or omit.

Two properties define success:

- **Alien.** The prose should not read as a modern assistant doing a costume. The
  register should be native to the model, present in its syntax and cadence, not
  applied as a topical filter.
- **Useful.** It should remain steerable and coherent enough to interact with. A
  beautiful ghost that cannot follow an instruction is a failed artifact.

The tension between those two is *the* creative decision of the project, and it is
resolved empirically in Phase 1 (§4.2), not by argument.

The architecture is a scaled-down Kimi K3 (KDA + Gated MLA hybrid attention,
block Attention Residuals, Stable LatentMoE with Quantile Balancing). This is a
deliberate second thesis: that the K3 recipe's efficiency and stability claims
reproduce at small scale on a non-standard corpus. That claim is what makes the
project legible to funders even if the literary result is subjective.

---

## 2. Constraints and resources

| Resource | Reality |
|---|---|
| **Windows laptop** (this box) | RTX 4070 Laptop, **8 GB** VRAM. Windows 11, Python 3.13, no torch/triton/tiktoken, no WSL. **Data pipeline and authoring only** — not a training machine. |
| RTX 4090 | 24 GB VRAM. The dev box. Kernel development, debug runs, correctness harnesses. Confirm CUDA + Linux/WSL2 before Phase 0. |
| DGX Spark | 128 GB unified memory, ~273 GB/s bandwidth. Poor for training throughput. Use for fitting/inspecting larger configs and running Kimi Linear locally. |
| Rented H100s | The real training substrate. ~$3–4/GPU-hr on Modal. |
| Budget | Limited. Requires a proposal. **The proposal gets written after Phase 1**, with a real loss curve and real sampled prose in it. |
| Time | Evenings and weekends. Sustained, not intense. |

**Two platform facts that bite immediately.** Both `train_gpt_simple.py` and HEAD call
`dist.init_process_group(backend="nccl")`, and **NCCL is Linux-only** — this fails at
startup on native Windows even at `world_size=1`. And `torch.compile` needs Triton plus
MSVC on Windows. Consequence: **do the training on Linux or WSL2**, not native Windows.
`wsl.exe --install` on the 4090 box is a better investment than fighting either.

**Governing risk:** solo from-scratch projects die in data engineering — six months
of pipeline work, an ever-growing corpus, no model. The entire plan below is
structured to force an end-to-end run early against a deliberately mediocre corpus,
then improve each stage against a thing that already trains.

---

## 3. Architecture — "K3-mini"

Faithful to the K3 report where it concerns *stability and information flow*;
relaxed where the original choices exist only to solve trillion-parameter
infrastructure problems.

### 3.1 Keep exactly as specified — **all verified against the report**

These are the parts that are cheap, load-bearing, and where deviation causes
divergence at step 40K rather than step 400. Section references are to the primary
report; ⊕ marks a detail added in this revision.

- **3:1 KDA : Gated MLA** layer ratio, repeated in 4-layer macrocycles, with **one
  additional Gated MLA at the very end** so the final operation is always global
  attention. (§2.1. K3 = 69 KDA + 24 MLA = 93 layers: 23 macrocycles + 1 final MLA.)
- **NoPE on all MLA layers.** No RoPE, no positional embedding. KDA's recurrent
  gating carries position. Payoff: context extension needs no RoPE rescaling or YaRN
  (§2.1.2, §3.4 — K3 extrapolates to 1M with no positional-encoding change at all).
- **KDA parameterization** per Eq. 2: ShortConv → Swish on q/k/v, L2Norm on
  q/k, low-rank decay projection `z = W(Wx) + b_h` with per-head bias.
- **State layout `S ∈ ℝ^{d_k × d_v}`, read as `õ = Sᵀq`** (Eq. 1). Pick this and hold it.
- **Bounded log-decay**, `g = g_min · sigmoid(e^A · z)` with **`g_min = −5` fixed**,
  learnable per-head `A` **initialized to 0** (Eq. 5). Not optional. ⊕ The report gives
  the exact reason: with `g_min = −5` every retention factor exceeds `e⁻⁵ ≈ 6.7e−3`, so
  cumulative log-decay over a 16-token tile stays in (−80, 0), the reciprocal rescaling
  factor stays under `e⁸⁰` and inside bf16 range, and **all** causal tiles — diagonal
  included — run as dense Tensor Core matmuls. This is what eliminates the position-pair
  diagonal path that bottlenecks Kimi Linear.
- **Full-rank output gates** on both KDA and Gated MLA (Eqs. 6, 7), replacing Kimi
  Linear's low-rank gate.
- ⊕ **Head-wise RMSNorm on the KDA recurrent output, inside the gate:**
  `y = W_o[sigmoid(W_g x) ⊙ RMSNorm(õ)]` (Eq. 6). Easy to drop on a first pass. Don't.
- **SiTU-GLU** with **β₁ = 4** (gate branch), **β₂ = 25** (up branch) (Eq. 12, Fig. 4).
  ⊕ Note the exact form: the gate branch is `β₁·tanh(W_g x / β₁) ⊙ sigmoid(W_g x)` —
  **the same `W_g` feeds both factors**; it is Swish with its linear half soft-capped,
  not two separate projections. Output bound is `|f| ≤ β₁β₂ = 100`.
- **RMSNorm between routed-expert aggregation and the up-projection** (`W↑`) —
  `y = Σ E_shared(x) + W↑·RMSNorm(u)` (Eq. 11, §2.3.1). Report notes it both stabilizes
  training *and* independently improves validation loss and downstream benchmarks.
- **Shared experts operate at full model width; only routed experts live in the
  latent space.** `N_s = 2` in every layer. (Eq. 11 — see §3.4.)
- ⊕ **The router is a sigmoid, not a softmax:** `s = sigmoid(W_r x)`, and mixture
  weights `p` are normalized over the *selected* experts only (Eq. 13).
- **Quantile Balancing** for load balancing (§2.3.3), with histogram-based quantile
  estimation and mean-centered biases. Bias frozen at inference. ⊕ Mechanics worth
  getting right: route with **Top-(k+1)** on the biased score, where the (k+1)-th entry
  gives that token's cutoff `τ_i`; then set `b_j ← −quantile_{1−k/n}(s_{:,j} − τ)` and
  mean-center the bias vector (Eq. 14). ⊕ **The bias is applied only on the *next* step
  — a batch is never routed with a bias derived from itself.** ⊕ The bias is excluded
  from `p`, so it steers dispatch without touching router gradients.
- **Muon** (per-head variant for attention projections — partition Q/K/V momentum along
  the head dimension and orthogonalize each head's block separately, §2.5) + **weight
  clipping** for matrix params; AdamW for embeddings, norms, scalars.
- **Cosine LR decay**, 1% linear warmup, weight decay **0.1 throughout** (§3.3).
- **Block AttnRes** (§2.2). ⊕ K3 partitions 93 layers into **8 blocks of 12** (a partial
  final block; 9 total counting the embedding). ⊕ Structure per Eq. 8–10: a learnable
  per-layer pseudo-query `q_l`, keys/values are RMSNorm'd layer outputs, weights are a
  softmax kernel, and **the token embedding `b₀ = h₁` is always available as a source**.
  Within a block, layer outputs are summed into partial sums; across blocks, full
  attention runs over only the N block-level representations. Overhead drops O(Ld) → O(Nd).
- **MTP head** — cheap during pretraining (K3 uses 1 MTP layer), becomes a free
  EAGLE-style speculative decoding draft later.
- ⊕ **MLA attention output kept in FP32 during training** to correct flash-attention's
  biased rounding error (§2.1.2). Free correctness; note it before debugging a slow
  divergence.

### 3.2 Deliberately relax

| K3 | K3-mini | Why |
|---|---|---|
| 16-of-896 experts (sparsity 56) | 8-of-64 or 8-of-128 | Sparsity 56 exists for serving economics at 2.8T and requires MoonEP-grade expert parallelism. Not justified below ~50B total params. |
| MXFP4 weights / MXFP8 activations, QAT from SFT | bf16, optionally FP8 GEMMs via Transformer Engine | QAT is a deployment optimization for a model that must be served. Irrelevant pre-first-checkpoint. |
| 160K vocab | ~64K, **trained on our own mix** (pad to 65536) | See §5.4. |
| 8K → 64K pretrain, 256K → 1M in cooldown | 1K (blueprint) → 4K/8K, extend later if wanted | Long-context curriculum is a separate project. Note **8K is K3's actual starting context**, so 8K is the faithful floor and 4K is a genuine relaxation below the recipe. KDA + NoPE makes extension cheap to add later, which is a nice property to hold in reserve. |
| Native vision (MoonViT-V2) | Text only | Out of scope. |
| 93 layers / 2.78T total / 104B active | See §3.3 | — |

### 3.3 Candidate configs

Parameter counts are estimates from the standard MoE arithmetic; validate against an
actual instantiation before quoting them anywhere.

| | Debug | Ablation | Flagship |
|---|---|---|---|
| d_model | 768 | 1024 | 1536 |
| Layers | 9 (2 cycles + final MLA) | 17 (4 cycles + final MLA) | 25 (6 cycles + final MLA) |
| Heads × head_dim | 6 × 128 | 8 × 128 | 12 × 128 |
| Latent MoE dim (ℓ) | 384 | 512 | 768 |
| Expert hidden | 320 | 480 | ~704 |
| Routed experts / active | 16 / 4 | 32 / 8 | 128 / 8 |
| Shared experts | 2 | 2 | 2 |
| Dense layers | 1 | 1 | 1 |
| AttnRes blocks | 2 | 4 | 6 |
| ≈ Active params | ~120 M | ~300 M | ~1.2 B |
| ≈ Total params | ~350 M | ~1.1 B | ~5.5 B |
| Target tokens | 2 B | 15 B | 60–100 B effective |
| Hardware | 4090 | 1× H100 | 8× H100 |
| Wall clock | days | ~1 day | ~3 days |
| ≈ Cost | $0 | ~$100/run | $2–5 K |

**Ratios check out against K3.** Latent ℓ = 0.5 × d_model exactly matches K3
(3584 / 7168). Expert hidden at 0.83–0.94 × ℓ is close to K3's 3072 / 3584 = 0.857 —
tighten toward 0.857 if you want to be strict. `N_s = 2` and 1 dense layer match K3
exactly. Head_dim 128 matches. AttnRes blocks are sized to land near K3's ~12 layers
per block at these depths; at 9 layers, 2 blocks is the honest floor.

**On sizing:** the flagship is deliberately smaller in active params than the 3B I
first suggested. Register and texture reward *tokens*, not parameters. Spend the
compute on data passes.

### 3.4 Known trap — **confirmed wrong by the primary report**

A circulating GPT-2→K3 explainer states that shared experts are down-projected into
the latent space. **This is wrong**, and the report is unambiguous in two places.
§2.3: "shared experts retain a full-width path for common transformations, whereas
specialized routed experts operate in a compact latent space of width ℓ." And Eq. 11,
where `E_shared: ℝᵈ → ℝᵈ` while `E_routed: ℝ^ℓ → ℝ^ℓ`, with only the routed path
projected down by `W↓`, RMSNorm'd, and projected back up by `W↑`.

Getting this backwards builds a different and worse architecture. That same
explainer also omits the RMSNorm and has non-causal / stale-state bugs in its
pedagogical snippets. It is good for intuition, unsafe for copy-paste.

### 3.5 The blueprint — `train_gpt_simple.py`

**File:** `records/track_3_optimization/train_gpt_simple.py` (372 lines). Its own
docstring: *"prepared as a simplified version of the speedrun for use in neural net
optimization research."* That is precisely the job — it is maintained as a clean
reference against which optimizer and architecture changes are measured, which is
exactly what Phase 0.5 needs. Copy it into the new repo as `train_baseline.py` and
never edit it again; every variant is a sibling file.

**What it already gives us, for free:**

| Component | Blueprint | Relationship to K3-mini |
|---|---|---|
| Optimizer | Muon (Newton–Schulz 5, μ=0.95, Nesterov, `max(1, rows/cols)^0.5` scaling, round-robin sharded via `all_gather`) + AdamW for embed/head/scalars | **Exactly the right starting point.** Per-Head Muon is a ~15-line diff to `muon_update`. |
| Norm | RMSNorm, pre-norm blocks | Same as K3 |
| QK-norm | `F.rms_norm(q)`, `F.rms_norm(k)` before rotary | KDA uses **L2Norm** — close cousin, small diff |
| Attention | `F.scaled_dot_product_attention`, `is_causal=True`, fixed `scale=0.12`, head_dim 128 | No Triton, no FlexAttention. Clean surface to swap. |
| Logit softcap | `15 · logits · rsqrt(logits² + 15²)` | Orthogonal stability win. Keep. |
| Dataloader | `.bin` shards, microbatch grad accumulation, works at world_size 1/2/4/8 | Keep as-is |
| Depth mixing | Standard additive residual | → Block AttnRes |

**Deltas to K3-mini, and the ones that surprise you:**

1. ⚠ **The MLP is `relu().square()` at 4×dim — not SwiGLU, not gated at all.** One `fc`,
   one `proj`. So the FFN path is *two* diffs to SiTU-GLU, not one (§4.1).
2. ⚠ **RoPE is on every layer** (half-truncated, base freq 1024). K3-mini needs NoPE on
   MLA layers with KDA carrying position, so RoPE removal is part of the attention swap.
3. ⚠ **All `Linear`s have `bias=True`.** Unusual, and it interacts with the Muon/AdamW
   param split (`p.ndim < 2` → AdamW). Decide early whether K3-mini keeps biases; K3's
   only notable bias is the per-head `b_h` in KDA's decay projection.
4. `vocab_size=50304` hardcoded (50257 padded to a multiple of 128) → 65536 later.
5. Weight decay is **0.001 (AdamW) / 0.05 (Muon)**, not K3's 0.1 throughout.
6. LR schedule is **stable-then-linear-decay with `cooldown_frac=0.7`** — i.e. WSD-ish,
   not K3's cosine. Which is convenient: it makes the cosine-vs-WSD ablation in §4.2 a
   natural early experiment on ground you already control.
7. Embedding is `.bfloat16()`; `proj` weights are **zero-initialized**; other weights
   `normal_(std=0.33**0.5 / fan_in**0.5)`.

**Baseline hyperparameters to record before changing anything** (these are the numbers
your sweeps start from): 12 layers × 768 dim, 6 heads × 128, seq_len 1024,
batch 8×64×1024 = 524 288 tokens/step, `mbs=64`, 3250 steps (≈1.7 B tokens),
target **3.28 val loss**, AdamW lr — embed 0.7 / head 0.004 / scalars 0.015,
betas (0.8, 0.95), eps 1e-10; Muon lr 0.025.

---

## 4. Execution plan

### 4.0 Phase 0 — Control arm and infrastructure (4090, ~$0)

The point of Phase 0 is that when K3-mini misbehaves, you can tell *which* of
architecture / data / infra is at fault. Without a known-good baseline, every bug is
a three-way mystery.

1. `git init`. One config file defines a model size; everything reads from it.
   **Then build the run registry, before the second run exists.** This plan generates
   a lot of runs — eight swaps in §4.1, three mix runs in §4.2, plus LR sweeps — and
   the thing that kills a solo project's ability to *use* them is not compute, it's
   bookkeeping. By week 6, "which curve came from which config" has to be a lookup,
   not a memory. Every run emits one indexed record containing:

   | Field | Why |
   |---|---|
   | `run_id` (uuid) + short human tag | The blueprint already generates a uuid logfile name |
   | git SHA + dirty flag | Which code |
   | config hash | Which architecture/HPs |
   | **data manifest hash** | Which corpus. **The one people skip and regret** — it is what proves a Phase 1 mix comparison was actually a mix comparison and not a silently-changed shard set. |
   | tokenizer id/hash | Changes under you at Milestone 8 |
   | loss curve (train + val) | |
   | sample log path (§4.0.5) | |
   | wall clock, hardware, step count | Cost accounting for the proposal |

   The blueprint already does a crude version of this: `train_gpt_simple.py` opens
   `sys.argv[0]` at startup and writes its **entire own source** into the run log
   before training starts, so the code is always recoverable from the log. Keep that
   and extend it — adding the manifest hash and a top-level index (`runs/index.jsonl`)
   is about an hour of work and it is the difference between having ablations and
   having a folder of curves.

   *Borrowed from Poolside's Model Factory, whose stated aim is "reducing researcher
   overhead for bookkeeping" so evals and training changes integrate systematically.
   For a solo evenings-and-weekends project that overhead is the binding constraint,
   which is exactly why this is cheap and load-bearing here.*
2. **Copy in `records/track_3_optimization/train_gpt_simple.py`** as the control.
   Not HEAD's `train_gpt.py` — see Appendix C. Then make exactly these changes:
   - `backend="nccl"` → `"gloo"` (or run under WSL2/Linux and leave it alone).
   - `mbs=64` → 4 or 8 for a 24 GB card. It already accumulates microbatches, so the
     effective batch and the loss curve are unchanged — only wall clock moves.
   - Point `train_files` / `val_files` at your own shards.
   - Everything else: leave alone. It is the control.
3. **Write the shard writer against the real format.** `_load_data_shard` requires:

   | Offset | Content |
   |---|---|
   | 0–1023 | 256 × `int32` little-endian header |
   | `header[0]` | `20240520` (magic — asserted) |
   | `header[1]` | `1` (version — asserted) |
   | `header[2]` | token count (asserted against file size) |
   | `header[3:256]` | zeros |
   | 1024+ | `num_tokens` × `uint16` little-endian |

   This is the llm.c / modded-nanogpt format, **not** a raw memmap. Write a
   round-trip test (`write_shard` → `_load_data_shard` → compare) the same evening,
   so a wrong header costs ten seconds instead of a launch.
   *uint16 caps vocab at 65536 — another reason to pad the 64K tokenizer to exactly that.*
4. This produces the boring 80% that must exist regardless: tokenizer → binary
   shards → memmap dataloader → training loop → checkpoint/resume → sampling →
   eval harness. **Note the blueprint has no checkpointing and no sampling** —
   `save_checkpoint` doesn't exist in it. Those are yours to add, and they are the
   first two things to add.
5. **Wire a sample dump into every checkpoint, from the very first run.** Fixed prompts,
   fixed seed, every 2K steps, appended to a log. For a literary model this is the
   *primary instrument* — loss will not tell you when the model starts sounding eerie.
   Reading will. **Build it as a probe suite, not three prompts** — see §7.1 for the
   composition and the reason.

**Exit criterion:** a dense baseline loss curve and a sample log, on your data, that
you trust. Sanity anchor: the blueprint hits 3.28 val loss in 3250 steps on FineWeb10B.
Your Gutenberg-heavy corpus will land somewhere different — that difference is the
first real thing you learn about your data.

### 4.1 Phase 0.5 — Component build, one swap at a time

Never build K3-mini as a monolith. Each step is a measurable diff against the control,
in its own file, with its own short run and its own curve.

1. **`kda_mini.py` correctness harness, in isolation.** Sequential (loop-over-tokens)
   delta rule as ground truth → chunkwise version → `assert allclose(1e-4)` → diff
   both against `flash-linear-attention`'s KDA on identical inputs. Hold the state
   layout fixed at `d_k × d_v` read with `Sᵀq` (Eq. 1) or you will lose an afternoon to
   a silent transpose bug. **This harness becomes the permanent regression test** when
   fused kernels get swapped in later.
2. Then, in this order (cheapest and safest first — revised for the blueprint's actual
   architecture):

   | # | Swap | Note |
   |---|---|---|
   | a | ReLU² → **SwiGLU** | The blueprint isn't gated. This step exists only to give SiTU-GLU an honest comparand. |
   | b | SwiGLU → **SiTU-GLU** (β₁=4, β₂=25) | Now (a) vs (b) *is* the §4.2 SiTU-vs-SwiGLU ablation, already paid for. |
   | c | Muon → **Per-Head Muon** + weight clipping | ~15-line diff to `muon_update`; isolate it while the model is still dense. |
   | d | wd 0.001/0.05 → **0.1 throughout**, WSD-ish → **cosine + 1% warmup** | Also yields the cosine-vs-WSD ablation. Sweep each schedule's HPs independently — that is the whole point of K3's finding. |
   | e | dense FFN → **LatentMoE + QB** | 2 shared experts full-width, routed in latent ℓ=0.5·d, RMSNorm before `W↑`, 1 dense layer. |
   | f | SDPA + RoPE → **3:1 KDA / Gated MLA hybrid, NoPE on MLA** | Hardest and most bug-prone, so it comes late, when the rest is known-good. Consumes the `kda_mini.py` kernel. |
   | g | additive residual → **Block AttnRes** | |
   | h | add **MTP head** | |

   By the time the flagship starts, you have seven or eight small ablations in hand —
   which is both good engineering and most of the Phase 1 proposal already written.

**Reading protocol:** Moonshot's `modeling_*.py` (downloaded config/code only, no
weights) open beside the relevant report section. KDA class next to §2.1.1; MoE block
next to §2.3; QB next to §2.3.3 and Fig. 5, which is the clearest thing in the report.

### 4.2 Phase 1 — The mix ablations (~$500)

The one experiment that actually decides what this model *is*.

Train three ablation-scale models (§3.3) differing only in corpus mix:

| Run | Backbone (classical + general) | Register + scripture |
|---|---|---|
| A | 80% | 20% |
| B | 60% | 40% |
| C | 40% | 60% |

Read the sampled prose side by side. Also record held-out perplexity on (i) weird-
register text and (ii) general modern text, to see the tradeoff curve move. **The eval
here is you, sitting with the outputs, deciding which one is the model you meant.**

Secondary ablations, most of which fall out of Phase 0.5 for free: cosine vs WSD under
independently-tuned HPs (verified — K3 §3.2 found cosine wins when each schedule gets
its own scaling-law search, and explicitly attributes contrary results to shared HPs);
QB vs sign-update balancing; SiTU-GLU vs SwiGLU; 3:1 hybrid vs full attention.

**Exit criterion:** a chosen mix ratio, a stability story, and three paragraphs of
prose worth showing someone. **Write the funding proposal now**, not before.

### 4.3 Phase 2 — Flagship (~$2–5 K)

Flagship config, chosen mix, 60–100B effective tokens, ~3 days on 8× H100.

- LR/batch-size sweep at debug scale first to set hyperparameters for the new
  architecture. K3 retuned batch size, LR, TPP *and* model shape via dedicated
  scaling-law studies precisely because the architecture changes moved the optimum
  (§3.2); the resulting values are not published. Budget for this.
- **Cooldown phase matters disproportionately.** Final LR-decay stretch upweights the
  highest-quality register material and the longest coherent documents. A large share
  of final character comes from this phase. (K3 does the same, and additionally
  upsamples long documents during cooldown so short text doesn't swamp them, §3.4.)
- Checkpoint aggressively. Sample dump every 2K steps throughout.

---

## 5. Data plan

### 5.1 Strategy: hybrid

Do **not** attempt to out-curate FineWeb-Edu or DCLM on general web text. That is a
months-long detour with a guaranteed loss. Take an open, already-filtered backbone as
the bulk and spend all real effort on the slice nobody else has.

| Slice | Share | Notes |
|---|---|---|
| **Classical / public-domain literature backbone** | 50–70% | Gutenberg first; Internet Archive / HathiTrust PD scans later for volume. |
| **Weird & eerie register** | 10–25% | Hand-curated. Small in absolute tokens; earns influence via upweighting and cooldown placement. |
| **Scripture & commentary** | 5–15% | Sefaria (Talmud, Midrash, Zohar) with aligned Hebrew/Aramaic–English; Perseus for Greek/Latin; PD translations of mystics, hermetica, early anthropology. |
| **General modern text (+ a sliver of code)** | 10–20% | **Not optional.** Not for benchmarks — for syntactic and logical scaffolding, and because it is what keeps the model steerable rather than a beautiful ghost. |
| **Register-rendered synthetic** | 5–15% | See §5.3. |

### 5.2 Sourcing order (deliberately: easiest first)

1. **Gutenberg — tonight.** `gutendex` API or the rsync mirror. Strip license
   boilerplate headers. A few billion clean tokens in a day or two. Not enough for the
   flagship; plenty for Phases 0–1. **This is the commit that ends "starting from
   scratch."** It is also CPU-only and needs nothing but `tiktoken`, so it runs on the
   laptop regardless of the GPU situation.
2. **Sefaria.** Well-structured API and GitHub export; clean bilingual output. Decide
   the interleaving format early (source ¶ / translation ¶ / commentary) — it shapes
   the model's bilingual behavior.
3. **The weird & eerie slice, by hand.** Author by author. Automation does not help
   here; taste does. Separate directory, per-document metadata from day one.
   Canon (all pre-1931, so US public domain): Poe, Bierce, Machen, Blackwood, Dunsany,
   M. R. James, Hodgson, Chambers, early *Weird Tales*.
4. **Internet Archive / HathiTrust PD scans — last.** The big backbone, but a real
   project: the data-engineering problem becomes *OCR quality filtering*, not web
   filtering (broken hyphenation, long-s/f confusions, running headers, binary
   garbage). Perplexity-based filtering with a small model plus OCR-artifact
   heuristics. Defer until Phase 1 says the mix actually needs the volume. Note
   HathiTrust restricts bulk full-text download for non-members; IA is more open.

**Scale reality check:** Gutenberg is only ~5–10B tokens total — it feels infinite, it
isn't. The canonical weird register is perhaps 100–300M tokens. The volume, if needed,
lives in the OCR'd PD book scans.

### 5.3 Invert the rephrasing trick

K2/K3 use fidelity-verified rephrasing to multiply *knowledge* while discarding surface
form — confirmed in K3 §3.1, which applies it to knowledge and mathematics corpora with
"style and perspective-diverse prompting, chunk-wise autoregressive generation, and
fidelity verification against the source documents." Note *which* domains: knowledge and
math, never style-bearing text.

**So do not do this to the register data** — the surface form is the entire value;
rephrasing Machen into diverse modern styles launders out exactly what is being bottled.

Run it the other direction instead: use a strong existing model to render **modern
factual text** (encyclopedia entries, procedures, explanations) **into the register**,
fidelity-verified against the source facts. Borrow K3's machinery — chunk-wise
generation and fidelity verification — and invert the axis being diversified: hold
content fixed, move style, rather than hold facts fixed and diversify style. This is how
the register generalizes into a *voice* usable on any subject, rather than welding itself
to cosmic-horror topics. Use public-domain authors as style targets. Plausibly the
highest-leverage single thing to build, and a strong candidate for 5–15% of the budget.

**The verifier needs two checks, not one.** Fidelity against the source facts is the
easy half and the only half the plan originally had. The hard half is taste, and the
failure mode is specific enough to name: a model asked to render encyclopedia prose
"in the register" drifts to **pastiche** — stacked adjectives, portent, gestures at the
unnameable — which would pass any naive style metric while being bad prose. That is
reward hacking in the style dimension, and at 5–15% of the token budget it would teach
the model to write exactly the thing this project exists to avoid.

So: a second judge, scoring for register *without* pastiche, **calibrated against a few
hundred examples you labelled yourself**, with its agreement rate against your labels
tracked rather than assumed. Poolside calibrate their LLM-as-a-Judge against human labels
and monitor its hit rate continuously for the same reason — an uncalibrated judge on
synthetic data is a generator of confident garbage.

This also sharpens the §9 decision on whether to build the slice at all: **if the taste
judge cannot be calibrated to agree with you, that is the answer.** Do the calibration
first, on a few hundred hand-labelled examples, before generating a single token of
production synthetic data. It is a cheap gate on an expensive build.

### 5.4 Tokenizer: train your own

Reversing earlier advice — the corpus forces it. Off-the-shelf tokenizers shred
Hebrew/Aramaic into bytes, fragment Greek, and waste vocabulary on `\n\n###` and
JavaScript. A ~64K BPE trained on the actual mix is an afternoon's work, materially
improves effective context and loss on this distribution, and there is no compatibility
to preserve when training from scratch. One of the few places a solo builder gets a real
edge. (K3 uses 160K for a multilingual frontier model; 64K is right for this corpus.)

**Pad the vocab to exactly 65536** — a multiple of 128 for tensor-core alignment, and
the ceiling of the `uint16` shard format.

**Timing:** use GPT-2's tokenizer (`tiktoken`, 50257 → padded 50304, matching the
blueprint) for Phase 0 throwaways. Train the real one once the mix ratio is settled by
Phase 1.

### 5.5 Pipeline

In the order things cause damage:

1. **Dedup first, aggressively.** Exact (hash on normalized text), then fuzzy
   (MinHash-LSH, ~0.8 threshold). Duplicates waste compute and inflate memorization.
   `datatrove` (built FineWeb) does all of it; Dolma's toolkit and NeMo Curator are
   alternatives. (K3 §3.1 and §3.4 both lean on exact + fuzzy dedup as the first stage.)
2. **Two-tier quality filter.** Cheap heuristics (length, symbol ratios, repetition),
   then a small model-based classifier — the same rule-heuristics-then-classifier
   structure K3 describes. The FineWeb-Edu approach is reproducible and *retargetable*:
   label a few thousand documents from this corpus by quality and train on those.
3. **Hold out an OOD validation set now**, before any training. K3 fits its scaling laws
   on held-out OOD validation data (§3.2) for the same reason. Run n-gram contamination
   checks against anything to be reported.
4. **Tune the mix on small models.** This is Phase 1. (Also K3's method — "domain-specific
   sampling rates determined by ablation studies on smaller models," §3.1.)
5. **Storage:** parquet for the document-level corpus and manifest; pre-tokenized
   `.bin` shards in the header format above (§4.0.3) for training, so GPUs never wait
   on tokenization. Disk ≈ tokens × 2 bytes.

### 5.6 Tokens-per-parameter

Chinchilla's ~20 TPP is loss-optimal; modern models run 200–2000 TPP for inference
economics. What genuinely needs trillions of tokens is *broad world knowledge and
reasoning*. Register, diction, syntax, and voice saturate far earlier. A
style-and-register model at 60–200 effective TPP is not undertrained for its purpose.

Note that K3 treats TPP as a *tuned* quantity, not a constant — it is one of the four
things retuned by the scaling-law study (§3.2). Same applies here: it is a knob, and
your corpus is unusual enough that borrowed values are guesses.

Data-constrained scaling (Muennighoff et al.): repeating unique data up to **~4 epochs**
is nearly as good as fresh tokens, with returns decaying after. So ~25B unique tokens
→ ~80–100B effective. **Size the model against the corpus, not the reverse.**

Caveat: a uniform corpus means low entropy and fast loss descent, but also brittleness
off-manifold. This is what the general-text slice in §5.1 is buying.

---

## 6. Legal and provenance

The bright line established in *Bartz v. Anthropic*: **training on lawfully acquired
text was held fair use; acquiring text from pirate sources was not.** That was a single
N.D. Cal. district ruling from June 2025, never appealed (Anthropic settled for
$1.5 B over the pirated corpus, final approval July 2026), so it binds no other court —
but the *acquisition* half of the distinction is the practical rule to live by.

**The live risk for this project specifically:** the moment the search turns to Ligotti,
Aickman, or the post-1930 *Weird Tales* run, the top result will be a shadow library and
it will be easy to rationalize. Don't. There is no fair-use shelter for the acquisition
step, and the register survives intact without any single 20th-century author.

**Rules:**

- **US public domain rule of thumb: publication date ≤ 1930.** Check per *work*, not per
  author. (Lovecraft: "The Colour Out of Space" (1927) and "The Call of Cthulhu" (1928)
  are clear; *At the Mountains of Madness* and *The Shadow over Innsmouth* (both 1936)
  are not until 2032. His renewal status is separately contested — a fun rabbit hole,
  not something to stake a corpus on.)
- **`manifest.jsonl` from document one.** Source, license, publication date, cleaning
  steps applied, hash. This is the artifact that keeps the corpus clean, enables
  contamination checks, answers "what's in it," and is the first thing a funder asks for.
- Print-book purchase-and-destructive-scan pipelines are irrelevant here: industrial
  scale, and every target text is already free.
- Not legal advice. If this moves toward public release or funding, a short IP consult
  is cheap insurance.

---

## 7. Evaluation

Standard benchmarks are near-useless for this model. The instrument stack:

### 7.1 The probe suite — primary instrument

The sample log (§4.0.5) is the thing you actually steer by, so its design matters more
than any other eval decision here. **Do not run it as three prompts at one seed and one
temperature.** The failure mode is not that the model overfits the probe — it's that
*you* do. Read the same three continuations every 2K steps for six weeks and you start
reading for whether it does the thing you've watched it learn to do, which is the point
at which your primary instrument stops measuring and becomes a habit.

Compose it instead as:

- **Three or four prompt families**, not three prompts: cold open / scene-setting;
  continuation of a supplied paragraph in a given voice; plain exposition on a mundane
  subject (this is the one that catches register-as-costume, because a model with the
  register in its *syntax* stays eerie describing a kitchen); and instruction-shaped
  ("describe X") to watch steerability.
- **More than one sampling setting** — at minimum a low and a high temperature. Register
  collapses differently at each, and a single setting hides it.
- **A held-out set you look at only at phase boundaries.** Same discipline as the OOD
  validation split in §5.5, applied to your own taste. If the held-out prompts and the
  routine ones disagree about whether the model got better, trust the held-out ones.

*Borrowed from Poolside's multi-harness rollout practice, whose purpose is to ensure
behavior generalizes across scaffolds rather than overfitting to one. Same principle,
different axis.*

### 7.2 The rest of the stack

1. **Probe suite** (§7.1) — primary. Read it.
2. **Held-out perplexity on two sets**: weird-register text, and general modern text.
   The gap between them *is* the alien/useful tradeoff, made numeric.
3. **Loss curve vs. the `train_gpt_simple.py` control** — the architecture claim. One
   number to anchor against: the control reaches 3.28 val loss in 3250 steps on
   FineWeb10B at 12L×768.
4. **Ablation comparison table** — the Phase 1 deliverable. Phase 0.5 fills most of it in.
5. Optionally, late: a scored instruction-following set to check steerability survived
   as a *capability*, beyond what the instruction-shaped probe family in §7.1 shows you
   qualitatively from day one.

---

## 8. Milestones

| # | Milestone | Gate |
|---|---|---|
| 0 | ~~Gutenberg tokens on disk, round-trip test, `manifest.jsonl`~~ **DONE** — 3,431 books → 3,178 trained → 462 M tokens | ✅ |
| 1 | ~~baseline trains on own data; checkpointing, sampling, sample log running~~ **DONE 2026-08-08** — 3250 steps, val_loss 2.80559, 3.07 h on Modal A100. Ran on rented A100, not the 4090. | ✅ |
| 2 | `kda_mini.py` passes sequential ↔ chunkwise ↔ `fla` at 1e-4 | week 2–3 |
| 3 | **First K3-mini checkpoint**: ~120M active params, ~2B tokens, 3 readable paragraphs. It will be bad. It will be yours, architecture and data both. | **week 3** |
| 4 | All swaps (a)–(h) benchmarked against control | week 4–6 |
| 5 | Sefaria + hand-curated register slice integrated | week 4–8 |
| 6 | Phase 1 mix ablations complete; mix ratio chosen | week 8–10 |
| 7 | **Funding proposal written** — real curves, real prose | week 10 |
| 8 | Own tokenizer trained on final mix (64K → pad 65536) | after 6 |
| 9 | Flagship run | on funding |

**Explicitly do not yet:** train the real tokenizer, build distributed training, or
write the proposal. All of those get cheaper and better after Milestone 3.

*Amended 2026-08-08:* "do not rent GPUs" is retired. It meant "don't rent before
you have something to run"; Milestone 1 needed a GPU and the laptop was never one.
Milestone 1 cost ~4 GPU-hours total including ~20 diagnostic L4 runs.

*Also amended:* Milestone 1's samples showed the **register share (8.7%) and the
missing general-text slice (0%) are the binding constraints, not model size.** The
mixing dataloader that Phase 1's A/B/C ablation depends on is still unbuilt, which
makes it the highest-value next build — ahead of `kda_mini.py`.

---

## 9. Open decisions

**Resolved in this revision:**

- ~~Which modded-nanogpt to baseline against~~ → `records/track_3_optimization/train_gpt_simple.py`.
- ~~Whether the §3.1 constants are trustworthy~~ → verified against the primary report.
- ~~Shard format~~ → 256×int32 header, magic 20240520, uint16 payload.

**Still open:**

- **Corpus size target.** Determines flagship active params. Unknown until Gutenberg +
  Sefaria are on disk and the OCR question is answered.
- **Bilingual interleaving format** for Sefaria. Shapes model behavior; decide before
  bulk ingestion.
- **Whether the synthetic register-rendering slice (§5.3) is worth building.** Now has a
  cheap gate in front of it: hand-label a few hundred examples and try to calibrate the
  taste judge against them. If it can't be calibrated, don't build the slice. Only then
  test at ablation scale, before committing budget.
- **Sparsity: 8-of-64 vs 8-of-128.** Higher sparsity is more faithful to K3 and more
  interesting as a claim; lower is easier to keep balanced and to debug. Note K3's own
  §2.3 says extreme sparsity is precisely what *forced* the RMSNorm, SiTU-GLU and QB
  stabilizers — so at low sparsity you are testing those components in the regime where
  they matter least. That is an argument for 8-of-128 if you can afford the debugging.
- **Keep `bias=True` from the blueprint, or drop biases?** Interacts with the Muon/AdamW
  parameter split. Decide before swap (e).
- **Context length ambition.** 4K/8K is sufficient, and 8K matches K3's actual pretrain
  floor. But KDA + NoPE makes long-context extension unusually cheap, and "a model that
  holds a whole novel in mind" is thematically apt. Hold in reserve.
- ~~**Windows vs WSL2 vs Linux on the 4090 box.**~~ **RESOLVED 2026-08-08** — moot.
  Training happens on rented Modal A100s; the 4090 was never used and the laptop
  never could be. Nothing runs locally. Original note retained below for context:
  WSL2 was the
  recommendation: it gets NCCL, Triton and inductor working without per-tool workarounds.

---

## Appendix A — Glossary

*For re-onboarding after a pre-RoPE hiatus. Rough dependency order.*

**Attention lineage**

- **KV cache** — cached keys/values so decoding doesn't recompute history. Grows
  linearly with context; at long context it *is* the serving cost. Most architecture
  news since 2023 is secretly about shrinking it.
- **GQA** — multiple query heads share one KV head. First major cache reduction.
- **MLA (Multi-head Latent Attention)** — DeepSeek-V2. Project each token to one small
  latent vector, cache *that*, reconstruct K/V via learned up-projections. Large cache
  reduction, full attention quality.
- **Gated MLA** — MLA + input-dependent **full-rank** sigmoid gate on output channels,
  letting each token modulate what it reads from global attention.
- **NoPE** — layers with no positional encoding at all. Works in hybrids where
  recurrent layers carry position implicitly. In K3, *all* MLA layers are NoPE, and the
  payoff is 1M-token extrapolation with no positional surgery whatsoever.
- **Linear attention** — replace the growing cache with a fixed-size state matrix
  **S** (d_k × d_v): write with `k vᵀ`, read with `q`. O(1) memory per token.
- **Delta rule / DeltaNet** — the fix that made linear attention competitive. Before
  writing, *erase* what the state currently returns for key `k`:
  `S ← (I − β k kᵀ) S + β k vᵀ`. Associative memory with overwrite, not blind
  accumulation. A fixed-size memory **must** have an eviction policy.
- **Gated DeltaNet / Mamba-2** — add a decay gate α so the state leaks. Mamba-2's
  state-space duality result unified the SSM and linear-attention literatures.
- **KDA (Kimi Delta Attention)** — gated DeltaNet with *channel-wise* decay (per-dim α,
  not scalar), ShortConv on q/k/v, L2-normed q/k, and in K3: sigmoid-bounded log-decay
  (g_min = −5), head-wise RMSNorm on the recurrent output, and a full-rank output gate.
- **Chunkwise parallel form** — split the sequence into chunks; exact parallel
  attention *within* a chunk, recurrent state carried *across* chunks. C=N recovers
  full attention, C=1 recovers pure linear attention. Cost splits into a fixed 2Ld²
  state term and a growing 2LCd diagonal term. All modern linear-attention kernels
  work this way. K3 subdivides chunks into 16-token tiles; the whole point of
  `g_min = −5` is that the reciprocal rescaling then fits in bf16 so *every* tile,
  diagonal included, is a dense Tensor Core matmul.
- **Hybrid attention** — pure linear attention loses precise long-range retrieval, so
  interleave. K3: 3 KDA : 1 Gated MLA, plus a final MLA (69 + 24 over 93 layers).
- **AttnRes (Attention Residuals)** — replace the additive residual stream with
  *attention over depth*: each layer computes softmax weights over previous layers'
  outputs, with RMSNorm on the keys so large-magnitude layers don't dominate. Block
  variant attends over ~8 block summaries to bound memory (O(Ld) → O(Nd)) and to bound
  the inference-time state. ~25% training-efficiency gain at <2% cost. The token
  embedding is always an available source.

**MoE**

- **Routed vs shared experts** — a few always-on shared experts for common
  transformations; many small routed experts for specialization. *Active* params =
  shared + k routed; *total* = all of them. K3 fixes shared = 2 per layer.
- **Aux-loss-free balancing** — DeepSeek-V3: add a per-expert bias to router scores
  used only for top-k selection (not for mixture weights), nudged by observed load.
  No gradient interference.
- **Quantile Balancing (QB)** — K3's replacement: set each bias directly from the
  score quantile producing exactly the target load `q = mk/n`. Route with Top-(k+1) to
  read each token's cutoff, take the `(1 − k/n)`-quantile of the margins, mean-center,
  apply next step. Closed-form, no step-size hyperparameter, equilibrates in a few
  steps even at ~10³ experts. Estimated in practice from per-expert margin histograms
  (additive across ranks, so one all-reduce of bin counts gives the true global-batch
  quantile up to bin width). Derived as the exact dual of an optimal-assignment LP; the
  sign-nudge is just SignSGD on the same objective.
- **LatentMoE** — routed experts operate in a compressed latent space (0.5× hidden in
  K3): project down once, run tiny experts, project up once. **Shared experts stay at
  full width** (§3.4).
- **SiTU-GLU** — SwiGLU's two factors are both unbounded, and coincident large
  activations blow up low-precision training. SiTU soft-caps both branches with
  `β·tanh(x/β)`. Matches SwiGLU near the origin, bounded far from it, with
  `|f| ≤ β₁β₂`.

**Training**

- **Muon** — post-AdamW optimizer for matrix params: orthogonalize the momentum matrix
  (Newton–Schulz) and use that as the update. Keep AdamW for embeddings/norms/scalars.
  **MuonClip/QK-clip** (K2) for attention-logit growth; **Per-Head Muon** (K3)
  partitions Q/K/V momentum along the head dimension and orthogonalizes each head's
  block separately, so large-gradient heads stop dominating the shared update
  direction. Slightly *cheaper* too — Newton–Schulz on tall per-head blocks beats the
  full matrix.
- **WSD vs cosine** — WSD holds LR flat then decays fast. K3 found cosine wins *when
  each schedule gets its own independent scaling-law search*, and attributes the
  contrary literature to reusing one schedule's HPs for both.
- **TPP** — tokens per parameter. A tuned quantity, not a constant. See §5.6.
- **MTP** — extra head predicting token t+2 alongside t+1. Small pretraining gain;
  doubles as a speculative-decoding draft model. K3 uses 1 MTP layer.
- **QAT / MXFP4 / MXFP8** — quantization-aware training in low precision so the model
  adapts to the noise. MX = 4/8-bit floats with per-32-element shared scales.
- **Cooldown / annealing** — final LR-decay phase upweighting the highest-quality data
  and long documents. Disproportionate share of final quality.
- **MFU** — model FLOPs utilization. 40–50% is good for dense bf16; MoE + linear
  attention hybrids land lower until kernels are tuned. Cost estimates scale inversely.

*Skip the post-training literature (RLVR, GRPO, on-policy distillation, reward models)
until there is a base model worth post-training. The K3 report's §4 is ~40% of its
length and none of it matters yet.*

---

## Appendix B — References and repos

**Papers / reports**
- **Kimi K3 technical report** — primary architecture source. On disk at
  `C:\Users\keppy\Downloads\k3_tech_report.pdf` (25pp). Move it into the repo.
  Sections that matter: §2.1.1 KDA, §2.1.2 Gated MLA, §2.2 AttnRes, §2.3 Stable
  LatentMoE, §2.3.3 + Fig. 5 QB, §2.5 Per-Head Muon, §3.2 scaling law, §3.3 recipe.
  Table 1 is the K2↔K3 architectural diff. §4–§7 are post-training, infra and evals —
  not needed yet.
  *Extract text with `pdftotext -layout` (present in Git-for-Windows at
  `/mingw64/bin`); `pdftoppm`/poppler is not installed.*
- Kimi Linear (arXiv 2510.26692) — KDA's origin, and the closest runnable ancestor.
  Needed for the UT transform and the full chunkwise derivation, which K3 §2.1.1
  explicitly defers to it.
- Kimi K2 (arXiv 2507.20534) — MuonClip, weight clipping, data rephrasing recipe
- DeepSeek-V2 (MLA), DeepSeek-V3 (aux-loss-free balancing)
- Gated Delta Networks; Parallelizing Linear Transformers with the Delta Rule
- Muennighoff et al., data-constrained scaling (the ~4-epoch result)

**Code**
- **`records/track_3_optimization/train_gpt_simple.py`** — the blueprint (§3.5). In
  this repo already.
- `MoonshotAI/Kimi-K3` — download config + `*.py` only, **not** the ~1.5–2 TB of weights:
  `hf download moonshotai/Kimi-K3 --include "*.json" "*.py" "*.txt" "*.model" --exclude "*.safetensors" --local-dir ./kimi-k3-ref`
- `fla-org/flash-linear-attention` — KDA kernels, chunkwise reference, KCP context
  parallelism (PR #691)
- `MoonshotAI/FlashKDA` — CUTLASS chunkwise kernel
- `datatrove` — dedup and filtering at scale
- **Kimi Linear 48B-A3B** — the weights actually worth downloading; fits the DGX Spark
  and teaches KDA behavior hands-on

**Tooling note:** the HF CLI is now `hf`, not the deprecated `huggingface-cli`; console
scripts land in `~/.local/bin`, which often isn't on PATH.

---

## Appendix C — Things not to do

- Do not build the corpus for six months before training anything.
- Do not construct K3-mini as a monolith.
- Do not baseline against 2019-era GPT-2.
- **Do not baseline against `train_gpt.py` at HEAD.** It looks like the obvious choice
  and it is a trap. It is 2278 lines of hyper-tuned speedrun artifact: bigram hash
  embeddings over ~377K rows, MUDD, XSA, a DC-attention Triton kernel, YaRN window-size
  schedules, an MTP weight schedule, sparse gradient comms, and an optimizer that splits
  parameters into uint16 mantissa halves for extra precision. Its fast path asserts
  `world_size == 8`, its 1285-step schedule is co-tuned for FineWeb10B on 8×H100, and at
  `world_size=1` it configures validation at a 262 144-token sequence length. Change the
  data distribution and dozens of co-tuned tricks are invalidated at once with no way to
  attribute the difference. It is also the opposite of "get Muon working in a *simple*
  dense setting first." Use `train_gpt_simple.py`.
- Do not copy code from secondary explainers without checking it against the report
  (see §3.4 — the trap is real and now confirmed).
- Do not rephrase the register data.
- Do not touch shadow libraries.
- Do not try to out-curate FineWeb-Edu on general web text.
- Do not write the funding proposal before Phase 1 exists.
- Do not skip the general-text slice in pursuit of purity.
- Do not try to train on native Windows. NCCL is Linux-only.
