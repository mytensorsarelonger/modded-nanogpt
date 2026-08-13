# Changelog

K3-mini — a literary model in the weird & eerie register. See [PLAN.md](PLAN.md)
for goals and [MODAL.md](MODAL.md) for cloud runs.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). Dates are
the day work landed. Entries note *why* where the reason is not obvious, because
several decisions here were made against a specific failure that is not visible
from the code alone.

---

## 2026-08-12 — PR #1 merge-blocker repair

- **Unified the executable runtime on PyTorch 2.13.0.** The Linux requirements,
  Modal image, Colab notebook, and operator documentation now agree. The Colab
  gate also uses the control arm's real 524,288-token batch instead of the
  65,536-token diagnostic setting that diverged under the fixed learning rates.
- **Made runtime identity complete.** `INIT_SEED`, `ADAMW_FUSED`,
  `MUON_COMPILE`, `VAL_EVERY`, and the already-resolved sizing/framework values
  all participate in `config_hash` and are recorded in the run registry. A
  numerically different launch can no longer collide with or resume under the
  same identity.
- **Made early-stop outcomes truthful.** `STOP_AFTER` records the actual
  completed step, actual corpus coverage, the step of the last validation, and
  whether the run stopped early. Non-finite losses are stored as JSON `null`
  with an explicit flag rather than the non-standard bare `NaN` token.
- **Tracked the intended experiment registry.** Replaced four stale local CPU
  records with the fetched 19-run Modal registry, including the A100 milestone
  receipt (`21c92807-418e-49ce-a78b-566b376f0914`). Historical non-finite values
  were normalized to strict JSON without hiding that they were non-finite.
- Added runtime-contract and provenance regression coverage; **31 local tests
  pass**, along with syntax compilation and shard/config validation.
- **Re-ran the repaired gate on a real L4.** Run
  `eadfda3f-8ae6-40af-a26a-e6eca74a4206` completed 100/100 steps under Torch
  2.13.0+cu130 with the 524,288-token batch, finite val_loss 4.37255, 16 probe
  generations, and both format-3 checkpoint files committed. Cold-cache wall
  time was 27.5 minutes, so the function timeout is now 45 minutes rather than
  leaving only ~2.5 minutes of failure-prone margin.

---

## 2026-08-09 — mixing dataloader (PLAN.md §5.1.1)

Phase 1's A/B/C ablation varies the slice mix; until now the mix was whatever the
corpus happened to contain and could not be varied at all. It can now.

### Added

- **`data/mixing.py`** — the policy half, pure arithmetic, no torch or I/O.
  `MixSchedule` is piecewise-linear keyframes over training progress, so a constant
  ratio is one keyframe and §4.3's "flat then ramp through cooldown" is three.
  Presets: `corpus` (8.7% as-is), `cooldown-ramp`, and `mix-a/b/c` for §4.2.
- **`data/mix_loader.py`** — the I/O half. Per-slice pools, contiguous reads within
  a slice, rows scattered so no microbatch is 100% one slice.
- **Per-slice shards**, written *in addition to* the combined stream into
  `data/shards/by_slice/`. Costs ~890 MB of duplicate storage and buys the
  guarantee that building mixing cannot perturb the control arm.
- 39 tests (23 policy, 16 loader).

### Design decisions worth keeping

- **No RNG.** Allocation is a deterministic function of step index, so a resumed
  run recovers its per-slice cursors by replaying scalar arithmetic instead of
  needing generator state checkpointed. This is why resume still works.
- **Driftless allocation.** A fractional carry per slice, largest-remainder within
  each step. Plain rounding of `w × N` biases the same direction forever when
  `w × N` sits just below .5 — at 0.7% of 64 sequences it would deliver *nothing*,
  ever. Verified on real data: target 0.800/0.200 → realised **0.8001/0.1999**.
- **Contiguous reads within a slice**, matching the control's target-offset
  convention. Reading each sequence separately would change that convention at
  every sequence boundary and confound every A/B/C comparison with an unrelated
  change.
- **Progress keyed to `train_steps`, not the early-exit step**, mirroring the LR
  schedule, so `STOP_AFTER` cannot change the mix a given step sees.

### Verified on GPU

- Mixing trains: 6 steps at mix-a, 10.826 → 7.103 → 6.076.
- **Resume under mixing is faithful** — `val@3` restored exactly (7.10260),
  `train_loss@6` identical, `val@6` within 1e-5 (the established noise floor).
- Registry records `mix`, realised fractions, epochs per slice, and pool wraps, so
  an ablation row states what it actually trained on rather than what was intended.

### Fixed

- **Windows paths in the manifest broke the loader on Linux.** `slice_shards`
  recorded backslash-separated Windows paths, and `Path(...).name` on POSIX returns the
  *entire string* because backslash is not a separator — so no shard was found.
  The loader now globs its own naming convention and ignores recorded paths, and
  the tokenizer also records portable basenames. Third instance of
  Windows-authored data consumed on Linux; see the 2026-08-06 lesson.
- **The checkpoint guard rejected mixed runs** (`loader batches=6, got None`) —
  correctly, since the mixing path never populated loader state. It now mirrors the
  control's `state` dict contract. Good example of a guard earning its place.

### Known gaps

- **Per-slice shards are flat-globbed, so `by_slice/` must not gain unrelated
  `*_train_*.bin` files.** Naming collision is the hazard that already bit once:
  `gutenberg_train_*.bin` would have matched `gutenberg_train_backbone_000.bin`,
  silently making the control read the corpus three times.
- `mix-a/b/c` assume backbone+register only. §4.2's real design includes general
  modern text and scripture, both at **0%**, so those presets are not yet the
  ablation §4.2 describes.
- Upweighting a small pool buys *repetition*, not diversity: mix-c (60% register)
  over a 40 M-token pool means ~26 epochs on 371 books in a full run while the
  backbone sees ~1.5. `epochs_per_slice` is reported for exactly this reason.

---

## 2026-08-08 (later) — training-dynamics changes from arXiv 2606.06533

Read Biderman, Khan, Mireshghallah, Arnett, Barez & Saphra, *"Position: Don't Just
'Fix it in Post': A Science of AI Must Study Training Dynamics"* (ICML 2026). A
**position paper, not a method paper** — it supplies framing and citations, no
algorithms. Three things taken from it:

- **Seed-band gate before any Phase 0.5 swap is believed** (PLAN §4.1). Every swap
  produces a val_loss delta; a delta smaller than run-to-run seed variance is a
  draw, not a result. We know the band is nonzero — 0.0148 between two identical
  6-step runs before `init_seed` existed — and unmeasured at 3250 steps. Their §2.3
  argues treating seed variation as noise-to-average-over is a choice that must be
  made deliberately, and that variation near thresholds produces distinct
  generalization clusters in different loss basins. Filed for discussion on PR #1
  (repo has issues disabled).
- **The mixing dataloader must take a schedule, not a static ratio** (new PLAN
  §5.1.1). Later data has larger influence on final behaviour, which is the
  mechanism behind §4.3's cooldown upweighting. "15% register" and "8% rising to
  25% through cooldown" are different interventions and a ratio-only loader cannot
  express the second. **This also surfaced that §4.2's A/B/C mix ablation is
  under-specified**: "60/40" does not say whether the ratio is constant or
  scheduled, so as written runs A/B/C would not be comparable to each other.
- **`checkpoint_every` 1000 → 250**, matching `sample_every`. Biderman et al. 2023a
  found intermediate checkpoints of one run predict final memorization *better than
  smaller fully-trained models do*. That partially undercuts the small-proxy-run
  logic Phase 1 leans on, and dense checkpoints are the precondition for any
  retrospective dynamics work. 13 × ~1.4 GB per run is trivial on a Volume.

The paper also supplies the citation for Milestone 1's own finding: models "can
only systematically compose concepts which appear in diverse contexts during
training" (Allen-Zhu & Li 2025; Okawa 2023; Chang 2025). Our 0% general-text slice
leaving no expository mode was predicted, not novel.

*For method, follow its citations rather than the paper:* Pythia (Biderman 2023),
Tigges 2024 (circuit consistency across checkpoints), Li 2025 (hundreds of small
models).

---

## 2026-08-08 — Milestone 1 complete: baseline trained end to end

First real training run. **3,250 steps, 462 M-token corpus, final val_loss
2.80559, 3.07 h on A100-SXM4** at ~156 k tok/s / ~37% MFU. Curve smooth
throughout, no instability. W&B:
`wandb.ai/stemuli-studios/k3mini/runs/21c92807-418e-49ce-a78b-566b376f0914`.

Milestone 1's gate was "a dense baseline loss curve and a sample log, on your
data, that you trust." Both exist.

### Verified

- **Resume works in the configuration that will actually be used** — torch 2.13,
  compile on, resuming a real long-run checkpoint. `val@1000` restored
  **exactly** (3.32969 → 3.32969); 125 resumed steps landed within 2e-5 of the
  original trajectory (~1.6e-7/step, consistent with GPU reduction noise).
  `STOP_AFTER` made this an 11-minute test instead of a 2-hour one by exiting
  early without touching `train_steps`, so the LR schedule stayed intact.

### Added

- `STOP_AFTER` — exit at a given step while leaving `train_steps` (and therefore
  the LR schedule and the resume guard's sizing check) unchanged. Exists because
  verifying a resume from a 3250-step checkpoint otherwise costs a full run.

### Read the samples — the finding that matters

The probe suite (§7.1) earned its keep on first real use. On the `mundane`
family at T=0.7, across the run:

| step | output |
|---|---|
| 250 | footnote debris (`[Footnote 768: _Lord Ellenborough._]`) |
| 1000 | contentless dialogue loop (`"is a sort of a sort of a room"`) |
| 2000 | **best of the run** — actually about kitchens, grammatical |
| 3250 | total collapse: `cooks and cooks and cooks…` for the whole sample |

**Loss improved 3.33 → 2.81 across that window while this sample got much
worse.** Loss cannot see this; only reading can. That is the entire argument for
§7.1 existing, demonstrated.

Diagnosis, from the two-temperature design: at **T=1.0 step 3250 does not loop**,
and `cold_open` at T=0.7 does not loop either. So it is not model collapse — the
*mundane prompt specifically* is off-distribution. The corpus is ~all narrative
fiction, so asked for exposition the model has no expository mode, drops to the
lowest-entropy continuation, and with no top-k/top-p never escapes the cycle.

**The defect is not "register as costume" — it is "no register for non-narrative
prose at all."** The general-modern-text slice in §5.1 is currently **0%** of the
corpus, and §5.1 calls it "not optional… what keeps the model steerable rather
than a beautiful ghost." This probe priced that omission. The `instruction`
family corroborates: asked to "Describe a staircase in three sentences," it
produced pseudo-philology about "the number of the fantastic syllables."

**Is it eerie? No.** Competent late-Victorian pastiche, grammatical, incoherent
past a sentence or two. The register did not take at 8.7%. The one moment with
real cadence is `continuation` at T=1.0 — "he pocketed his revolver, placed his
hand on the lock, and felt for the secret lock which contained a key long since
found" — and that is the easiest case, since it was handed an eerie paragraph to
imitate. At 124 M params / 8.7% register / 3.7 epochs this is the expected
result; §8 predicted "it will be bad, it will be yours."

### Consequences for the plan

1. **Register share is the binding constraint, not model size.** 8.7% is below
   §5.1's 10–25% band and the mixing dataloader is still unbuilt. It is now the
   highest-value unbuilt thing, because it is the variable Phase 1's A/B/C
   ablation turns on.
2. **The general-text slice must exist.** 0% is why `mundane` and `instruction`
   fail.
3. **Probe suite needs top-p, or T=1.0 as the reading of record.** Pure
   multinomial on a sharpened post-cooldown model loops on OOD prompts. Keep
   T=0.7 as a degeneracy detector — which is what it just was.

### Corrected

- I claimed mid-session that "W&B was never enabled for the full run." Wrong —
  the full run logged normally. The evidence was the *resume* run's overrides,
  which lacked `WANDB=1` because I launched that one myself.
- `index.jsonl` records no W&B URL on any of 19 runs, so the registry and the
  curve are not linked. §4.0.1 wants one record pointing at everything.

---

## 2026-08-06 (later) — compile blocker resolved by torch 2.13.0

- **Bumped the Modal image from `torch==2.10.0` to `torch==2.13.0`** (cu130).
  Compile-enabled training is correct again: 10.826 → 10.209 → 8.506 → 7.299 →
  6.813, tracking the eager 2.10 curve as it should. So the NaN was a torch-2.10
  inductor regression against this model, not a defect in the training code.
- **Throughput correction.** The previously reported "51k tok/s, 32% MFU" came
  from the 2.10 *compiled* run — which was computing NaNs for most of its steps,
  so it was never a valid measurement. Measured on the clean 2.13 run instead:

  | | s/step | tok/s | TFLOP/s | A100 est. |
  |---|---|---|---|---|
  | 2.10 eager | 35.0 | 14,980 | 11.1 | ~12.2 h |
  | 2.13 compiled | 12.7 | 41,283 | 30.7 | **~4.4 h** |

  Compile is worth **2.7×**, not the 3.4× estimated from the bad number, and the
  control arm is ~4.4 h on A100-40GB rather than ~3.5 h. Comfortably inside the
  6 h function timeout.
- **The pin is chosen on Linux merit alone now.** With nothing executing locally,
  `requirements.txt`'s Windows pin constrains no result.
- *Not re-verified on 2.13:* resume equivalence was established on 2.10 eager.
  The resume logic is torch-independent (loader arithmetic, state dicts), but the
  measurement has not been repeated under compile.

---

## 2026-08-06 — first GPU contact (Modal L4)

The launch path is proven end to end and the corpus is confirmed byte-identical
in the cloud. One blocker found, which is the reason Milestone 1 has still not
produced a loss curve.

### Blocked

- **`torch.compile` produces NaN weights on the second optimizer update**
  (torch 2.10.0+cu128, L4/sm89). Eager mode trains correctly from `ln(50304)`:
  10.826 → 10.190 → 8.488 → 7.306 → 6.816. Isolated by elimination, not guessed:

  | model compile | muon compile | fused AdamW | result |
  |---|---|---|---|
  | on | on | on | NaN at update 2 |
  | on | on | off | NaN at update 2 |
  | on | off | on | NaN at update 2 |
  | off | eager | off | trains ✓ |
  | off | eager | on | trains ✓ |

  A cold inductor cache (`--cache-bust`) still NaNs, so this is not the
  poisoned-cache failure MODAL.md §7 warns about. Fused AdamW and `muon_update`'s
  own `@torch.compile` are both innocent; it is the compiled model graph.
  **Cost impact** (figures superseded — see the later entry): compile measured
  2.7×, and the control arm is ~4.4 h on A100 compiled vs ~12.2 h eager.
  **Resolved** by torch 2.13.0.

### Fixed (found by running on a GPU)

- **Wall-clock was understated ~100×.** `t0` was reset at the end of *every*
  step by the sample/checkpoint block, so `time_since_last_val` measured one step
  rather than the whole window. Reported `step_avg: 105 ms` when the truth was
  ~10,300 ms. This fed `train_time_s` in the run registry, so recorded wall-clock
  — the basis for every cost and throughput comparison in the ablation table —
  was wrong. Now shifts `t0` forward by the excluded duration instead of
  clobbering it.
- **Model init was unseeded.** The only `manual_seed` in the file was the
  sampler's, so parameter init drew from process entropy. Two identical 6-step
  runs measured **0.0148** apart in `val_loss` — a noise floor that would have sat
  underneath all eight Phase 0.5 ablations, making any effect smaller than it
  indistinguishable from the random draw. Added `config.init_seed`; two runs now
  match exactly.
- **Smoke used a batch size 8× too small**, so it exercised a different
  LR-to-batch ratio than the real run. (This was also my incorrect first
  diagnosis of the NaN — worth recording, since it looked convincing and was
  wrong. The full 524,288 batch still NaN'd.)

### Verified on GPU

- **Resume is faithful.** `final_loader_state` matched **exactly** on every
  resume (`{file_idx: 0, pos: 3145728, batches: 6}`) — the failure mode that
  would otherwise be silent. `val_loss` agrees to ~1.4e-5 (2.2e-6 relative)
  against a measured nondeterminism floor of ~1e-5. For scale, a real data or
  schedule error shows at ~1e-2, as the unseeded-init difference did.
  *Not closed:* with n=2 per group a sub-noise systematic offset cannot be ruled
  out; `torch.use_deterministic_algorithms(True)` plus
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` would settle it bitwise.
- **Gloo on CUDA with `device_id=` works** — MODAL.md had this as speculative.
- **Shards are byte-identical in the cloud**: sha256 verified per shard.
- ~~**Throughput: ~51k tok/s, 37.9 TFLOP/s, ~32% MFU on L4.**~~ **Superseded** —
  this was measured on a run that was computing NaNs. See the 2026-08-06 (later)
  entry for the corrected figures (41,283 tok/s, 30.7 TFLOP/s).
- **The NaN guard works.** Divergence exits 0 with a marker in `samples.log`
  rather than a CUDA abort — which is what a 3-hour unattended run needs.

### Lesson — run nothing locally

Roughly half the friction in this session was Windows-local-execution friction,
not project difficulty, and none of it was load-bearing work:

- **cp1252 killed three separate things.** The Gutenberg download died at 371 of
  3,000 books on a title containing `ā`; the sample log (the *primary
  instrument*, §7.1) crashed on a CJK glyph from a random-init model; and Modal's
  own CLI crashed printing a `✓` after a successful upload.
- **Git Bash MSYS2 path mangling** rewrote the remote path `/gutenberg_train_000.bin`
  into `C:/Program Files/Git/...`, silently uploading all 890 MB to the wrong
  place — twice the transfer, and ~933 MB of strays still in the Volume.
- **PowerShell parsing** broke inline Python twice, once discarding a queued
  Modal run because the block failed to parse before anything executed.
- **The torch pin had to split by platform** (2.13.0 on Windows, 2.10.0
  elsewhere) purely because the 2.10 Windows wheel reports `unsupported gloo
  device` — a constraint that exists *only* because Python was run locally.
- Locally, NCCL is unavailable, `torch.compile` needs MSVC, and the GPU is an
  8 GB laptop card that OOM-killed the first CPU smoke.

PLAN.md §2 and Appendix C already said "do not train on native Windows." The
drift was that *orchestration* stayed local even after the training moved
remote. **Standing rule from here: this machine edits files and invokes
`modal run`. It executes no Python that matters.** A corollary worth acting on:
the Windows torch pin in `requirements.txt` is now irrelevant to any result, so
the Linux/Modal pin can be chosen purely on merit.

---

## [Unreleased] — Milestone 0 complete, Milestone 1 not yet run

### Corpus (final state)

| | value |
|---|---|
| Books downloaded | 3,431 |
| In training | 3,178 (462.27 M tokens, 5 shards) |
| Held-out validation | 29 books (4.33 M tokens) |
| Excluded | 27 in-copyright, 177 tier-1 filter, 20 near-duplicate |
| Register slice | 40.19 M tokens (8.7% — under §5.1's 10–25% band) |
| Full run | 1.70 B tokens = **3.7 epochs** |

### Added

- `PLAN.md` — project plan. §3.1 verified line-by-line against the primary Kimi
  K3 report; every constant (`g_min=−5`, β₁=4, β₂=25, `N_s=2`, 3:1 + final MLA,
  cosine + 1% warmup, wd 0.1, N≈8 blocks) confirmed unchanged.
- `config.py` — single source of truth. `config_hash()`, `manifest_hash()`, and
  `validate_against_shards()`, which refuses to run when config and shards
  disagree so the file cannot quietly become decorative.
- `data/download_gutenberg.py` — bulk crawl plus **targeted fetch**
  (`--canon`, `--author`, `--ids`, `--dry-run`). Targeted mode exists because the
  bulk crawl pages Gutendex in *download-count* order, which yielded six Poe and
  zero Dunsany. Skips in-copyright editions by default.
- `data/shard_writer.py` — `.bin` writer/reader for the llm.c format (1024-byte
  header, magic `20240520`, uint16 payload) with a round-trip test.
- `data/tokenize_corpus.py` — corpus → shards. EOT-prefixed documents,
  whole-book validation holdout, uint16 ceiling guard.
- `data/backfill_manifest.py` — provenance per §6: sha256, counts, cleaning
  steps, license, Gutendex copyright status, author dates.
- `data/quality_filter.py` — tier-1 heuristic filter (§5.5.2) with `--report`.
- `data/dedup.py` — MinHash + LSH near-duplicate detection (§5.5.1).
- `data/slices.py` — backbone/register classification (§5.1) with per-document
  overrides via `data/slice_overrides.json`.
- `probes.py` — the probe suite (§7.1): four routine families, a held-out pair,
  two temperatures. The load-bearing family is `mundane`, which is what
  distinguishes register-in-the-syntax from register-as-costume.
- `train_baseline.py` — the control arm, adapted from
  `records/track_3_optimization/train_gpt_simple.py`. Now with resume, the probe
  suite, and a run registry.
- `modal_app.py` + `MODAL.md` — cloud runs: `smoke` (L4, ~$0.05) then `train`
  (A100, ~3.2 h, ~$9–15). Read MODAL.md §8 "Unverified assumptions" first.
- `data/prepare_colab_smoke.py` + `colab/colab_l4_smoke.ipynb` — a reduced,
  verified two-shard bundle and the same 100-step L4 gate for Colab Pro.
- `smoke_test.py` — thin runner over the real script.
- `CHANGELOG.md`, `.gitignore` for the corpus, shards, checkpoints.

### Chosen

- **Baseline is `records/track_3_optimization/train_gpt_simple.py`**, not
  `train_gpt.py` at HEAD. HEAD is 2,278 lines of tuned speedrun artifact — bigram
  hash embeddings, MUDD, XSA, Triton DC-attention, YaRN window schedules, sparse
  gradient comms — whose fast path asserts `world_size == 8` and whose schedule is
  co-tuned for FineWeb10B on 8×H100. As a control it would confound every result.
- **Modal over Colab.** Decisive reason: with no resume, a Colab disconnect loses
  the run, so Colab required building resume first. `modal run --detach` has no
  disconnect to survive. Resume now exists anyway, but the reproducibility and
  ~10 Phase-0.5 runs still favour Modal.
- **GPT-2 tokenizer for Phase 0** (50257 → padded 50304). Own 64K tokenizer waits
  for Milestone 8, per §5.4.

### Fixed

Bugs found and fixed during construction, kept because each is a trap that would
recur:

- **`device_name()` was self-recursive** — a global `sed` replaced
  `torch.cuda.get_device_name(device)` *inside the function that defines it*.
  Returned `"CPU"` on CPU so every local test passed; would `RecursionError` on
  the first `print0` on CUDA, before step 0.
- **MinHash reported zero duplicates.** Word ids came from
  `np.unique(return_inverse=True)`, which numbers words *per document*, so the
  same 5-gram hashed differently in different documents and no two signatures were
  comparable. Now hashes the word string with `crc32` — not `hash()`, which is
  salted per process and unstable across runs.
- **LSH alone silently missed a real duplicate.** Gutenberg 940/27681 has exact
  full-set word-5-gram Jaccard 0.878, but shared no complete 12×10 band. Candidate
  generation now unions LSH with a blockwise all-signature scan at a 0.50 floor;
  every removal is verified against the complete shingle sets.
- **The old “exact” dedup check was another MinHash estimate.** It dropped pairs
  with exact Jaccard 0.727 and 0.772 despite the configured 0.80 threshold. Sparse
  samples could also switch signature universes and cluster audiobook boilerplate.
  Sparse documents now receive cardinality-safe candidate coverage, all-dirty
  components are left to tier 1, and every loser is directly similar to its clean
  representative rather than merely connected transitively.
- **Checkpoint/sample cadence was nested inside the validation block**, so it only
  fired on steps that were also validation steps; `sample_every=100` against
  `val_step_freq=125` silently did nothing.
- **`step_avg` divided by zero on resume** — the guard tested `step > 0` while
  `last_val_step` now starts at `start_step`.
- **Dataloader had `device="cuda"` hardcoded** while the device was selected
  elsewhere; and `val_tokens` was not a multiple of `seq_len × mbs`, so
  `view(-1, seq_len)` would throw.
- **`val_tokens` was a literal** exceeding what the val shards hold, giving an
  opaque `StopIteration`. Now derived from the shard manifest and floored to the
  microbatch granule.
- **`config_hash` hashed a hand-listed subset** of seven fields, so two runs
  differing only in `muon_lr` or `adamw_betas` collided. Also `adamw_betas` had no
  type annotation, making it a plain class attribute invisible to `asdict()`.
- **Final checkpoints were one optimizer update stale.** A file labelled step 6
  contained loader/Adam state after five batches. Checkpoint labels now require
  `loader_state["batches"] == step`, and final sampling/checkpointing happens only
  after all updates (with the sample made durable before the final checkpoint).
- **Resume trusted incompatible state.** Config/data/shard hashes are now mandatory
  matches, effective runtime and framework versions participate in the config
  hash, world-size changes are refused, and rank-local Muon/RNG state is stored in
  per-rank sidecars. `RESUME=auto` searches prior run directories and prior sample
  blocks are carried forward without duplicating a boundary block.
- **The Modal image omitted `probes.py`**, guaranteeing an import failure before
  CUDA initialization. Both local trainer dependencies are now image inputs and
  preflighted; resume paths and git provenance are forwarded explicitly.
- **Multi-GPU validation sizing ignored world size.** The val count is now floored
  to `seq_len × mbs × world_size`, so the advertised 2/4/8-rank modes tile each
  rank's microbatches instead of failing an assertion.
- **PyTorch 2.10's Windows wheel could not initialize Gloo** (`unsupported gloo
  device`). Requirements now keep the Linux/Modal control on 2.10.0 and use the
  verified 2.13.0 wheel only for Windows CPU smoke tests.
- **No document separators** — books were concatenated directly, so training
  windows straddled book boundaries. Now EOT-prefixed per document.
- **Validation was a tail slice** of the concatenated stream, i.e. one book by
  whichever sorted last. Now whole books, pinned to `data/val_books.json` so the
  set does not drift as the corpus grows.
- **`tokens[:-val_size]` returned empty** when `val_size == 0` (`[:-0]` is `[:0]`).
- **UTF-8 crashes on Windows** killed the 3,000-book download at book 371 (a title
  containing `ā`) and the sample log (a CJK glyph from an untrained model). Streams
  and file writes now force UTF-8.
- **Unseeded sampler** — samples were not reproducible, folding sampling noise into
  the step-over-step comparison the instrument exists to make.
- **`runs/` in `.gitignore`** also excluded `runs/index.jsonl`; git will not
  descend into an ignored directory to un-ignore a child.
- **Stale `split` in the manifest** — filtered documents kept a `split` from an
  earlier run, claiming to be in training data when they were not.
- Download retry `continue` targeted the attempt loop, not the page loop, so three
  failures fell through to `resp.json()` on the *previous* page's response.

### Verified

Claims checked by test rather than assertion:

- **Format-3 checkpoint restore is coherent.** A fresh step-1 checkpoint records
  step/batches 1, a complete Adam common file and rank-local Muon/RNG sidecar;
  resuming it reproduces `val_loss 10.25107` and a byte-identical sample log.
- **Sampling is reproducible.** Two independent runs → byte-identical sample logs.
- **Loss at init = `ln(50304) = 10.8258`**, matching observed `10.82584` — confirms
  correct init and correct loss normalisation.
- **Dedup validated over all 3,404 eligible files:** 2,626 proposed pairs, 24 exact
  edges, 20 actionable clusters. It catches 940/27681, retains the two
  below-threshold pairs, and leaves audiobook-template stubs to tier 1.
- **Corrected shard integrity:** all six SHA-256s match; headers and payload lengths
  round-trip; all ids are below vocab; EOT counts are exactly 3,178 train / 29 val.

### Known gaps

*Refreshed 2026-08-08 — post-Milestone-1 state.*

**Blocking Phase 1:**
- **Mixing dataloader unbuilt.** Slices are tagged and measured but cannot be
  reweighted, so the ratio Phase 1's A/B/C ablation varies is not yet a knob.
  Highest-value unbuilt thing in the repo.
- **General-modern-text slice is 0%.** §5.1 budgets 10–20% and calls it "not
  optional." Milestone 1's samples showed exactly what its absence costs: no
  expository mode, repetition collapse on non-narrative prompts.
- **Register slice is 8.7%**, below §5.1's 10–25% band.

**Architecture — none of K3-mini exists yet:** no `kda_mini.py`, no LatentMoE,
no Quantile Balancing, no AttnRes, no SiTU-GLU, no MTP head. Phase 0.5 has not
started. The architectural thesis in §1 is entirely untested.

**Instrumentation:**
- Probe suite has no top-k/top-p, so a sharpened post-cooldown model loops on OOD
  prompts at T=0.7. Keep it as a degeneracy detector; read T=1.0 for prose.
- `index.jsonl` records no W&B URL, so registry and curve are not linked.
- Resume guard treats checkpoint/sample *cadence* as run identity, so you cannot
  resume while changing how often it saves. Unfixed on purpose — which fields
  constitute identity is a judgement call.
- Sampling has no KV cache (O(n²)).

**Data provenance:**
- `pub_year` is null for all 3,431 documents; Gutendex cannot supply it, so §6's
  per-work ≤1930 check cannot be automated. Matters for the hand-curated slice.
- `slice_overrides.json` empty → E. F. Benson's social comedies tagged register.
- ~933 MB of misrouted duplicate shards still sit under `C:/Program Files/Git/`
  in the `k3mini-shards` Volume (MSYS2 path bug). Harmless, untidy.
  Clean with: `modal volume rm -r k3mini-shards "C:"`
- No tier-2 model-based quality classifier (§5.5.2 tier 1 only).

**Original list follows.**


Deliberately unbuilt, in rough priority order:

1. **No GPU run has happened.** Everything is verified on CPU at ≤6 steps. All
   CUDA-only paths — bf16, `pin_memory`, `non_blocking` H2D, inductor — are
   exercised for the first time by `modal_app.py::smoke`.
2. **Register slice at 8.7%**, below §5.1's band. Measurement exists; the mixing
   dataloader to upweight it does not. Splitting shards per slice without one would
   hand you an accidental curriculum with all register material at the end.
3. **Tier-1 thresholds were tuned on the 200-book sample** and then applied to
   3,431 books. The borderline cases have not been re-inspected at the new scale.
4. **`slice_overrides.json` is empty**, so E. F. Benson's ~52 books are all tagged
   register, though most are *Mapp and Lucia* social comedies.
5. **`pub_year` is null for all 3,431 documents** — Gutendex cannot supply it, so
   §6's per-work ≤1930 rule is not automatable. Fine for Gutenberg; the register
   slice needs manual verification.
6. No `kda_mini.py` (Milestone 2). No tokenizer training run (Milestone 8).
   No second-tier model-based quality classifier (§5.5.2).
