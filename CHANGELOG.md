# Changelog

K3-mini — a literary model in the weird & eerie register. See [PLAN.md](PLAN.md)
for goals and [MODAL.md](MODAL.md) for cloud runs.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). Dates are
the day work landed. Entries note *why* where the reason is not obvious, because
several decisions here were made against a specific failure that is not visible
from the code alone.

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
