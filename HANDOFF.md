# Review handoff

**What this is.** Data pipeline + control-arm training script for a from-scratch
literary LM (see [PLAN.md](PLAN.md) for why; you don't need to read it to review).
~3,900 lines across 12 files. Corpus is built: 3,431 Gutenberg books → 3,178 in
training → 462,267,303 tokens in 5 shards.

**The baseline has now trained end to end**: 3,250 steps, val_loss 2.80559,
3.07 h on Modal A100-SXM4 at ~156 k tok/s / ~37% MFU, curve smooth throughout.
Resume is verified in the exact configuration used (torch 2.13, compile on,
against a real long-run checkpoint): `val@1000` restored **exactly**, 125 resumed
steps within 2e-5. Total spend to here ~4 GPU-hours including ~20 diagnostic runs.

One finding shaped the current state: `torch.compile` on **torch 2.10.0** NaN'd
the weights on the second optimizer update. Isolated by elimination (not fused
AdamW, not `muon_update`'s own compile, not a stale inductor cache), then
resolved by pinning **torch 2.13.0**, which compiles and trains correctly. The
control arm is ~4.4 h on A100-40GB at 41,283 tok/s.

**The one framing thing that changes how you review it.** `train_baseline.py` is a
*control arm*, adapted from `records/track_3_optimization/train_gpt_simple.py`
(modded-nanogpt's maintained reference for optimizer research). Its architecture
and hyperparameters are deliberately unmodified — all changes are
infrastructure: resume, probe suite, run registry, device selection, env-driven
sizing. **A change that makes the model better is a defect here**, because the
whole value of a control is that it isn't tuned. Flag architecture/HP drift as a
bug, not an improvement.

## Review in this order — risk-ranked

1. **`data/dedup.py`**. MinHash + LSH plus a blockwise high-recall prefilter and
   exact full-shingle verification. Was the highest risk per line. I already
   shipped one silent bug here: word ids came from `np.unique(return_inverse=True)`,
   which numbers words *per document*, so no two signatures were comparable and it
   confidently reported zero duplicates. The full 3,404-document pass yields 2,626
   candidates, 24 exact edges and 20 actionable clusters; removal is never based
   on MinHash alone.
   **Independently verified 2026-08-12.** A from-scratch reimplementation sharing
   no code paths — blake2b word hashes (vs crc32), splitmix64 multiply-shift
   MinHash over the *full* shingle universe (vs modular-Mersenne over the
   `h % 64` subsample), different seed — plus a title-token recall oracle and a
   small-doc cardinality sweep confirmed **exactly the same 24 exact-Jaccard
   edges** on the real corpus, zero disagreements either way. Cluster selection
   was audited on the real output (every drop is directly adjacent to its
   recorded representative; no representative dropped). Offset-invariance and
   threshold behavior were probed on real books: J=0.97 front-matter shift and
   J=0.85 caught, J=0.70 kept. The object-dtype Mersenne math matches a
   pure-Python bigint reference on all 120 permutations.
2. **Resume path in `train_baseline.py`** — `distributed_data_generator`'s
   `start_batch` fast-forward must stay in exact lockstep with the live loop's
   position arithmetic. If they diverge, a resumed run reads different data with no
   error. The shared recurrence and resume metadata invariants now have focused
   tests, and a full 3,250-step run plus a resume from its step-1000 checkpoint
   have now both completed on A100 (`val@1000` restored exactly).
3. **Setup block in `train_baseline.py`** (~lines 300–340) and `save/load_checkpoint`.
   These CUDA branches have now executed on L4 without error. The earlier self-recursive
   `device_name()` defect is fixed and guarded by the Modal preflight; still
   scrutinize fused AdamW, Gloo CUDA collectives, bf16, compile, and rank-local
   checkpoint sidecars.
4. **`modal_app.py` + [MODAL.md](MODAL.md) §8.** Local Modal API/CLI checks now
   pass, both imported helper modules are in the image, resume paths are
   constrained to the runs Volume, and git provenance is forwarded to the
   trainer. The remaining
   nested data Volume mount, Gloo-on-CUDA with `device_id=`, and the shard
   round-trip are all confirmed working against the real service, and a 3-hour
   detached run survived the local process dying. The compile NaN is resolved
   (torch 2.13.0). No known open item here.
5. **`data/quality_filter.py` thresholds.** Empirically tuned by inspecting a
   200-book sample, then applied to 3,431 books; the borderline cases still need
   a human policy pass at the new scale. It drops 177 documents after dedup. Check
   `python data/quality_filter.py --report` output rather than the constants.

## Skip these

- Architecture in `train_baseline.py` (verbatim from the reference; see above).
- `data/download_gutenberg.py` fetch logic — exercised on 3,431 real downloads.
- `data/shard_writer.py` — small, and has a round-trip test.
- PLAN.md / CHANGELOG.md prose.

## Verify my claims yourself

```bash
python data/shard_writer.py                      # shard format round-trip
python config.py                                 # config <-> shard agreement, hashes
python data/dedup.py                             # 20 clusters / 20 dupes, ~4 min
python data/quality_filter.py --report --top 20  # filter decisions + borderline cases
python data/slices.py                            # backbone/register split
python smoke_test.py --steps 3 --batch-size 8192 --mbs 1   # end-to-end, CPU, ~4 min
python -c "import modal_app; print('modal_app preflight OK')"
modal run modal_app.py::train --help             # confirms resume/provenance CLI
```

Colab L4 gate — **secondary**; Modal is the primary path now (MODAL.md):

```bash
python data/prepare_colab_smoke.py --copy --zip
```

Upload `colab_smoke_bundle.zip` to Drive, open
`colab/colab_l4_smoke.ipynb`, select an L4 GPU, and run all cells. The notebook
enforces the exact 100-step smoke settings and preserves artifacts to Drive.

Resume equivalence (the claim most worth re-checking, ~8 min):

```bash
SMOKE=1 TRAIN_STEPS=6 CHECKPOINT_EVERY=3 SAMPLE_EVERY=99 BATCH_SIZE=8192 MBS=1 VAL_TOKENS=8192 python train_baseline.py
```

then rerun with `RESUME=runs/<id>/ckpt_00003.pt`. **Check the invariant, not a
number:** the resumed run must reach the *same* `val_loss` the uninterrupted run
reported at the same step, and `final_loader_state` in `runs/index.jsonl` must
match exactly. (An earlier version of this file named a specific expected loss.
That was wrong twice over — it drifts with any sizing change, and it was recorded
while init was unseeded, so it was never reproducible in the first place.)

Cloud equivalent, ~11 min on L4, using `STOP_AFTER` so a 3250-step checkpoint can
be resume-tested without paying for a full run:

```bash
modal run modal_app.py::train --resume "<run_id>/ckpt_01000.pt" --stop-after 1125
```

Cloud recovery uses paths relative to the `k3mini-runs` Volume:

```bash
modal run --detach modal_app.py::train --resume auto --resume-dir <run_id>
modal run --detach modal_app.py::train --resume <run_id>/ckpt_01000.pt
```

## Already known — please don't file these

Full list in [CHANGELOG.md](CHANGELOG.md) under *Known gaps*. The main ones:
register slice is 8.7% (under target) and the mixing dataloader to upweight it is
unbuilt; `pub_year` is null for all documents because Gutendex cannot supply it;
`slice_overrides.json` is empty so E. F. Benson's social comedies are mis-tagged as
register; no `kda_mini.py` yet; no tier-2 model-based quality classifier.

Sampling has no KV cache (O(n²) generation) — known, and deliberate for now.

## Most useful thing you could tell me

The original two questions are answered (resume verified against a real long-run
checkpoint; CUDA paths survived a 3-hour A100 run), and as of 2026-08-12 so is
the third: **`data/dedup.py` is verified correct** by independent
reimplementation (see item 1 above). What remains is editorial, not engineering:

1. **Does the quality filter's threshold set still make sense at 3,431 books?**
   It was tuned by eye on 200 and now drops 177 documents. That needs a human
   policy judgement, not a code review.
2. **Is the register/backbone slice boundary defensible?** Author-level tagging
   puts all ~52 E. F. Benson books in `register`, most of which are social
   comedies rather than ghost stories. `slice_overrides.json` exists for exactly
   this and is empty.

Both are the calls I have been making by proxy from §5.2.3's author list, and
they set the corpus composition that Phase 1's whole experiment varies.

One output of the dedup verification feeds that editorial pass: at the 0.80
threshold the corpus **retains same-work edition pairs just below it** — *A
Child's Garden of Verses* twice (J=0.77), *Treasure Island* twice (0.72), *The
Secret Garden* twice (0.68), *Little Women* vs its full two-part edition (0.69),
Frankenstein's 1818 vs 1831 texts (0.74), two translations of Plato's *Republic*
(0.72), and *Carmilla* both standalone and inside *In a Glass Darkly* v.3
(0.73). Whether any of these should also collapse is a threshold policy call,
not a dedup defect. (The same sweep surfaced ~2 KB LibriVox audiobook-metadata
stubs with high mutual boilerplate similarity; all are dropped downstream by the
quality filter's `audiobook_metadata` rule, so nothing reaches training.)
