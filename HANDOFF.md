# Review handoff

**What this is.** Data pipeline + control-arm training script for a from-scratch
literary LM (see [PLAN.md](PLAN.md) for why; you don't need to read it to review).
~3,900 lines across 12 files. Corpus is built: 3,431 Gutenberg books → 3,178 in
training → 462,267,303 tokens in 5 shards. **Nothing has run on a GPU yet** — everything
is verified on CPU at ≤6 steps, so every CUDA-only path is unexecuted code.

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
   exact full-shingle verification. Highest risk per line. I already
   shipped one silent bug here: word ids came from `np.unique(return_inverse=True)`,
   which numbers words *per document*, so no two signatures were comparable and it
   confidently reported zero duplicates. Remaining risk is the content-defined
   subsampling (`h % KEEP_MOD == 0`, must stay offset-invariant), sparse-candidate
   coverage, and the object-dtype modular arithmetic in `_signature`. The full
   3,404-document pass now yields 2,626 candidates, 24 exact edges and 20
   actionable clusters; removal is never based on MinHash alone.
2. **Resume path in `train_baseline.py`** — `distributed_data_generator`'s
   `start_batch` fast-forward must stay in exact lockstep with the live loop's
   position arithmetic. If they diverge, a resumed run reads different data with no
   error. The shared recurrence and resume metadata invariants now have focused
   tests; full-scale training remains unrun.
3. **Setup block in `train_baseline.py`** (~lines 300–340) and `save/load_checkpoint`.
   Every CUDA branch here is unexecuted. The earlier self-recursive
   `device_name()` defect is fixed and guarded by the Modal preflight; still
   scrutinize fused AdamW, Gloo CUDA collectives, bf16, compile, and rank-local
   checkpoint sidecars.
4. **`modal_app.py` + [MODAL.md](MODAL.md) §8.** Local Modal API/CLI checks now
   pass, both imported helper modules are in the image, resume paths are
   constrained to the runs Volume, and git provenance is forwarded to the
   trainer. The remaining
   first-contact risks are the nested data Volume mount and the actual CUDA
   tensor/compile/memory paths; run the L4 smoke before an A100. The pinned
   PyTorch v2.10.0 source explicitly supports Gloo on CUDA and safely binds the
   supplied CUDA `device_id`, so that API combination is no longer speculative.
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

Colab L4 gate (the no-Modal alternative):

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

then rerun with `RESUME=runs/<id>/ckpt_00003.pt`. Both must reach
`val_loss 7.42766` and produce byte-identical `samples.log`.

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

Whether the dedup and resume implementations are correct, and whether anything in
the CUDA-only paths will fail on first contact with an A100. Those are the two
places where a bug costs real money or silently corrupts a result, and they are
exactly the two places local testing could not reach.
