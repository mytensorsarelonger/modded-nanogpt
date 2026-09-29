# PAUSE NOTE — general-text slice (FineWeb-EDU), 2026-09-27

Working note, not repo documentation. Task: add the general-text slice per
PLAN.md §5.1.2 ("the cheap one and should go first"). PAUSED mid-task for
research on the approach.

**RESOLVED 2026-09-28** (see CHANGELOG same day): decision was **FineWeb-EDU**;
verification re-run against final bytes, manifest keys added, shards uploaded,
`::verify` ok=true, shakeout run launched. Steps 1–6 below are DONE; step 7
(no preset changes) remains deliberately undone per the Phase 1 plan. The pitfall
notes below stay — they cost real time and the HF-Xet finalizer will bite again
on the next bulk download.

## Done

- Downloaded 1 chunk of `kjj0/finewebedu10B-gpt2` (already GPT-2-tokenized in
  the exact llm.c shard format) via `data/cached_finewebedu10B.py 1`:
  train_000001 (100M tokens) + val_000000 (100M tokens). Downloader exited 0.
- Copies in place (uncommitted, untracked):
  - `data/shards/by_slice/general_train_000.bin`
  - `data/shards/general_val_000.bin` (sits beside `gutenberg_val_000.bin`)
- Header/tokenizer checks PASSED on the first-settled content: magic 20240520,
  version 1, header token count = payload = 100,000,000, header[3:] zero,
  max token id 50256 (≤ 50304 vocab), EOT 50256 present (~98k docs train,
  ~96k val). Tokenizer agreement holds.
- `modal_app.py::verify_data` EXTENDED (uncommitted): now checks the slice pool
  (`slice_shards` under `by_slice/`) and `slice_val_shards`, not just
  train/val_shards; summary also reports `slice_pool_tokens`. Before this the
  whole by_slice pool was uploaded and consumed with no checksum ever run.
- Modal CLI quirk: must run as `PYTHONPATH= modal ...` — the Hermes session
  exports a PYTHONPATH whose `watchfiles` shadows the working one and breaks
  the modal CLI import.

## Stable final hashes (terminal-side only, 3 rounds sha256sum + certutil)

- train: `56c89ff62b84a8645ad7d9b3d2fbe33cb588e0d774b7b966e263fd1b945bd39a`
- val:   `64682bb1bcc5437ac1cce0efd9dcbee2fd78877b74159e21970f6b9cba734c97`
- mtimes: 10:28:50, 200,001,024 bytes each.

## Pitfall hit (cost real time — do not repeat)

hf_hub 1.x Xet finalization kept rewriting the .bin files for minutes after
they reached full size. Hashes oscillated between two complete, valid-looking
byte-sets (identical headers, identical token stats) while reads raced the
writer. Two mitigations, both now proven:
1. Wait for the downloader process to EXIT before hashing anything.
2. Hash from terminal-side python / sha256sum / certutil. The Hermes
   execute_code kernel kept seeing stale content the terminal never saw.

## Not done yet (in order)

1. Re-run the header/token-stat verification against the FINAL bytes (the
   pass above ran on pre-finalization content; hashes since settled).
2. Update `data/shards/manifest.json` — add ONLY:
   `slice_shards.general`, `slice_pool_tokens.general` (100,000,000),
   `slices.general`, and new key `slice_val_shards.general`.
   Do NOT touch `train_shards` / `val_shards` / `total_train_tokens` /
   `total_val_tokens` — those describe the control-arm corpus and headline
   val, which stay Gutenberg-only (val glob in train_baseline.py is
   hardcoded to `gutenberg_val_*.bin`, so Milestone 1 comparability is
   preserved either way).
3. Sizing rationale to record: 1 shard = 100M → general = 17.8% of the new
   562.3M-token pool (§5.1 budget 10–20%).
4. Upload to volume `k3mini-shards`: `general_train_000.bin` → `/by_slice/`,
   `general_val_000.bin` → `/`, updated `manifest.json` → `/`.
5. `PYTHONPATH= modal run modal_app.py::verify` (checksums on) — authoritative
   contract check; now covers the slice pool.
6. CHANGELOG entry (the why) + PLAN.md status line; commit.
7. Deliberately NOT done: no MixSchedule preset changes. A/B/C get rewritten
   over the full slice taxonomy as a Phase 1 design decision (§4.2 note in
   mixing.py). Shards with zero weight are harmless until then.

## Open research question (why we paused)

Whether FineWeb-EDU is the right general-text slice at all (EDU vs plain
FineWeb, quality fit for "syntactic/logical scaffolding", and how a modern-web
slice interacts with the §5.3 inverted-rephrasing plan) — user is researching
before the manifest commit makes the slice official.
