# AGENTS.md — entry point for any agent or harness

**Read this first, then [PLAN.md](PLAN.md).** This file is harness-agnostic: it holds
what an agent needs to not waste a day, and nothing else. It is not a second plan and
not a second changelog.

> The `AGENTS.md` **one directory up** (workspace root) is Prime Lab / verifiers
> guidance for RL environment work. It does **not** apply to this repo. Do not try to
> fit this project to those conventions.

---

## What this repo is

A long-running personal fork of `modded-nanogpt` hosting **K3-mini**: a small language
model trained from scratch whose distinguishing quality is *voice* — fluency in the
weird and eerie literary register — on a scaled-down Kimi K3 architecture (KDA +
Gated MLA hybrid, block AttnRes, Stable LatentMoE with Quantile Balancing).

**This fork will not merge upstream.** Do not open upstream PRs, do not rebase onto
upstream, do not "fix" the fork's divergence. The upstream README is preserved below
the fork banner in [README.md](README.md) on purpose: its
`records/track_3_optimization/train_gpt_simple.py` is our control arm and its
provenance matters.

## Where state lives — one job each

| file | authority over |
|---|---|
| [PLAN.md](PLAN.md) | **intent.** Goals, architecture spec, data plan, milestones, open decisions. |
| [CHANGELOG.md](CHANGELOG.md) | **history.** What happened, when, and *why*. Includes a *Known gaps* section. |
| [HANDOFF.md](HANDOFF.md) | code-review orientation, risk-ranked. |
| [MODAL.md](MODAL.md) | how to run anything on a GPU. |
| [RL_STRATEGY.md](RL_STRATEGY.md) | post-training strategy; why Phase 1 carries three jobs. |
| `config.py` | the run config. Single source of truth for shapes, seeds, paths. |
| `data/shards/manifest.json` | the **data recipe** — tokenizer, thresholds, per-slice token counts. |
| `runs/<run_id>/run.json` | the truth about one run. `index.jsonl` is best-effort only. |

Keep it that way. Do not maintain two changelogs, and do not let status drift into a
third file.

## Four hard rules

1. **Run nothing locally.** The dev box is Windows 11 and is not the compute box. Every
   local execution path here has cost real time: cp1252 encoding crashes ×3, MSYS2 path
   mangling that misrouted 890 MB of uploads, a platform-split torch pin. This machine
   **edits files and invokes `modal run`.** Tests are the one exception and they run on
   Modal too (`modal run modal_app.py::tests`).
2. **`train_baseline.py` is a control arm, not a model.** It is deliberately untuned.
   **A change that makes the model better is a defect here** — it destroys
   comparability with every measurement already taken. Improvements belong in the
   K3-mini files, which do not exist yet.
3. **Don't over-architect.** This is a solo research repo. Prefer one file that does the
   thing over a framework that could. Best practices yes; layers of abstraction no.
4. **Log the *why*.** Add a CHANGELOG entry when behaviour changes, and say what failure
   the change was made against. Several decisions here are invisible from the code
   alone and were re-derived expensively once already.

## Compute

Modal, three Volumes. Nothing runs by itself; nothing is running right now.

```bash
modal run modal_app.py::verify_data              # cheap sanity check, no GPU
modal run modal_app.py::smoke                    # short GPU run, always before a long one
modal run --detach modal_app.py::train           # the real thing, survives local death
modal run modal_app.py::fetch                    # pull artifacts to .\modal_out\
```

| volume | holds |
|---|---|
| `k3mini-shards` | the corpus (~2.8 GB, includes ~933 MB of known-stray duplicates) |
| `k3mini-runs` | checkpoints, logs, samples, `run.json` (~44 GB and growing) |
| `k3mini-torch-cache` | inductor cache |

`--detach` matters: a long run must not depend on this laptop staying awake. Always
smoke before a long run — it has caught real GPU-only failures three times.

## Where the project is

Milestones 0 and 1 of 10 are done. Corpus: 3,431 books → 3,178 trained → 462 M tokens.
Baseline trained end to end: 3,250 steps, val_loss 2.80559, 3.07 h on an A100. The
mixing dataloader is built and GPU-verified; the seed band is measured (range 0.00198
at 1,000 steps, so single-seed ablations are defensible at that horizon).

**None of K3-mini itself exists yet.** No KDA, MoE, AttnRes, SiTU-GLU or MTP head.

**The binding constraint is the corpus, not model size and not the loader.** The
register slice is 8.7% (target 10–25%), general modern text is 0% (budget 10–20%,
"not optional"), craft essays 0%. Milestone 1's prose is competent Victorian pastiche
and not eerie, and the `mundane` probe collapsed into repetition — that is the
general-text gap, measured.

**Next, in order:** source the missing slices (PLAN.md §5.1, mechanics and
hand-vs-automation split in **§5.1.2**), then the §7.2 exemplar/contrast-pair eval
harness. Both are CPU-only. `kda_mini.py` (Milestone 2) is compute-side and runs in
parallel. Do not launch a mix ablation before §7.2 exists — its curves cannot be
scored on the axis that matters.

## Traps that have already cost time

- **Windows paths in artifacts are unparseable on Linux.** `Path("a\b\c.bin").name`
  returns the whole string on POSIX. Glob our own naming convention instead of trusting
  a `shard_path` from the manifest — `data/mix_loader.py` does this deliberately.
- **`.gitignore` patterns with a single `*` do not cross directory separators.**
  `data/shards*/*.bin` missed `by_slice/` entirely; 882 MB sat untracked-but-not-ignored,
  one `git add -A` from entering history permanently.
- **Dataclass fields need annotations.** An un-annotated `config.py` attribute is not a
  field, vanishes from `asdict()`, and silently drops out of `config_hash()`.
- **Modal Volume commits are whole-file snapshots.** Concurrent runs appending to one
  index file overwrite each other — that is why `run.json` is the source of truth.
- **`torch.compile` NaN'd on torch 2.10.0+cu128/sm89** at the second optimizer update.
  Fixed by pinning **torch 2.13.0**. Don't unpin casually.
- **PowerShell quoting breaks inline Python.** Use a heredoc via the Bash tool, or a
  file. This has discarded a queued Modal run once.
- **pytest needs `--basetemp`** pointed somewhere writable on this box (WinError 5).

## Conventions

- Branch off `master`, which is this fork's main branch. Small commits, imperative
  subject lines, and the *why* in the body when it isn't obvious.
- Tests live in `tests/`, mirror the module name, and run on Modal.
- Provenance artifacts are committed on purpose and must not be gitignored:
  `data/manifest.jsonl`, `data/val_books.json`, `data/shards/manifest.json`.
- Corpus text, shards, checkpoints and tokenizer artifacts are **never** committed —
  they are regenerable from the manifest, and committing them bloats history forever.
