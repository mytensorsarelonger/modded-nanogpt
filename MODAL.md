# Running the Phase 0 baseline on Modal

`modal_app.py` launches the existing `train_baseline.py` on Modal's GPUs without
modifying it. It is a launcher only: model size, LR, schedule and step count all
still come from `config.py`. The only knobs it touches are the environment
variables `train_baseline.py` already reads.

---

## 0. Preflight state

The previously reported self-recursive `device_name()` defect is fixed. The
launcher retains a local source check for that exact regression and now also:

- verifies every image input exists;
- syntax-compiles `train_baseline.py`, `config.py`, `probes.py`, and
  `training_state.py` locally;
- ships both imported helper modules (`probes.py` and `training_state.py`); and
- forwards `GIT_SHA`/`GIT_DIRTY`, so the trainer's run registry records the
  checkout provenance even though the image intentionally contains no `.git`.

The exact process-group call was checked against PyTorch **v2.10.0** source.
Its backend capability table declares Gloo for both CPU and CUDA; constructing
the group registers that Gloo instance for both device types. A CUDA
`device_id` is validated and bound to the group, while eager device connection
is attempted only for a backend that advertises splitting support. Therefore
`init_process_group(backend="gloo", device_id=torch.device("cuda:0"))` and the
CUDA `broadcast`/`all_reduce` calls are supported in the pinned version. Keeping
Gloo also preserves the control arm's configured backend and config hash.
See the [v2.10.0 backend table](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L265-L267)
and [group construction](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/distributed_c10d.py#L1856-L1857).

That resolves the API-level concern, but it is not a substitute for the L4
smoke: the wheel build, fused optimizer, actual CUDA tensors, bf16
Newton–Schulz path, and compiled graphs remain unexecuted here.

The Linux image remains on PyTorch 2.10.0. `requirements.txt` uses PyTorch
2.13.0 only on Windows because the 2.10 Windows wheel reports
`unsupported gloo device` before process-group initialization; this does not
change the Modal control environment.

Run this cheap local gate before uploading or allocating a GPU:

```powershell
python -c "import modal_app; print('modal_app preflight OK')"
modal run modal_app.py::train --help
```

---

## 1. Prerequisites

```powershell
pip install modal
modal setup            # opens a browser; writes ~/.modal.toml
```

You need:

- A Modal account with billing enabled (GPUs are not in the free tier).
- The repo checked out locally, with `data/shards/` populated
  (5 train shards + 1 val shard + `manifest.json`, ~940 MB total).
- Nothing else. No Docker, no CUDA locally. The image is built remotely.

Sanity check without touching a GPU:

```powershell
python -c "import ast;ast.parse(open('modal_app.py',encoding='utf-8').read());print('ok')"
```

---

## 2. One-time data upload

The shards go in a **Volume**, not the image. A 940 MB image layer would make
every one-character code edit a multi-minute rebuild, and it would couple the
data recipe to the code version for no benefit.

Upload is done with the **`modal volume put` CLI**, not a Modal function. This is
a deliberate choice: the CLI already does chunked, retrying uploads and is the
maintained path for bulk data. A function that streamed the same bytes would
have to pass them as serialized arguments (size-limited), could not resume, and
would bill container time to do worse. What the CLI does *not* do is check the
bytes against `manifest.json` — so `modal_app.py` provides `verify` for that
instead.

`manifest.json` must land **inside** the volume, next to the `.bin` files:
`config.validate_against_shards()` reads `shard_dir / "manifest.json"`, and
`shard_dir` is `data/shards`, which is the volume mount point.

```powershell
# creates the volume if it does not exist
modal volume create k3mini-shards

foreach ($f in Get-ChildItem data\shards\*.bin) {
    modal volume put k3mini-shards $f.FullName "/$($f.Name)"
}
modal volume put k3mini-shards data\shards\manifest.json /manifest.json

modal volume ls k3mini-shards
```

Bash equivalent:

```bash
for f in data/shards/*.bin; do modal volume put k3mini-shards "$f" "/$(basename "$f")"; done
modal volume put k3mini-shards data/shards/manifest.json /manifest.json
```

Expect roughly 10–40 minutes on a home upstream link. If a file fails, re-put
that one file; the loop is per-file precisely so a failure is cheap.

Then verify, which is the step that actually tells you the upload worked:

```powershell
modal run modal_app.py::verify
```

This re-reads `manifest.json` from the volume and checks every shard's presence,
byte size and **sha256** against it. `--no-checksums` does presence+size only
(seconds instead of a couple of minutes). A non-zero exit means do not launch.

The other two volumes (`k3mini-runs`, `k3mini-torch-cache`) are created
automatically on first use by `create_if_missing=True`.

> Note: `manifest.json` records absolute *Windows* shard paths
> (`C:\Users\...`). This is harmless — `config.py` never reads `shard_path`, and
> the dataloader globs `Path.cwd()`. Only the basenames matter, and `verify`
> compares basenames.

---

## 3. Smoke test first — always

```powershell
modal run modal_app.py::smoke
```

~4–8 minutes end to end on an **L4**, of which most is image build (first time
only) and inductor cold start. **Under $0.10.**

What it buys: every code path that cannot be exercised on a CPU box.
bf16 parameters and the bf16 Newton–Schulz iteration inside `muon_update`,
`pin_memory=True` in the shard loader, `non_blocking=True` H2D copies, fused
AdamW (a CUDA-only kernel), the gloo-on-CUDA collectives, TorchInductor
codegen, and the run-end checkpoint + sample path (including a ~1.4 GB
checkpoint, which is most of the smoke's tail latency).

Settings: `SMOKE=1 TRAIN_STEPS=100 COMPILE=1 BATCH_SIZE=65536 MBS=8
VAL_TOKENS=524288`. Note `COMPILE=1` is forced — `train_baseline.py` defaults
compile *off* under `SMOKE`, which is right for a ten-step CPU check and wrong
here, since inductor cold start is one of the main risks a smoke exists to
retire.

**Do not read the smoke's val loss as signal.** `SMOKE=1` shrinks the batch to
64 Ki tokens/step and evaluates a token slice. It is a plumbing test.

More steps if you want a rough throughput reading before committing:

```powershell
modal run modal_app.py::smoke --train-steps 300
```

---

## 4. The full run

```powershell
modal run --detach modal_app.py::train
```

**Use `--detach` for long runs.** Resume protects work up to the latest committed
checkpoint, but detaching prevents a laptop sleep, VPN blip, or `Ctrl-C` from
killing the worker in the first place. Reattach with `modal app logs <app-id>`
(find the id with `modal app list`).

Overridable:

```powershell
modal run --detach modal_app.py::train --train-steps 1000      # shorter run
modal run --detach modal_app.py::train --mbs 16                # 80GB cards only
modal run --detach modal_app.py::train --no-compile            # eager, for debugging
modal run --detach modal_app.py::train_h100 --train-steps 3250 # H100, see §5
```

Resume paths are always relative to the `k3mini-runs` Volume; absolute paths and
`..` are rejected. Resume from the newest checkpoint in a prior run directory,
or name one exactly:

```powershell
modal run --detach modal_app.py::train --resume auto --resume-dir <run_id>
modal run --detach modal_app.py::train --resume <run_id>/ckpt_01000.pt
```

The same `--resume` / `--resume-dir` flags are available on `smoke` and
`train_h100`. Keep the original run's sizing arguments (`--mbs`, batch size,
step target, and smoke/full mode); the trainer rejects incompatible checkpoint
metadata rather than silently changing the experiment.

Configuration, and why:

| Setting | Value | Reason |
|---|---|---|
| GPU | `A100-40GB` | bf16 needs Ampere+. Never T4 (sm75) or V100 (sm70) — no bf16 tensor cores. |
| `timeout` | `6 * 60 * 60` | Modal's default is **5 minutes**. Explicit, ~2x headroom over the ~3.2h estimate. |
| `retries` | `0` | A blind Modal retry cannot know which prior run/checkpoint is the intended recovery source. Training is not idempotent. Inspect the failure, then resume explicitly. |
| `MBS` | `8` | Memory bound is the **loss head**, not the model: logits are materialised fp32 at `(mbs*1024, 50304)` = 1.65 GB at mbs=8, and the softcap arithmetic plus `cross_entropy` keep several of those live. mbs=16 is about the 40 GB ceiling; mbs=32 OOMs. Must divide 512 (`524288 % (1024*mbs) == 0`). |
| `cpu=8, memory=32768` | | The loader pins a whole 200 MB shard in host memory and re-pins on every shard rollover. |

Watch it:

```powershell
modal app list
modal app logs <app-id>
```

Progress lines are `step:N/3250 train_time:...s step_avg:...ms`; `step_avg`
after step ~200 is your real throughput number.

---

## 5. Expected cost and wall clock — 3250 steps on A100

**Tokens processed.** `config.batch_size = 8 * 64 * 1024 = 524,288` tokens/step.

```
D = 3250 steps x 524,288 tokens = 1,703,936,000 tokens  (~1.70e9)
```

**Model size.** The task spec says ~124M, which is the GPT-2-small figure.
The actual parameter count of this model is higher because the LM head is
**untied**:

```
embedding   50304 x 768                      =  38.6M
12 blocks   12 x (4 x 768^2 attn + 2 x 768x3072 MLP + biases + norms)  =  85.0M
LM head     768 x 50304 + bias               =  38.7M
--------------------------------------------------------
total                                          162.4M
```

**FLOPs, at `6ND`** (2 for the forward multiply-accumulate, 4 for the backward):

| N used | `6ND` |
|---|---|
| 124M (nominal, used below) | **1.27e18** |
| 162.4M (all params) | 1.66e18 |
| 85.0M (non-embedding, Chinchilla convention) | 8.69e17 |

The spread is 2x, which is the honest uncertainty in any `6ND` estimate for a
small model with a fat vocabulary. Everything below uses **N = 124e6**, so
treat the numbers as a floor if you prefer the 162M convention.

**MFU assumption: 35%.** Justification: A100 bf16 dense peak is 312 TFLOP/s.
This model has ReLU² MLPs, an untied 50304-wide head with an extra fp32 softcap,
and — the real limiter — runs microbatches of only `8 x 1024 = 8192` tokens, so
64 sequential forward/backward passes per optimizer step. Reference speedrun
implementations reach 40–50% MFU at much larger microbatches; 35% is the middle
of a plausible 25–45% band here. The band, not the point estimate, is the
result.

```
1.27e18 / (312e12 x 0.35) = 11,609 s = 3.22 h
```

| MFU | wall clock | A100-40GB @ $2.78/h | A100-80GB @ $3.40/h |
|---|---|---|---|
| 25% (pessimistic) | 4.51 h | $12.55 | $15.35 |
| **35% (assumed)** | **3.22 h** | **$8.96** | **$10.96** |
| 45% (optimistic) | 2.51 h | $6.97 | $8.53 |

**Overheads on top, ~5–8%:**

- 38 validation passes (every 125 steps until 90% of the run, every 25 after —
  `24 + 14`), each a forward over ~4.33M capped val tokens:
  `38 x 2 x 124e6 x 4.33e6 = 4.1e16` = **+3.2%**.
- 13 sample dumps (every 250 steps, including the final step), uncompiled eager;
  each pauses the training clock.
- 4 checkpoints (1000/2000/3000/final) at ~1.4 GB, written to a Volume.
- Container start + inductor compile: 1–4 min cold, seconds with a warm cache
  volume.

**Budget $10–15 for one A100 run.** Add ~$0.10 for the smoke.

**H100 is worth measuring.** At 989 TFLOP/s bf16 peak, the same 35% MFU gives
1.02 h at $4.56/h = **$4.64** — cheaper *and* faster than the A100. The catch is
that a 124M model at 8192-token microbatches under-feeds an H100, so realised
MFU is usually lower; at 25% it is 1.42 h and $6.49, still competitive. Run
`train_h100` once and compare `step_avg` before deciding. Do not assume the 3.2x
peak-FLOP ratio.

For scale: the FineWeb10B sanity anchor in PLAN.md §4.0 is 3.28 val loss at 3250
steps. Your Gutenberg corpus is 462M train tokens against a 1.70e9-token budget,
so this run makes **3.69 epochs** over the corpus — `train_baseline.py` reports
this as `epochs_over_corpus` and cycles shards deliberately. Expect the val
curve to reflect that.

> **The `$/hr` figures above are the one number here I could not verify.** They
> are from memory of Modal's published GPU pricing and may be stale or wrong.
> Check <https://modal.com/pricing> before quoting a cost to anyone, and note
> that Modal bills CPU and memory reservations on top of the GPU. PLAN.md §2
> assumes "~$3–4/GPU-hr on Modal" for H100, which is consistent with the $4.56
> used here to within the same margin.

---

## 6. Getting results back

Small text artifacts — `runs/index.jsonl`, every `samples.log`, the launcher
provenance sidecar, and the 5 most recent `logs/*.txt` (each contains the full source of
the script that produced it):

```powershell
modal run modal_app.py::fetch                       # -> .\modal_out\
modal run modal_app.py::fetch --dest .\pulled
modal run modal_app.py::fetch --no-include-logs
```

Safe to run **while a detached job is still training** — it calls
`Volume.reload()` first, so you get the log as of now. This is how you read the
loss curve and the sampled prose mid-run without touching the training
container.

Checkpoints are ~1.4 GB each and are deliberately *not* returned by `fetch`
(it lists them instead). Pull one explicitly:

```powershell
modal volume ls k3mini-runs
modal volume get k3mini-runs <run_id>/ckpt_03250.pt .\runs\<run_id>\
modal volume get k3mini-runs index.jsonl .\
```

### Where things land in the runs volume

```
runs/                              <- k3mini-runs volume
  index.jsonl                      the run registry (PLAN.md §4.0.1)
  modal_provenance.jsonl           launcher audit: git, GPU, env, exit code
  <run_id>/ckpt_NNNNN.pt           ~1.4 GB each
  <run_id>/ckpt_NNNNN.rankRRRRR.pt rank-local Muon/RNG resume state
  <run_id>/samples.log             the primary instrument (PLAN.md §7.1)
  _logs/<uuid>.txt                 what train_baseline.py writes to logs/
```

`logs/` inside the container is a **symlink** into `runs/_logs` on the volume.
`train_baseline.py` does `os.makedirs("logs")` in CWD and writes there; without
the symlink those logs are container-local and vanish. `makedirs(exist_ok=True)`
is happy with a symlink to an existing directory, so no code change is needed.

The runs volume is committed by a background thread **every 120 seconds**, plus
once in a `finally` block. A run that dies retains recent logs and any committed
checkpoint set needed for diagnosis or explicit resume.

### `torch.compile` cache — the tradeoff

`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` point into the
`k3mini-torch-cache` volume, namespaced per GPU type. Upside: inductor cold
start drops from 1–4 minutes to seconds, and that cost is paid **per
container**. Downsides, both real:

- A container killed mid-compile can leave a partial entry that produces
  baffling errors later. **If a run fails inside inductor, delete the cache
  volume before debugging anything else:**
  `modal volume delete k3mini-torch-cache`.
- It is a shared mutable directory; concurrent runs write the same keys. Benign
  in practice, but a run whose wall clock you intend to *publish* should start
  from a cold cache, because the first-container compile time is part of the
  honest number.

---

## 7. Current launcher/trainer integration

- `device_name()` uses `torch.cuda.get_device_name(device)`; the launcher keeps
  a regression check for the old self-recursive form.
- `probes.py` and `training_state.py` are copied into the image next to the
  trainer and syntax-checked before launch.
- `GIT_SHA` and `GIT_DIRTY` are forwarded to the trainer's authoritative run
  registry. `modal_provenance.jsonl` remains as a launcher/runtime audit trail.
- `RESUME` and `RESUME_DIR` are exposed as `--resume` and `--resume-dir` on all
  three training functions. The launcher constrains paths to the runs Volume
  and confirms the selected directory/checkpoint exists.
- `config.dist_backend` is environment-overridable, but the launcher does not
  silently switch it: changing the backend changes `config_hash()` and would
  make this a different control-arm configuration.

`modal_out/` (the default `fetch` destination) is ignored as a generated artifact.

---

## 8. Verification status

The launcher has been checked locally with Modal 1.4.2. Importing the app and
building its CLI schema succeeds. The installed API accepts `copy=True` on
`Image.add_local_file`, named Volumes with `create_if_missing`, container-side
`commit()`/`reload()`, direct function entrypoints, boolean negation flags, and
the GPU strings `L4`, `A100-40GB`, and `H100`. The `modal volume put` CLI order
used in §2 also matches its current help. The pinned PyPI releases
`torch==2.10.0`, `numpy==2.2.6`, and `tiktoken==0.11.0` exist.

Still remote-only and therefore gated by the L4 smoke:

1. A Volume mount nested below image content (`data/shards` below `data/`) and
   a periodic `Volume.commit()` while the trainer subprocess is writing.
2. The default Linux Torch wheel, Triton, compiler toolchain, and baked
   tiktoken cache working together in the built Modal image.
3. The exact tensor mix through Gloo CUDA collectives, fused mixed-dtype AdamW,
   bf16 Newton–Schulz, and both compiled graphs. PyTorch v2.10.0 source confirms
   the Gloo + CUDA `device_id` API path; this experiment has not executed its
   tensors on a GPU.
4. Peak memory. MBS=8 on 40 GB is an estimate, dominated by fp32 logits at
   `(8192, 50304)` = 1.65 GB plus live softcap/cross-entropy intermediates.
5. Throughput and cost. The 35% MFU in §5 is a planning assumption; replace it
   with `step_avg` from `modal run modal_app.py::smoke --train-steps 300`.

Checks resolved from source/data rather than left as assumptions:

- The val shard has 4,333,759 tokens. At MBS=8 the capped usable count is
  **4,333,568 tokens** (4,232 rows), divisible by 8.
- A checkpoint has been observed at approximately 1.38 GB. A full run writes
  four checkpoint sets (1000/2000/3000/final), so allow roughly 5.5 GB plus
  rank-state sidecars and logs per run.
- `forward_logits` is intentionally eager; sampling growing sequence lengths
  therefore does not cause a dynamic-shape recompilation storm.
