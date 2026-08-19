"""
modal_app.py — run `train_baseline.py` on Modal's cloud GPUs.

This file is a *launcher*, not a trainer. It owns exactly three things:

  1. A container image that can run `train_baseline.py` unmodified.
  2. Three Volumes: shards (read-mostly), runs/logs (written), compile cache.
  3. A filesystem layout that satisfies the two hard assumptions the training
     script makes about where it lives (see LAYOUT below).

Everything about the *model* — steps, batch size, LR, schedule — stays in
`config.py` and `train_baseline.py`. The only knobs exposed here are the
environment variables that script already reads (SMOKE, TRAIN_STEPS,
BATCH_SIZE, MBS, VAL_TOKENS, COMPILE, RESUME, RESUME_DIR).

See MODAL.md for prerequisites, the one-time data upload, exact commands,
cost arithmetic, and the list of things this file assumes but cannot verify.

LAYOUT — why the paths below are not arbitrary
----------------------------------------------
`train_baseline.py`'s dataloader does `Path.cwd().glob("data/shards/...bin")`,
so the *process CWD* must be the repo root. Independently, `config.py` derives
`shard_dir` from `Path(__file__).parent`. Both must agree, so the code and the
data have to sit in one tree:

    /root/modded-nanogpt/            <- REPO_DIR, and the subprocess CWD
      train_baseline.py              (image layer)
      config.py                      (image layer)
      probes.py                      (image layer; imported by trainer)
      training_state.py              (image layer; resume/checkpoint helpers)
      data/manifest.jsonl            (image layer; needed for manifest_hash())
      data/shards/                   <- data Volume (bins + manifest.json)
      runs/                          <- runs Volume (checkpoints, samples.log, index.jsonl)
      logs/                          -> symlink into the runs Volume

RECOVERY
--------
Checkpoint resume is explicit: pass `--resume auto --resume-dir <run_id>` or
`--resume <run_id>/ckpt_NNNNN.pt`. Both paths are resolved strictly inside the
runs Volume and checked before the paid subprocess starts. Consequences baked
into this file:
  * `retries=0` — training is not idempotent and a blind retry cannot choose
    the correct source run/checkpoint.
  * The runs Volume is committed every 2 minutes by a background thread, not
    only at exit, so a timeout or crash preserves recent logs/checkpoints.
  * Long runs should still use `--detach`. See MODAL.md.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "k3mini-baseline"

# Container-side paths. REPO_DIR is deliberately NOT /root: Modal places this
# app module at /root, and a Volume mounted over /root would shadow it.
REPO_DIR = "/root/modded-nanogpt"
DATA_MOUNT = f"{REPO_DIR}/data/shards"
RUNS_MOUNT = f"{REPO_DIR}/runs"
CACHE_MOUNT = "/cache"

# Local paths. This file lives at the repo root, same as config.py's ROOT.
LOCAL_ROOT = Path(__file__).parent
LOCAL_SHARDS = LOCAL_ROOT / "data" / "shards"

# Files that must be real image-layer content. Keep this list in lockstep with
# the trainer's local imports AND with anything the test suite imports: the suite
# runs in this image (see run_tests), and a runner that silently cannot collect
# part of the suite is worse than no runner, because "tests pass" stops meaning it. During a remote import, the image-layer copies
# already live under REPO_DIR rather than beside modal_app.py.
_IMAGE_SOURCE_ROOT = LOCAL_ROOT if modal.is_local() else Path(REPO_DIR)
IMAGE_FILES = {
    "train_baseline.py": _IMAGE_SOURCE_ROOT / "train_baseline.py",
    "config.py": _IMAGE_SOURCE_ROOT / "config.py",
    "probes.py": _IMAGE_SOURCE_ROOT / "probes.py",
    "training_state.py": _IMAGE_SOURCE_ROOT / "training_state.py",
    "data/mixing.py": _IMAGE_SOURCE_ROOT / "data" / "mixing.py",
    "data/mix_loader.py": _IMAGE_SOURCE_ROOT / "data" / "mix_loader.py",
    "data/slices.py": _IMAGE_SOURCE_ROOT / "data" / "slices.py",
    "data/dedup.py": _IMAGE_SOURCE_ROOT / "data" / "dedup.py",
    "data/prepare_colab_smoke.py": _IMAGE_SOURCE_ROOT / "data" / "prepare_colab_smoke.py",
    "data/quality_filter.py": _IMAGE_SOURCE_ROOT / "data" / "quality_filter.py",
    "data/shard_writer.py": _IMAGE_SOURCE_ROOT / "data" / "shard_writer.py",
    "data/manifest.jsonl": _IMAGE_SOURCE_ROOT / "data" / "manifest.jsonl",
    "data/val_books.json": _IMAGE_SOURCE_ROOT / "data" / "val_books.json",
}
_missing_image_files = [name for name, path in IMAGE_FILES.items() if not path.is_file()]
if _missing_image_files:
    raise FileNotFoundError(
        "Modal image inputs are missing: " + ", ".join(_missing_image_files)
    )

# Hardware. bf16 is unconditional in this model (`nn.Embedding(...).bfloat16()`,
# `X.bfloat16()` inside the Newton-Schulz iteration), so the GPU must be Ampere
# or newer. T4 (Turing, sm75) and V100 (Volta, sm70) have no bf16 tensor cores
# and will either fall over or run emulated at a crawl — never use them here.
SMOKE_GPU = "L4"          # Ada, 24 GB, cheapest bf16-capable option on Modal.
TRAIN_GPU = "A100-40GB"   # 40 GB is ample at MBS=8; see MODAL.md for the math.
BIG_GPU = "H100"          # Often *cheaper in total* than A100 — see MODAL.md.

# Modal's default function timeout is 5 minutes, which is absurdly short for a
# 3-hour training run, so every timeout below is explicit. 6h leaves ~2x
# headroom over the ~3.2h central estimate for a 3250-step A100 run.
# The corrected full-batch, cold-cache L4 gate measured 27.5 minutes. Forty-five
# minutes leaves useful variance headroom without turning a hung smoke into an
# unbounded paid job.
SMOKE_TIMEOUT = 45 * 60
TRAIN_TIMEOUT = 6 * 60 * 60
BIG_TIMEOUT = 4 * 60 * 60

COMMIT_INTERVAL_S = 120
_COMMON_CHECKPOINT = re.compile(r"ckpt_\d+\.pt\Z")

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
#
# Version pinning rationale:
#
#  torch==2.13.0   Matches requirements.txt and is the first verified runtime
#                  for both compiled CUDA training and Windows Gloo. Torch 2.10
#                  produced non-finite weights on the second compiled update.
#                  torch is
#                  the one dependency where an unpinned float is genuinely
#                  dangerous: inductor codegen, `F.rms_norm`, the `device_id=`
#                  kwarg to init_process_group, and fused AdamW's dtype-group
#                  handling have all moved between minor releases. A run that
#                  silently picks up a different torch is not the same
#                  experiment, and `config_hash()` does not capture torch's
#                  version — so the pin here is part of the provenance.
#                  The default PyPI linux wheel bundles CUDA and pulls the
#                  matching triton, which is what inductor needs.
#  numpy==2.2.6    Only used for the zero-copy `f.readinto(tokens.numpy())` in
#                  the shard loader. Pinned to a known-good 2.x so a future
#                  numpy 3 ABI break cannot surprise a launch; nothing here
#                  needs a newer numpy.
#  tiktoken==0.11.0  Only used for `get_encoding("gpt2")`, whose merges are
#                  frozen — but the *tokenizer identity* is recorded in the run
#                  registry, so pinning keeps that record honest.
#
# apt: build-essential because TorchInductor shells out to a C++ compiler for
# its wrapper/CPU fallbacks and debian_slim ships none; git remains available
# for diagnostics even though provenance is forwarded explicitly via env.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "git")
    .pip_install(
        "torch==2.13.0",
        "numpy==2.2.6",
        "tiktoken==0.11.0",
        # Always installed, even when tracking is off. Adding a package changes
        # the image hash, so making it conditional would force an image rebuild
        # every time you toggled WANDB. One rebuild now, stable thereafter; the
        # trainer only imports it when WANDB=1.
        "wandb==0.19.11",
        # So the suite can run in the real environment. The dev box is Windows and
        # nothing there is representative; see CHANGELOG 2026-08-06 "Lesson".
        "pytest==8.3.4",
    )
    # Bake the GPT-2 BPE files into the image. Without this, every container
    # cold-start fetches them from openaipublic.blob.core.windows.net, which
    # turns a third-party outage into a failed launch.
    .env({"TIKTOKEN_CACHE_DIR": "/opt/tiktoken-cache"})
    .run_commands('python -c "import tiktoken; tiktoken.get_encoding(\'gpt2\')"')
    # copy=True (not the default lazy mount) for two reasons: it puts these
    # files in a real image layer so a Volume can be mounted at the nested path
    # data/shards without racing a runtime file-mount layer, and it makes the
    # training code part of the image's identity, so editing train_baseline.py
    # produces a new image rather than silently changing what "the same image"
    # runs. The layer is a few hundred KB, so the rebuild is seconds.
    .add_local_file(IMAGE_FILES["train_baseline.py"], f"{REPO_DIR}/train_baseline.py", copy=True)
    .add_local_file(IMAGE_FILES["config.py"], f"{REPO_DIR}/config.py", copy=True)
    .add_local_file(IMAGE_FILES["probes.py"], f"{REPO_DIR}/probes.py", copy=True)
    .add_local_file(IMAGE_FILES["training_state.py"], f"{REPO_DIR}/training_state.py", copy=True)
    # manifest.jsonl (~5 MB) is what makes `data_manifest_hash` in the run
    # registry a real value instead of "unknown". PLAN.md §4.0.1 calls this the
    # field people skip and regret, so it ships.
    .add_local_file(IMAGE_FILES["data/mixing.py"], f"{REPO_DIR}/data/mixing.py", copy=True)
    .add_local_file(IMAGE_FILES["data/mix_loader.py"], f"{REPO_DIR}/data/mix_loader.py", copy=True)
    .add_local_file(IMAGE_FILES["data/slices.py"], f"{REPO_DIR}/data/slices.py", copy=True)
    .add_local_file(IMAGE_FILES["data/dedup.py"], f"{REPO_DIR}/data/dedup.py", copy=True)
    .add_local_file(IMAGE_FILES["data/prepare_colab_smoke.py"], f"{REPO_DIR}/data/prepare_colab_smoke.py", copy=True)
    .add_local_file(IMAGE_FILES["data/quality_filter.py"], f"{REPO_DIR}/data/quality_filter.py", copy=True)
    .add_local_file(IMAGE_FILES["data/shard_writer.py"], f"{REPO_DIR}/data/shard_writer.py", copy=True)
    .add_local_file(IMAGE_FILES["data/manifest.jsonl"], f"{REPO_DIR}/data/manifest.jsonl", copy=True)
    .add_local_file(IMAGE_FILES["data/val_books.json"], f"{REPO_DIR}/data/val_books.json", copy=True)
    # The suite runs HERE, not on the dev box. Windows is unrepresentative of the
    # training environment in every way that has already cost this project time
    # (see CHANGELOG 2026-08-06 "Lesson"), so "the tests pass" should mean they
    # passed on the image that trains.
    .add_local_dir(str(LOCAL_ROOT / "tests"), f"{REPO_DIR}/tests", copy=True)
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
#
# data_vol   ~940 MB of .bin shards + manifest.json. Uploaded once with
#            `modal volume put`, then read-only in practice. Deliberately NOT
#            baked into the image: a 940 MB layer makes every code edit a
#            multi-minute rebuild, bloats the image cache, and couples the data
#            recipe to the code version for no benefit.
# runs_vol   Everything that must outlive the container: runs/<uuid>/ckpt_*.pt,
#            runs/<uuid>/samples.log, runs/index.jsonl, and (via symlink) logs/.
# cache_vol  TorchInductor + Triton compile cache. See _build_env() for the
#            tradeoff.
data_vol = modal.Volume.from_name("k3mini-shards", create_if_missing=True)
runs_vol = modal.Volume.from_name("k3mini-runs", create_if_missing=True)
cache_vol = modal.Volume.from_name("k3mini-torch-cache", create_if_missing=True)

VOLUMES = {DATA_MOUNT: data_vol, RUNS_MOUNT: runs_vol, CACHE_MOUNT: cache_vol}

# ---------------------------------------------------------------------------
# Optional W&B
# ---------------------------------------------------------------------------
#
# Opt-in via a LOCAL env var, read at `modal run` time when this module is
# imported: WANDB=1 modal run modal_app.py::train
#
# The Secret attachment is UNCONDITIONAL, and that is load-bearing.
#
# It used to be `[Secret.from_name(...)] if WANDB_ENABLED else []`, intending
# that a missing secret could never block a run. That is wrong, and it fails
# exactly when you turn tracking on: WANDB is set in the *local* shell at
# `modal run` time, but the container re-imports this module without it, so the
# function was declared with 5 objects locally and 4 remotely --
#   ExecutionError: Function has 4 dependencies but container got 5 object ids.
# Any Modal object built under a condition that differs between the local and
# remote import has this bug. Attaching the Secret always keeps both imports
# identical; the trainer still only imports wandb when WANDB=1, so an unused
# key costs nothing.
#
# Consequence: `wandb-secret` must exist. One-time setup (the key never passes
# through this repo, and a placeholder is fine if you never enable tracking):
#   modal secret create wandb-secret WANDB_API_KEY=<key>
WANDB_ENABLED = os.environ.get("WANDB", "0") == "1"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "k3mini")
SECRETS = [modal.Secret.from_name("wandb-secret")]

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Local helpers (run on your machine at `modal run` time)
# ---------------------------------------------------------------------------

def _local_git() -> tuple[str, bool]:
    """(sha, dirty) of the local checkout, or ("", True) if unknowable.

    Guarded on .git existing so that importing this module inside the container
    — which Modal does — costs zero subprocesses and returns "".
    """
    if not (LOCAL_ROOT / ".git").exists():
        return "", True
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LOCAL_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=LOCAL_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip() != ""
        return sha, dirty
    except Exception:
        return "", True


_GIT_SHA, _GIT_DIRTY = _local_git()


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

# `def device_name(): return device_name() if CUDA else "CPU"` recurses forever
# the moment CUDA is available, and it is called from the very first log line.
# On CPU it returns "CPU" and nobody notices; on any GPU it is an instant
# RecursionError before a single step runs. Cheap to detect, so detect it
# rather than paying for a container to find out.
_RECURSION_BUG = re.compile(
    r"def\s+device_name\s*\([^)]*\)[^:]*:\s*\n\s*return\s+device_name\s*\(",
    re.MULTILINE,
)

_RECURSION_MSG = """\
Regression detected: train_baseline.py has a self-recursive device_name():

    def device_name() -> str:
        return device_name() if CUDA else "CPU"

On CUDA this raises RecursionError at the first print0() call, before step 0.
The one-line fix, in train_baseline.py, is:

    def device_name() -> str:
        return torch.cuda.get_device_name(device) if CUDA else "CPU"

Set K3_SKIP_PREFLIGHT=1 to launch anyway (it will fail on the GPU)."""


def _preflight_source(src: str) -> None:
    compile(src, "train_baseline.py", "exec")
    if os.environ.get("K3_SKIP_PREFLIGHT") == "1":
        return
    if _RECURSION_BUG.search(src):
        raise RuntimeError(_RECURSION_MSG)


# Run the same checks at import time, so `modal run` fails on the local machine
# in milliseconds. A remote import checks the image-layer copies under
# REPO_DIR as a second line of defense.
for _source_name in ("train_baseline.py", "config.py", "probes.py", "training_state.py"):
    _source = IMAGE_FILES[_source_name].read_text(encoding="utf-8")
    if _source_name == "train_baseline.py":
        _preflight_source(_source)
    else:
        compile(_source, str(IMAGE_FILES[_source_name]), "exec")


# ---------------------------------------------------------------------------
# Container-side machinery
# ---------------------------------------------------------------------------

def _runs_volume_path(value: str, label: str) -> str:
    """Resolve a user-supplied relative path strictly inside RUNS_MOUNT."""
    normalized = value.strip().replace("\\", "/")
    parts = normalized.split("/")
    rel = PurePosixPath(normalized)
    if not normalized or rel.is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise ValueError(
            f"{label} must be a path relative to the k3mini-runs Volume "
            "(no absolute paths, empty components, '.' or '..')"
        )
    return str(PurePosixPath(RUNS_MOUNT).joinpath(*parts))


def _resume_overrides(resume: str, resume_dir: str) -> dict[str, str]:
    """Translate CLI resume arguments to the trainer's environment contract."""
    resume = resume.strip()
    resume_dir = resume_dir.strip()
    if resume in ("", "0"):
        if resume_dir:
            raise ValueError("--resume-dir requires --resume auto")
        return {}
    if resume == "auto":
        if not resume_dir:
            raise ValueError("--resume auto requires --resume-dir <run_id>")
        return {
            "RESUME": "auto",
            "RESUME_DIR": _runs_volume_path(resume_dir, "--resume-dir"),
        }
    if resume_dir:
        raise ValueError("--resume-dir is only valid with --resume auto")
    checkpoint = _runs_volume_path(resume, "--resume")
    name = PurePosixPath(checkpoint).name
    if not _COMMON_CHECKPOINT.fullmatch(name):
        raise ValueError("--resume must name a ckpt_*.pt file, or be 'auto'")
    return {"RESUME": checkpoint}


def _training_overrides(overrides: dict[str, str], git_sha: str,
                        git_dirty: bool, resume: str,
                        resume_dir: str) -> dict[str, str]:
    """Build the complete, testable environment overlay for the trainer."""
    result = dict(overrides)
    result.update(_resume_overrides(resume, resume_dir))
    result["GIT_SHA"] = git_sha or "unknown"
    result["GIT_DIRTY"] = "1" if git_dirty else "0"
    return result


def _validate_resume_target(overrides: dict[str, str]) -> None:
    """Fail before starting the trainer if the selected Volume artifact is absent."""
    resume = overrides.get("RESUME")
    if not resume:
        return
    if resume == "auto":
        resume_dir = Path(overrides["RESUME_DIR"])
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {resume_dir}")
        if not any(_COMMON_CHECKPOINT.fullmatch(p.name)
                   for p in resume_dir.glob("ckpt_*.pt")):
            raise FileNotFoundError(f"No ckpt_*.pt files in resume directory: {resume_dir}")
    elif not Path(resume).is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume}")


def _wandb_overrides(enabled: bool, project: str) -> dict[str, str]:
    """Env for the trainer's optional tracker.

    WANDB_API_KEY arrives from the attached Secret and is inherited via
    dict(os.environ). WANDB / WANDB_PROJECT cannot be: they are set on the
    LOCAL machine at `modal run` time and are absent inside the container, so
    they must travel as explicit function arguments (same reason git provenance
    does). Returns {} when disabled, so the trainer never imports wandb.
    """
    return {"WANDB": "1", "WANDB_PROJECT": project} if enabled else {}


def _build_env(overrides: dict[str, str], gpu_tag: str) -> dict[str, str]:
    env = dict(os.environ)
    # CACHE_BUST namespaces the inductor/triton cache under an extra path
    # segment, giving a provably cold compile without deleting a shared Volume.
    # Two uses: isolating a suspected poisoned cache entry (MODAL.md warns a
    # container killed mid-compile can leave one), and honest wall-clock for a
    # run you intend to publish, where first-container compile time counts.
    cache_bust = overrides.pop("CACHE_BUST", "")

    # torchrun vars. train_baseline.py already defaults these to single-process
    # values via setdefault, so this is belt-and-braces: it makes the intended
    # topology explicit in the process environment (and therefore in the run
    # log) rather than implicit in the script's fallbacks.
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("RANK", "0")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")

    # Compile cache on a Volume. Upside: inductor+triton autotuning for this
    # model is a 1-4 minute cold start that a warm cache reduces to seconds,
    # and that cost is paid per container.
    #
    # Tradeoffs, both real:
    #   * The cache is keyed by torch version, GPU capability, and graph hash,
    #     so it is *supposed* to be safe — but a half-written entry from a
    #     container that was killed mid-compile has been known to produce
    #     baffling errors. If a run fails inside inductor, delete the volume
    #     (`modal volume delete k3mini-torch-cache`) before debugging anything
    #     else.
    #   * Two runs compiling concurrently write the same keys. Benign in
    #     practice, but it is a shared mutable directory, so a run whose
    #     wall-clock you intend to *publish* should use a fresh cache: the
    #     first-container compile time is part of the honest number.
    # Namespaced per GPU so an L4 (sm89) smoke and an A100 (sm80) run cannot
    # interleave entries even if the keying ever regresses.
    _ns = f"{gpu_tag}/{cache_bust}" if cache_bust else gpu_tag
    env["TORCHINDUCTOR_CACHE_DIR"] = f"{CACHE_MOUNT}/inductor/{_ns}"
    env["TRITON_CACHE_DIR"] = f"{CACHE_MOUNT}/triton/{_ns}"
    os.makedirs(env["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
    os.makedirs(env["TRITON_CACHE_DIR"], exist_ok=True)

    env.update(overrides)
    return env


def _prepare_tree() -> None:
    """Make the container filesystem match what train_baseline.py expects."""
    shards = sorted(Path(DATA_MOUNT).glob("gutenberg_train_*.bin"))
    manifest = Path(DATA_MOUNT) / "manifest.json"
    if not shards:
        raise RuntimeError(
            f"No gutenberg_train_*.bin under {DATA_MOUNT}. The data Volume is "
            "empty or was never populated — see MODAL.md, 'One-time data upload'."
        )
    if not manifest.exists():
        raise RuntimeError(
            f"{manifest} missing. manifest.json lives *inside* the shards "
            "directory and config.validate_against_shards() reads it, so it "
            "must be uploaded to the Volume alongside the .bin files."
        )
    print(f"[modal] {len(shards)} train shard(s), "
          f"{sum(p.stat().st_size for p in shards) / 1e9:.2f} GB")

    # train_baseline.py does os.makedirs("logs") in CWD and writes
    # logs/<uuid>.txt there. That is container-local and would evaporate, so
    # point it into the runs Volume. makedirs(exist_ok=True) is happy with a
    # symlink to an existing directory.
    logs_target = Path(RUNS_MOUNT) / "_logs"
    logs_target.mkdir(parents=True, exist_ok=True)
    logs_link = Path(REPO_DIR) / "logs"
    if not logs_link.exists():
        logs_link.symlink_to(logs_target, target_is_directory=True)


def _start_committer() -> tuple[threading.Event, threading.Thread]:
    """Commit the runs Volume periodically.

    Without this, a run that hits the timeout or dies at step 3000 takes its
    recent logs and checkpoints with it, including the artifacts needed to
    diagnose the failure or resume from the latest committed checkpoint.
    """
    stop = threading.Event()

    def loop():
        while not stop.wait(COMMIT_INTERVAL_S):
            try:
                runs_vol.commit()
            except Exception as e:  # a failed commit must not kill training
                print(f"[modal] periodic commit failed (continuing): {e!r}")

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop, t


def _write_provenance(record: dict) -> None:
    """Launcher-side audit record, redundant with the trainer run registry.

    GIT_SHA/GIT_DIRTY are forwarded to train_baseline.py for its authoritative
    registry entry. This sidecar also records Modal GPU/runtime/exit facts.
    """
    path = Path(RUNS_MOUNT) / "modal_provenance.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _run_training(overrides: dict[str, str], gpu_tag: str, git_sha: str,
                  git_dirty: bool, resume: str = "",
                  resume_dir: str = "") -> dict:
    import torch  # imported here: the launcher's local import must not need it

    src = (Path(REPO_DIR) / "train_baseline.py").read_text(encoding="utf-8")
    _preflight_source(src)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible — check the gpu= argument.")
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    if cap[0] < 8:
        raise RuntimeError(
            f"{name} is compute capability {cap[0]}.{cap[1]}; this model is "
            "bf16 throughout and needs Ampere (sm80) or newer. T4/V100 will not do."
        )
    print(f"[modal] {name} (sm{cap[0]}{cap[1]}), torch {torch.__version__}")

    data_vol.reload()   # pick up shards uploaded after this container started
    runs_vol.reload()   # pick up a checkpoint committed by an earlier worker
    _prepare_tree()

    effective_overrides = _training_overrides(
        overrides, git_sha, git_dirty, resume, resume_dir
    )
    _validate_resume_target(effective_overrides)

    env = _build_env(effective_overrides, gpu_tag)
    print("[modal] overrides: " + " ".join(
        f"{k}={v}" for k, v in sorted(effective_overrides.items())
    ))

    stop, thread = _start_committer()
    t0 = time.time()
    rc = -1
    try:
        # A subprocess, not an import. train_baseline.py is a top-level script:
        # it trains at import time, reads its own source via sys.argv[0], calls
        # init_process_group / destroy_process_group at module scope, and
        # resolves every output path against CWD. A subprocess gives exact
        # control over CWD and env and keeps the process group teardown out of
        # the Modal worker process. No extra argv: the script reads
        # sys.argv[-1] as a trial count when len(argv) > 1.
        rc = subprocess.run(
            [sys.executable, "-u", "train_baseline.py"],
            cwd=REPO_DIR,
            env=env,
            check=False,
        ).returncode
    finally:
        stop.set()
        thread.join(timeout=10)
        elapsed = round(time.time() - t0, 1)
        _write_provenance({
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_sha": git_sha or "unknown",
            "git_dirty": git_dirty,
            "gpu": name,
            "gpu_tag": gpu_tag,
            "torch": torch.__version__,
            "env_overrides": effective_overrides,
            "exit_code": rc,
            "wall_clock_s": elapsed,
        })
        # Final commit. The periodic thread covers crashes; this covers the
        # last few seconds, including the final checkpoint and index write.
        runs_vol.commit()
        print(f"[modal] exit={rc} wall_clock={elapsed}s (runs volume committed)")

    if rc != 0:
        raise RuntimeError(f"train_baseline.py exited {rc}")
    return {"exit_code": rc, "wall_clock_s": elapsed, "gpu": name}


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------
#
# retries=0 on every training function, deliberately.
#
# Modal's retry would relaunch without knowing which prior run/checkpoint is a
# safe recovery source. Training is not idempotent, and a blind relaunch can
# create another paid experiment plus another registry entry. Inspect the
# failure and resume explicitly instead.

@app.function(
    gpu=SMOKE_GPU,
    timeout=SMOKE_TIMEOUT,
    volumes=VOLUMES,
    secrets=SECRETS,
    cpu=4.0,
    memory=16384,   # the loader pins a whole 200 MB shard; leave real headroom
    retries=0,
)
def smoke(train_steps: int = 100, resume: str = "", resume_dir: str = "",
          checkpoint_every: int = 0, sample_every: int = 0, val_every: int = 0,
          compile: bool = True, adamw_fused: bool = True, cache_bust: str = "",
          muon_compile: bool = True, init_seed: int = 0, mix: str = "",
          wandb: bool = WANDB_ENABLED, wandb_project: str = WANDB_PROJECT,
          git_sha: str = _GIT_SHA, git_dirty: bool = _GIT_DIRTY) -> dict:
    """100 steps on a cheap L4 — buys the CUDA-only code paths for cents.

    Everything that cannot be exercised on a CPU box gets touched here:
    bf16 params and the bf16 Newton-Schulz iteration, `pin_memory=True` in the
    shard loader, `non_blocking=True` H2D copies, fused AdamW (CUDA-only
    kernel), the gloo-on-CUDA collectives, TorchInductor codegen, and the
    checkpoint + sample path at the end of the run.

    Note COMPILE=1 is forced. train_baseline.py defaults COMPILE off under
    SMOKE, which is right for a ten-step CPU check and wrong here — inductor
    cold-start is one of the main things a pre-flight smoke exists to de-risk.

    The launcher restores the real 524,288-token batch so the control learning
    rates keep their intended LR-to-batch relationship; SMOKE still limits the
    validation slice. This is a plumbing test, not a loss measurement.
    """
    return _run_training(
        {
            "SMOKE": "1",
            "TRAIN_STEPS": str(train_steps),
            "COMPILE": "1" if compile else "0",
            "ADAMW_FUSED": "1" if adamw_fused else "0",
            "MUON_COMPILE": "1" if muon_compile else "0",
            **({"MIX": mix} if mix else {}),
            **({"INIT_SEED": str(init_seed)} if init_seed else {}),
            **({"CACHE_BUST": cache_bust} if cache_bust else {}),
            # Deliberately the REAL batch size, not SMOKE's shrunken default.
            # The LRs in config.py (embed 0.7, Muon 0.025) are tuned for 524288
            # tokens/step; at 64Ki they are ~8x too aggressive per token and the
            # run reaches val_loss=nan inside 100 steps. That divergence then
            # surfaced as a device-side assert in torch.multinomial during
            # sampling, which reads like a CUDA bug and is not one.
            # Memory is unaffected: it is set by MBS (8 x 1024 tokens per
            # microbatch), so a larger batch only means more microbatches per
            # step -- ~8x the step time on a run that took 1.6s of compute.
            "BATCH_SIZE": str(524288),
            "MBS": "8",
            "VAL_TOKENS": str(64 * 8192),
            # 0 means "leave config.py's value alone". Exposed so a resume
            # equivalence test can force a checkpoint partway through a short
            # run -- otherwise the only checkpoint is the final one and the
            # resume path cannot be exercised at all.
            **({"CHECKPOINT_EVERY": str(checkpoint_every)} if checkpoint_every else {}),
            **({"SAMPLE_EVERY": str(sample_every)} if sample_every else {}),
            **({"VAL_EVERY": str(val_every)} if val_every else {}),
            **_wandb_overrides(wandb, wandb_project),
        },
        gpu_tag=SMOKE_GPU,
        git_sha=git_sha,
        git_dirty=git_dirty,
        resume=resume,
        resume_dir=resume_dir,
    )


@app.function(
    gpu=TRAIN_GPU,
    timeout=TRAIN_TIMEOUT,
    volumes=VOLUMES,
    secrets=SECRETS,
    cpu=8.0,
    memory=32768,
    retries=0,
)
def train(train_steps: int = 3250, mbs: int = 8, batch_size: int = 0,
          val_tokens: int = 0, compile: bool = True, adamw_fused: bool = True,
          mix: str = "",
          resume: str = "", resume_dir: str = "",
          checkpoint_every: int = 0, sample_every: int = 0, val_every: int = 0,
          stop_after: int = 0,
          muon_compile: bool = True, cache_bust: str = "", init_seed: int = 0,
          wandb: bool = WANDB_ENABLED, wandb_project: str = WANDB_PROJECT,
          git_sha: str = _GIT_SHA, git_dirty: bool = _GIT_DIRTY) -> dict:
    """The full baseline run. 3250 steps on an A100-40GB, ~3.2h, ~$9.

    LAUNCH THIS WITH --detach. If the worker later fails, resume explicitly
    from a committed checkpoint; detaching prevents a client disconnect from
    becoming the failure in the first place.

        modal run --detach modal_app.py::train

    Args mirror train_baseline.py's env vars; 0 means "leave the script's own
    default alone" (which comes from config.py, the single source of truth).

    MBS defaults to 8 rather than something larger because the loss head is
    the memory bound, not the model: logits are materialised in fp32 at
    (mbs*1024, 50304), i.e. 1.65 GB at mbs=8, and the logit-softcap arithmetic
    plus cross_entropy keep several of those alive at once. mbs=16 is roughly
    the ceiling on 40 GB and mbs=32 will OOM. Raise it only on an 80 GB card.
    Constraint from the script: 524288 % (1024*mbs) == 0, so mbs must divide 512.
    """
    overrides = {"TRAIN_STEPS": str(train_steps), "MBS": str(mbs),
                 "COMPILE": "1" if compile else "0",
                 "ADAMW_FUSED": "1" if adamw_fused else "0",
            "MUON_COMPILE": "1" if muon_compile else "0",
            **({"MIX": mix} if mix else {}),
            **({"INIT_SEED": str(init_seed)} if init_seed else {}),
            **({"CACHE_BUST": cache_bust} if cache_bust else {}),
                 **_wandb_overrides(wandb, wandb_project)}
    if batch_size:
        overrides["BATCH_SIZE"] = str(batch_size)
    if val_tokens:
        overrides["VAL_TOKENS"] = str(val_tokens)
    if checkpoint_every:
        overrides["CHECKPOINT_EVERY"] = str(checkpoint_every)
    if sample_every:
        overrides["SAMPLE_EVERY"] = str(sample_every)
    if val_every:
        overrides["VAL_EVERY"] = str(val_every)
    if mix:
        # PLAN.md §5.1.1. Empty means the control's single-stream loader.
        overrides["MIX"] = mix
    if stop_after:
        # Deliberately NOT folded into train_steps: see STOP_AFTER in
        # train_baseline.py. Leaves the LR schedule and the resume sizing guard
        # untouched so a resumed trajectory stays comparable to the original.
        overrides["STOP_AFTER"] = str(stop_after)
    return _run_training(overrides, gpu_tag=TRAIN_GPU, git_sha=git_sha,
                         git_dirty=git_dirty, resume=resume,
                         resume_dir=resume_dir)


@app.function(
    gpu=BIG_GPU,
    timeout=BIG_TIMEOUT,
    volumes=VOLUMES,
    secrets=SECRETS,
    cpu=8.0,
    memory=32768,
    retries=0,
)
def train_h100(train_steps: int = 3250, mbs: int = 16,
               resume: str = "", resume_dir: str = "",
               wandb: bool = WANDB_ENABLED, wandb_project: str = WANDB_PROJECT,
               git_sha: str = _GIT_SHA, git_dirty: bool = _GIT_DIRTY) -> dict:
    """Same run on an H100. Higher $/hr, fewer hours — often cheaper overall.

    See MODAL.md for the arithmetic. The catch is that a 124M model at
    mbs*1024 tokens per microbatch under-feeds an H100, so realised MFU is
    typically lower than on an A100 and the speedup is well short of the 3.2x
    peak-FLOP ratio. Worth one measured comparison, not an assumption. The
    80 GB of HBM does let MBS default to 16 here.
    """
    return _run_training(
        {"TRAIN_STEPS": str(train_steps), "MBS": str(mbs), "COMPILE": "1",
         **_wandb_overrides(wandb, wandb_project)},
        gpu_tag=BIG_GPU, git_sha=git_sha, git_dirty=git_dirty,
        resume=resume, resume_dir=resume_dir,
    )


# ---------------------------------------------------------------------------
# Data verification (upload itself is done with the CLI — see MODAL.md)
# ---------------------------------------------------------------------------

# Repo-contract tests, as opposed to runtime tests. These assert things about the
# working tree -- that requirements.txt and the Colab notebook agree with the
# verified Modal runtime, that runs/index.jsonl holds a well-formed milestone
# record -- so they are meaningless in the training image and are skipped there.
#
# They are NOT skipped because they fail. Satisfying them would mean baking
# runs/index.jsonl into the image, and that file changes after every training run,
# so the image hash would bust on every run and force a rebuild each time. Run
# these against a checkout instead: `python -m pytest tests/test_runtime_contract.py`
REPO_ONLY_TESTS = ("tests/test_runtime_contract.py",)


@app.function(timeout=15 * 60, cpu=2.0)
def run_tests(target: str = "tests", verbose: bool = False,
              include_repo_tests: bool = False) -> int:
    """Run the pytest suite inside the training image. CPU only — no GPU cost.

    Exists because of the standing rule that nothing meaningful runs on the dev
    box. A suite that only ever passed on Windows says nothing about the runtime
    that actually trains: different OS, different torch build, different default
    encoding.
    """
    import subprocess as sp
    # Split so several paths can be passed as one --target argument; pytest takes
    # them as separate positionals, not one space-containing path.
    cmd = [sys.executable, "-m", "pytest", *target.split(),
           "-q" if not verbose else "-v"]
    if not include_repo_tests:
        for t in REPO_ONLY_TESTS:
            cmd += ["--ignore", t]
        print(f"[tests] skipping repo-contract tests: {', '.join(REPO_ONLY_TESTS)} "
              f"(run them against a checkout)")
    print(f"[tests] {' '.join(cmd)} (cwd={REPO_DIR})")
    rc = sp.run(cmd, cwd=REPO_DIR).returncode
    print(f"[tests] exit={rc}")
    return rc


@app.local_entrypoint()
def tests(target: str = "tests", verbose: bool = False,
          include_repo_tests: bool = False):
    rc = run_tests.remote(target=target, verbose=verbose,
                          include_repo_tests=include_repo_tests)
    if rc != 0:
        raise SystemExit(rc)


@app.function(volumes={DATA_MOUNT: data_vol}, timeout=15 * 60, cpu=4.0)
def verify_data(checksums: bool = True) -> dict:
    """Check that the data Volume matches data/shards/manifest.json.

    This exists instead of an upload function. `modal volume put` already does
    chunked, resumable, retrying uploads of a 940 MB directory from the CLI; a
    hand-rolled Modal function that streams the same bytes as function
    arguments would be strictly worse (arg size limits, no resumable transfer, burns
    container time, and re-implements a solved problem). What the CLI does
    *not* do is tell you whether the bytes that landed are the bytes the
    manifest pins — the shard sha256 is the whole point of having a manifest,
    and an unverified corpus is how a "mix ablation" quietly becomes a
    comparison of two truncated files. So: upload with the CLI, verify here.
    """
    import hashlib

    data_vol.reload()
    mpath = Path(DATA_MOUNT) / "manifest.json"
    if not mpath.exists():
        return {"ok": False, "error": f"{mpath} not found — nothing uploaded?"}
    manifest = json.loads(mpath.read_text())

    results, ok = [], True
    for kind in ("train_shards", "val_shards"):
        for entry in manifest[kind]:
            # manifest.json records absolute *Windows* shard paths. Only the
            # basename is portable, and it is all that is needed: config.py
            # never reads shard_path, and the dataloader globs.
            name = entry["shard_path"].replace("\\", "/").rsplit("/", 1)[-1]
            p = Path(DATA_MOUNT) / name
            row = {"file": name, "kind": kind, "present": p.exists()}
            if p.exists():
                row["size_ok"] = p.stat().st_size == entry["file_size"]
                row["size"] = p.stat().st_size
                if checksums:
                    h = hashlib.sha256()
                    with open(p, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 22), b""):
                            h.update(chunk)
                    row["sha256_ok"] = h.hexdigest() == entry["sha256"]
            ok &= row.get("present", False) and row.get("size_ok", False) \
                and row.get("sha256_ok", True)
            results.append(row)
            print(f"[verify] {row}")

    summary = {
        "ok": bool(ok),
        "shards": results,
        "total_train_tokens": manifest["total_train_tokens"],
        "total_val_tokens": manifest["total_val_tokens"],
        "tokenizer": manifest["tokenizer"],
        "vocab_size": manifest["vocab_size"],
    }
    print(f"[verify] ok={summary['ok']}")
    return summary


# ---------------------------------------------------------------------------
# Result retrieval
# ---------------------------------------------------------------------------

_TEXT_CAP = 2 * 1024 * 1024   # per file
_TOTAL_CAP = 16 * 1024 * 1024  # a Modal return value is not a file transport


@app.function(volumes={RUNS_MOUNT: runs_vol}, timeout=10 * 60)
def collect_results(include_logs: bool = True) -> dict:
    """Return the small text artifacts from the runs Volume.

    Skips *.pt entirely — checkpoints are ~1.4 GB each and belong in
    `modal volume get`, not in a function return value.
    """
    runs_vol.reload()   # essential: a detached run is still appending
    root = Path(RUNS_MOUNT)
    wanted: list[Path] = []
    if (root / "index.jsonl").exists():
        wanted.append(root / "index.jsonl")
    if (root / "modal_provenance.jsonl").exists():
        wanted.append(root / "modal_provenance.jsonl")
    wanted += sorted(root.glob("*/samples.log"))
    if include_logs:
        # 24, not 5: a seed-band sweep (PLAN.md §4.1) needs every seed's val curve
        # in one fetch, and the registry keeps only the FINAL loss, so a truncated
        # log set silently yields a band computed over fewer seeds than were run.
        # Logs are ~52 KB, so 24 is ~1.3 MB against the 16 MB total cap.
        wanted += sorted((root / "_logs").glob("*.txt"),
                         key=lambda p: p.stat().st_mtime, reverse=True)[:24]

    out, total, skipped = {}, 0, []
    for p in wanted:
        data = p.read_bytes()[:_TEXT_CAP]
        if total + len(data) > _TOTAL_CAP:
            skipped.append(str(p.relative_to(root)))
            continue
        total += len(data)
        out[str(p.relative_to(root)).replace("\\", "/")] = data.decode("utf-8", "replace")

    checkpoints = sorted(str(p.relative_to(root)) for p in root.glob("*/ckpt_*.pt"))
    return {"files": out, "skipped": skipped, "checkpoints": checkpoints,
            "bytes": total}


@app.local_entrypoint()
def fetch(dest: str = "modal_out", include_logs: bool = True):
    """Pull runs/index.jsonl, samples.log and logs back to this machine.

        modal run modal_app.py::fetch
        modal run modal_app.py::fetch --dest ./pulled --no-include-logs

    Safe to run while a --detach'ed job is still training: collect_results()
    reloads the Volume first, so you get the log as of right now.

    Checkpoints are listed but not downloaded (~1.4 GB each). Grab one with:
        modal volume get k3mini-runs <run_id>/ckpt_03250.pt ./
    """
    result = collect_results.remote(include_logs=include_logs)
    out_dir = LOCAL_ROOT / dest if not os.path.isabs(dest) else Path(dest)
    for rel, text in result["files"].items():
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
    if not result["files"]:
        print("nothing to fetch — the runs volume has no index.jsonl or samples.log yet")
    for rel in result["skipped"]:
        print(f"SKIPPED (size cap): {rel}")
    for rel in result["checkpoints"]:
        print(f"on volume, not downloaded: {rel}")


@app.local_entrypoint()
def verify(checksums: bool = True):
    """Verify the uploaded shards against manifest.json.

        modal run modal_app.py::verify
        modal run modal_app.py::verify --no-checksums   # presence + size only
    """
    r = verify_data.remote(checksums=checksums)
    print(json.dumps({k: v for k, v in r.items() if k != "shards"}, indent=2))
    if not r.get("ok"):
        bad = [s for s in r.get("shards", [])
               if not (s.get("present") and s.get("size_ok", False)
                       and s.get("sha256_ok", True))]
        print("PROBLEMS:")
        for s in bad:
            print(f"  {s}")
        raise SystemExit(1)
    print(f"\nLocal shards for reference: {LOCAL_SHARDS}")
