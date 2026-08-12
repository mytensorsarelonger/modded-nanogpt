"""
Build a portable, reduced Colab smoke bundle from active token shards.

The default bundle contains the first train shard, the first validation shard,
an honest reduced ``data/shards/manifest.json``, and the source files required to
run ``smoke_test.py`` / ``train_baseline.py`` from the bundle root.

Examples:

    python data/prepare_colab_smoke.py
    python data/prepare_colab_smoke.py --output C:/tmp/k3-colab-smoke --zip
    python data/prepare_colab_smoke.py --copy --force

Existing output is never replaced without ``--force``. Shards are copied by
default so modifying the bundle cannot modify the active corpus; ``--hardlink``
is an explicit local-space-saving option.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import struct
import sys
import uuid
import zipfile
from array import array
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "shards" / "manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "colab_smoke_bundle"
DEFAULT_SOURCE_FILES = (
    "train_baseline.py",
    "config.py",
    "probes.py",
    "training_state.py",
    "smoke_test.py",
    "requirements.txt",
    "data/manifest.jsonl",
    "data/val_books.json",
)

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_BYTES = 1024
SCAN_CHUNK_BYTES = 8 * 1024 * 1024


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_shard_path(entry: dict, manifest_path: Path) -> Path:
    """Resolve absolute, manifest-relative, and portable bundle-style paths."""
    raw = entry.get("shard_path")
    if not raw:
        raise ValueError(f"shard entry has no shard_path: {entry!r}")
    given = Path(raw)
    candidates = []
    if given.is_absolute():
        candidates.append(given)
    else:
        candidates.extend((
            manifest_path.parent / given,
            manifest_path.parent / given.name,
            manifest_path.parent.parent.parent / given,
            Path.cwd() / given,
        ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"cannot resolve shard {raw!r}; tried: {tried}")


def inspect_shard(path: Path, eot_token: int, expected: dict | None = None) -> dict:
    """Validate one shard and count document-start EOT markers in its payload."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        header = f.read(SHARD_HEADER_BYTES)
        if len(header) != SHARD_HEADER_BYTES:
            raise ValueError(f"{path}: truncated shard header ({len(header)} bytes)")
        digest.update(header)
        values = struct.unpack("<256i", header)
        magic, version, num_tokens = values[:3]
        if magic != SHARD_MAGIC:
            raise ValueError(f"{path}: magic {magic} != {SHARD_MAGIC}")
        if version != SHARD_VERSION:
            raise ValueError(f"{path}: version {version} != {SHARD_VERSION}")
        if num_tokens < 0:
            raise ValueError(f"{path}: negative token count {num_tokens}")

        payload_bytes = 0
        document_starts = 0
        while chunk := f.read(SCAN_CHUNK_BYTES):
            if len(chunk) % 2:
                raise ValueError(f"{path}: odd uint16 payload chunk")
            digest.update(chunk)
            payload_bytes += len(chunk)
            tokens = array("H")
            tokens.frombytes(chunk)
            if tokens.itemsize != 2:
                raise RuntimeError(f"platform unsigned-short width is {tokens.itemsize}, need 2")
            if sys.byteorder != "little":
                tokens.byteswap()
            document_starts += tokens.count(eot_token)

    file_size = path.stat().st_size
    expected_size = SHARD_HEADER_BYTES + num_tokens * 2
    if payload_bytes != num_tokens * 2 or file_size != expected_size:
        raise ValueError(
            f"{path}: header claims {num_tokens} tokens/{expected_size} bytes, "
            f"found {payload_bytes // 2} tokens/{file_size} bytes"
        )

    actual = {
        "num_tokens": num_tokens,
        "file_size": file_size,
        "sha256": digest.hexdigest(),
        "magic": magic,
        "version": version,
        "document_starts": document_starts,
    }
    if expected is not None:
        for key in ("num_tokens", "file_size", "sha256", "magic", "version"):
            if key in expected and expected[key] != actual[key]:
                raise ValueError(
                    f"{path}: manifest {key}={expected[key]!r}, "
                    f"actual={actual[key]!r}"
                )
    return actual


def rewrite_manifest(
    source: dict,
    train_entries: list[dict],
    val_entries: list[dict],
    *,
    train_document_starts: int,
    val_document_starts: int,
    source_manifest_sha256: str,
) -> dict:
    """Return an honest reduced manifest while preserving preparation recipes."""
    if not train_entries:
        raise ValueError("smoke bundle needs at least one train shard")
    if not val_entries:
        raise ValueError("smoke bundle needs at least one validation shard")

    reduced = copy.deepcopy(source)
    source_stats = {
        key: copy.deepcopy(source[key])
        for key in (
            "total_train_tokens",
            "total_val_tokens",
            "num_train_books",
            "num_val_books",
            "slices",
        )
        if key in source
    }

    reduced["train_shards"] = copy.deepcopy(train_entries)
    reduced["val_shards"] = copy.deepcopy(val_entries)
    reduced["total_train_tokens"] = sum(e["num_tokens"] for e in train_entries)
    reduced["total_val_tokens"] = sum(e["num_tokens"] for e in val_entries)
    reduced["num_train_books"] = train_document_starts
    reduced["num_val_books"] = val_document_starts

    # Per-slice token totals describe the full source corpus and cannot be
    # reconstructed from token ids alone. Keep them as clearly labeled source
    # provenance rather than lying in the reduced manifest's primary fields.
    reduced.pop("slices", None)

    dedup = reduced.get("dedup")
    if isinstance(dedup, dict):
        dedup_outcome = {}
        for key in ("clusters", "documents_dropped"):
            if key in dedup:
                dedup_outcome[key] = dedup.pop(key)
        if dedup_outcome:
            source_stats["dedup_outcome"] = dedup_outcome

    reduced["source_corpus"] = {
        "shard_manifest_sha256": source_manifest_sha256,
        **source_stats,
    }
    reduced["smoke_bundle"] = {
        "kind": "colab_smoke_subset",
        "selection": "first_n_shards",
        "selected_train_shards": len(train_entries),
        "selected_val_shards": len(val_entries),
        "shard_paths": "bundle_root_relative",
        "book_count_basis": "document-start eot_token markers in selected payloads",
    }
    return reduced


def _portable_entry(entry: dict, actual: dict, destination: Path) -> dict:
    rewritten = copy.deepcopy(entry)
    rewritten.update({
        "shard_path": destination.as_posix(),
        "num_tokens": actual["num_tokens"],
        "file_size": actual["file_size"],
        "sha256": actual["sha256"],
        "magic": actual["magic"],
        "version": actual["version"],
    })
    return rewritten


def _materialize_shard(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {"auto", "hardlink", "copy"}:
        raise ValueError(f"unknown materialization mode {mode!r}")
    if mode != "copy":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def _safe_relative_source(repo_root: Path, relative: str | Path) -> tuple[Path, Path]:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"source files must be repo-relative: {relative}")
    source = (repo_root / rel).resolve()
    try:
        source.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"source escapes repository: {relative}") from exc
    if not source.is_file():
        raise FileNotFoundError(f"bundle source file not found: {source}")
    return source, rel


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _validate_output_target(
    path: Path,
    protected: Iterable[Path],
    forbidden_trees: Iterable[Path] = (),
) -> Path:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError(f"refusing filesystem-root output: {resolved}")
    for item in protected:
        protected_path = item.resolve()
        try:
            protected_path.relative_to(resolved)
            contains_protected = True
        except ValueError:
            contains_protected = False
        if resolved == protected_path or contains_protected:
            raise ValueError(f"refusing protected output path: {resolved}")
    for tree in forbidden_trees:
        tree = tree.resolve()
        try:
            resolved.relative_to(tree)
        except ValueError:
            continue
        raise ValueError(f"refusing output inside protected directory: {tree}")
    return resolved


def _install_directory(staging: Path, output: Path, force: bool) -> None:
    if not output.exists():
        staging.replace(output)
        return
    if not force:
        raise FileExistsError(f"output exists (use --force): {output}")

    backup = output.with_name(f".{output.name}.old-{uuid.uuid4().hex}")
    output.replace(backup)
    try:
        staging.replace(output)
    except Exception:
        backup.replace(output)
        raise
    else:
        _remove_path(backup)


def _write_readme(path: Path) -> None:
    path.write_text(
        """# Colab smoke bundle

This bundle contains a reduced shard set for exercising the real control-arm
trainer. Run commands from this directory after unzipping/uploading it.

```bash
python -m pip install -r requirements.txt
python smoke_test.py --steps 3 --batch-size 8192 --mbs 1 --timeout 1800
```

Equivalent direct invocation:

```bash
SMOKE=1 TRAIN_STEPS=3 BATCH_SIZE=8192 MBS=1 VAL_TOKENS=8192 COMPILE=0 python train_baseline.py
```

`data/shards/manifest.json` describes the actual reduced shard subset. The full
source `data/manifest.jsonl` is included as provenance, so its hash continues to
identify the corpus from which these shards were selected; it is not a claim
that every source document is present in this smoke subset. Full-corpus shard
statistics are retained under `source_corpus`, and recipe fields are unchanged.
""",
        encoding="utf-8",
    )


def create_zip(bundle_dir: Path, zip_path: Path, force: bool = False) -> Path:
    """Archive a completed bundle without ever partially overwriting a zip."""
    bundle_dir = bundle_dir.resolve()
    zip_path = zip_path.resolve()
    try:
        zip_path.relative_to(bundle_dir)
    except ValueError:
        pass
    else:
        raise ValueError("zip path must be outside the bundle directory")
    if zip_path.is_dir():
        raise IsADirectoryError(f"zip path is a directory: {zip_path}")
    if zip_path.exists() and not force:
        raise FileExistsError(f"zip exists (use --force): {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_name(f".{zip_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for file in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
                arcname = Path(bundle_dir.name) / file.relative_to(bundle_dir)
                archive.write(file, arcname.as_posix())
        if zip_path.exists():
            _remove_path(zip_path)
        temporary.replace(zip_path)
    except Exception:
        _remove_path(temporary)
        raise
    return zip_path


def build_bundle(
    source_manifest: Path,
    output_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
    train_count: int = 1,
    val_count: int = 1,
    materialization: str = "copy",
    source_files: Iterable[str | Path] = DEFAULT_SOURCE_FILES,
    force: bool = False,
    zip_path: Path | None = None,
) -> dict:
    """Build one directory bundle and optionally a zip; return a result summary."""
    source_manifest = Path(source_manifest).resolve()
    repo_root = Path(repo_root).resolve()
    output_dir = _validate_output_target(
        Path(output_dir),
        protected=(repo_root, repo_root / "data", source_manifest.parent),
        forbidden_trees=(repo_root / ".git", source_manifest.parent),
    )
    if zip_path is not None:
        zip_path = _validate_output_target(
            Path(zip_path),
            protected=(repo_root, repo_root / "data", source_manifest.parent),
            forbidden_trees=(repo_root / ".git", source_manifest.parent),
        )
        try:
            zip_path.relative_to(output_dir)
        except ValueError:
            pass
        else:
            raise ValueError("zip path must be outside the bundle directory")
        if zip_path.is_dir():
            raise IsADirectoryError(f"zip path is a directory: {zip_path}")
        if zip_path.exists() and not force:
            raise FileExistsError(f"zip exists (use --force): {zip_path}")
    if train_count < 1 or val_count < 1:
        raise ValueError("train_count and val_count must both be >= 1")
    if output_dir.exists() and not force:
        raise FileExistsError(f"output exists (use --force): {output_dir}")
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source shard manifest not found: {source_manifest}")

    manifest_bytes = source_manifest.read_bytes()
    source = json.loads(manifest_bytes)
    source_train = source.get("train_shards") or []
    source_val = source.get("val_shards") or []
    if len(source_train) < train_count:
        raise ValueError(f"requested {train_count} train shards, found {len(source_train)}")
    if len(source_val) < val_count:
        raise ValueError(f"requested {val_count} val shards, found {len(source_val)}")
    eot_token = int(source["eot_token"])

    selected = [
        ("train", entry) for entry in source_train[:train_count]
    ] + [
        ("val", entry) for entry in source_val[:val_count]
    ]
    inspected = []
    names = set()
    for split, entry in selected:
        shard = resolve_shard_path(entry, source_manifest)
        if shard.name in names:
            raise ValueError(f"duplicate selected shard basename: {shard.name}")
        names.add(shard.name)
        inspected.append((split, entry, shard, inspect_shard(shard, eot_token, entry)))

    sources = [_safe_relative_source(repo_root, rel) for rel in source_files]
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    if staging.exists():
        raise FileExistsError(f"staging path unexpectedly exists: {staging}")
    staging.mkdir(parents=True)
    link_counts = {"hardlink": 0, "copy": 0}
    try:
        rewritten_train, rewritten_val = [], []
        train_docs = val_docs = 0
        for split, entry, shard, actual in inspected:
            relative = Path("data") / "shards" / shard.name
            method = _materialize_shard(shard, staging / relative, materialization)
            link_counts[method] += 1
            portable = _portable_entry(entry, actual, relative)
            if split == "train":
                rewritten_train.append(portable)
                train_docs += actual["document_starts"]
            else:
                rewritten_val.append(portable)
                val_docs += actual["document_starts"]

        reduced = rewrite_manifest(
            source,
            rewritten_train,
            rewritten_val,
            train_document_starts=train_docs,
            val_document_starts=val_docs,
            source_manifest_sha256=_sha256_bytes(manifest_bytes),
        )
        manifest_destination = staging / "data" / "shards" / "manifest.json"
        manifest_destination.parent.mkdir(parents=True, exist_ok=True)
        manifest_destination.write_text(
            json.dumps(reduced, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        for source_file, relative in sources:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
        _write_readme(staging / "COLAB_SMOKE.md")

        _install_directory(staging, output_dir, force)
    except Exception:
        _remove_path(staging)
        raise

    archive = create_zip(output_dir, zip_path, force=force) if zip_path else None
    return {
        "output_dir": output_dir,
        "zip_path": archive,
        "train_shards": len(rewritten_train),
        "val_shards": len(rewritten_val),
        "train_tokens": reduced["total_train_tokens"],
        "val_tokens": reduced["total_val_tokens"],
        "train_books": reduced["num_train_books"],
        "val_books": reduced["num_val_books"],
        **link_counts,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"active shard manifest (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"bundle directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--train-count", type=int, default=1,
                        help="number of leading train shards to include (default: 1)")
    parser.add_argument("--val-count", type=int, default=1,
                        help="number of leading validation shards to include (default: 1)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true",
                      help="copy shards (safe default)")
    mode.add_argument("--hardlink", action="store_true",
                      help="require hardlinks; fail instead of falling back to copies")
    parser.add_argument("--source", action="append", default=[], metavar="REPO_PATH",
                        help="additional repo-relative source file to include (repeatable)")
    parser.add_argument("--data-only", action="store_true",
                        help="omit default trainer source files (COLAB_SMOKE.md remains)")
    parser.add_argument("--zip", action="store_true",
                        help="also create <output>.zip using ZIP_STORED")
    parser.add_argument("--zip-path", type=Path,
                        help="custom zip path (implies --zip)")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing bundle/zip after the new bundle is ready")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    materialization = "hardlink" if args.hardlink else "copy"
    source_files = [] if args.data_only else list(DEFAULT_SOURCE_FILES)
    source_files.extend(args.source)
    zip_path = args.zip_path
    if (args.zip or zip_path is not None) and zip_path is None:
        zip_path = Path(str(args.output) + ".zip")

    result = build_bundle(
        args.source_manifest,
        args.output,
        train_count=args.train_count,
        val_count=args.val_count,
        materialization=materialization,
        source_files=source_files,
        force=args.force,
        zip_path=zip_path,
    )
    print(f"Created {result['output_dir']}")
    print(
        f"  train: {result['train_shards']} shard(s), {result['train_tokens']:,} tokens, "
        f"{result['train_books']} document starts"
    )
    print(
        f"  val:   {result['val_shards']} shard(s), {result['val_tokens']:,} tokens, "
        f"{result['val_books']} document starts"
    )
    print(f"  shards: {result['hardlink']} hardlink(s), {result['copy']} copy/copies")
    if result["zip_path"]:
        print(f"  zip: {result['zip_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
