"""
Tokenize the Gutenberg corpus into .bin shards.

Uses the GPT-2 tokenizer (tiktoken) for Phase 0 — see PLAN.md §5.4 on timing.
Shards are written in the llm.c / modded-nanogpt format:
  - 1024-byte header (256 x int32)
  - uint16 token payload

Three things this script is careful about, each of which was a bug:

1. Every document is prefixed with <|endoftext|> so the model sees an explicit
   boundary. Without it the dataloader's fixed 1024-token windows straddle book
   boundaries and the model learns transitions that do not exist. (cf. the
   modded-nanogpt record `2025-07-12_BosAlign`.)
2. Validation is held out as WHOLE BOOKS, chosen once and pinned to
   data/val_books.json, never as a tail slice of the concatenated stream — a
   tail slice is one book by whatever sorts last. (PLAN.md §5.5.3, §7.2.2)
3. Token ids are asserted below the uint16 ceiling before the cast, so a
   larger tokenizer cannot silently wrap ids into garbage at Milestone 8.
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from shard_writer import write_shard
from quality_filter import score_document, is_prose, THRESHOLDS
from dedup import (
    find_duplicates,
    resolve_manifest_books,
    JACCARD_THRESHOLD,
    MINHASH_CANDIDATE_FLOOR,
    N_PERM,
    BANDS,
    ROWS,
    SHINGLE,
    KEEP_MOD,
    MIN_SAMPLED_SHINGLES,
)
import slices


def load_manifest() -> list[dict]:
    """Load manifest entries in file order, keyed later by gutenberg_id."""
    entries = []
    if config.manifest_path.exists():
        with open(config.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def resolve_books(entries: list[dict]) -> list[dict]:
    """
    Deduplicate manifest entries by gutenberg_id (last wins) and keep only
    those whose text is actually on disk.
    """
    return resolve_manifest_books(entries, verbose=True)


def apply_quality_filter(books: list[dict]) -> tuple[list[dict], dict]:
    """
    Tier-1 heuristic filter (PLAN.md §5.5.2). Drops reference material —
    almanacs, gazetteers, biographical dictionaries, catalogues — that is not
    prose and would otherwise teach the model to produce tables.

    Returns (kept_books, decisions) where decisions maps id -> metrics+verdict
    so the outcome lands in the manifest and stays auditable.
    """
    kept, dropped, decisions = [], [], {}
    for b in books:
        text = b["path"].read_text(encoding="utf-8", errors="replace")
        m = score_document(text)
        ok, reasons = is_prose(m)
        decisions[b["gutenberg_id"]] = {
            "prose_filter_pass": ok,
            "prose_filter_reasons": reasons,
            "metrics": {k: (None if v == float("inf") else round(float(v), 5))
                        for k, v in m.items()},
        }
        (kept if ok else dropped).append(b)

    if dropped:
        drop_chars = sum(b["char_count"] for b in dropped)
        all_chars = sum(b["char_count"] for b in books) or 1
        print(f"  Tier-1 filter dropped {len(dropped)}/{len(books)} documents "
              f"({drop_chars/all_chars:.1%} of chars):")
        for b in dropped:
            why = ", ".join(decisions[b["gutenberg_id"]]["prose_filter_reasons"])
            print(f"    {b['gutenberg_id']}: {str(b['title'])[:44]}  [{why}]")
    return kept, decisions


def select_val_books(books: list[dict]) -> set[int]:
    """
    Choose whole books for validation, totalling ~val_frac of the corpus by
    characters (a cheap proxy for tokens, since we have not tokenized yet).

    The selection is written to config.val_books_path and reused on every
    subsequent run. That pinning is the whole point: the corpus is about to
    grow from hundreds of books to thousands, and a val set that reshuffles
    when the corpus grows makes every loss comparison across that boundary
    meaningless.
    """
    if config.val_books_path.exists():
        pinned = json.loads(config.val_books_path.read_text(encoding="utf-8"))
        val_ids = set(pinned["val_book_ids"])
        on_disk = {b["gutenberg_id"] for b in books}
        missing = val_ids - on_disk
        if missing:
            print(f"  WARNING: {len(missing)} pinned val books are missing: "
                  f"{sorted(missing)[:5]}...")
        print(f"  Using pinned val set: {len(val_ids)} books from "
              f"{config.val_books_path.name}")
        return val_ids

    total_chars = sum(b["char_count"] for b in books)
    target = total_chars * config.val_frac

    # Seeded shuffle so the choice is reproducible but uncorrelated with
    # filename / id order.
    rng = np.random.default_rng(config.val_split_seed)
    order = rng.permutation(len(books))

    # Satisfy BOTH the token-fraction target and a minimum book count, so the
    # val set always spans several authors. On a small corpus min_val_books
    # dominates; once the corpus is large val_frac does.
    val_ids, acc = set(), 0
    for idx in order:
        if acc >= target and len(val_ids) >= config.min_val_books:
            break
        b = books[int(idx)]
        val_ids.add(b["gutenberg_id"])
        acc += b["char_count"]

    if len(val_ids) < config.min_val_books:
        print(f"  WARNING: only {len(val_ids)} books available for val "
              f"(wanted >= {config.min_val_books})")

    payload = {
        "val_book_ids": sorted(val_ids),
        "val_titles": [b["title"] for b in books if b["gutenberg_id"] in val_ids],
        "seed": config.val_split_seed,
        "val_frac_target": config.val_frac,
        "val_frac_actual_chars": acc / total_chars if total_chars else 0.0,
        "note": "Whole-book holdout. Pinned: do not regenerate, or cross-corpus "
                "loss comparisons break. Delete this file only on a deliberate reset.",
    }
    config.val_books_path.parent.mkdir(parents=True, exist_ok=True)
    config.val_books_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Selected {len(val_ids)} val books "
          f"({acc/total_chars:.2%} of chars) -> {config.val_books_path.name}")
    return val_ids


def tokenize_books(books: list[dict], enc, val_ids: set[int]):
    """
    Tokenize each book into its own uint16 array, prefixed with <|endoftext|>.
    Returns (train_arrays, val_arrays, per_book_stats).

    Per-book arrays rather than one giant Python list: 44M boxed ints is ~1.4 GB
    of object overhead, and the corpus is meant to grow by 10-100x.
    """
    eot = config.eot_token
    assert enc.eot_token == eot, (
        f"tokenizer eot {enc.eot_token} != config.eot_token {eot}"
    )

    train_arrays, val_arrays, stats = [], [], {}

    for i, b in enumerate(books, 1):
        text = b["path"].read_text(encoding="utf-8", errors="replace")
        ids = enc.encode(text, allowed_special=set())

        # Guard before the cast. GPT-2 maxes at 50256 so this is slack today,
        # and is exactly the assert that saves us at Milestone 8.
        if ids:
            hi = max(ids)
            assert hi < config.uint16_ceiling, (
                f"{b['path'].name}: token id {hi} >= uint16 ceiling "
                f"{config.uint16_ceiling}; shard payload cannot represent it"
            )

        arr = np.empty(len(ids) + 1, dtype=np.uint16)
        arr[0] = eot                    # document separator / BOS
        arr[1:] = np.asarray(ids, dtype=np.uint16)

        split = "val" if b["gutenberg_id"] in val_ids else "train"
        (val_arrays if split == "val" else train_arrays).append(arr)

        sha = hashlib.sha256(b["path"].read_bytes()).hexdigest()
        stats[b["gutenberg_id"]] = {
            "token_count": int(len(arr)),   # includes the leading EOT
            "char_count": b["char_count"],
            "split": split,
            "sha256": sha,
        }

        if i % 25 == 0 or i == len(books):
            print(f"  [{i}/{len(books)}] {b['title'][:48] if b['title'] else '?'} "
                  f"({len(arr):,} tokens, {split})")

    return train_arrays, val_arrays, stats


def concat(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty(0, dtype=np.uint16)
    return np.concatenate(arrays)


def write_shards(tokens: np.ndarray, split_name: str) -> list[dict]:
    """Write tokens to one or more .bin shards."""
    if len(tokens) == 0:
        print(f"  WARNING: no {split_name} tokens, writing no shards")
        return []

    config.shard_dir.mkdir(parents=True, exist_ok=True)
    shard_size = config.shard_size
    num_shards = (len(tokens) + shard_size - 1) // shard_size
    print(f"Writing {num_shards} {split_name} shard(s)...")

    entries = []
    for i in range(num_shards):
        chunk = tokens[i * shard_size:(i + 1) * shard_size]
        path = config.shard_dir / f"{split_name}_{i:03d}.bin"
        entry = write_shard(chunk, path, num_tokens=len(chunk))
        entries.append(entry)
        print(f"  Shard {i+1}/{num_shards}: {entry['num_tokens']:,} tokens, "
              f"{entry['file_size'] / 1e6:.1f} MB")
    return entries


def update_manifest(stats: dict, decisions: dict, dup_drop: set = None,
                    dup_clusters: dict = None, slice_of: dict = None):
    """Backfill token_count / split / sha256 / filter verdict / slice, atomically."""
    entries = load_manifest()
    dup_drop = dup_drop or set()
    dup_clusters = dup_clusters or {}
    slice_of = slice_of or {}
    # invert cluster map so each dropped id records what it duplicated
    dup_of = {loser: rep for rep, losers in dup_clusters.items() for loser in losers}
    updated = 0
    for e in entries:
        gid = e["gutenberg_id"]
        e["slice"] = slice_of.get(gid, e.get("slice"))
        e["is_near_duplicate"] = gid in dup_drop
        e["duplicate_of"] = dup_of.get(gid)
        d = decisions.get(e["gutenberg_id"])
        if d is not None:
            e["prose_filter_pass"] = d["prose_filter_pass"]
            e["prose_filter_reasons"] = d["prose_filter_reasons"]
            e["quality_metrics"] = d["metrics"]

        s = stats.get(e["gutenberg_id"])
        if s is None:
            # Not tokenized this run: filtered out, in-copyright, or missing.
            # Overwrite rather than setdefault — a stale split from an earlier
            # run would otherwise claim a document is in the training data when
            # it no longer is, which is precisely the drift the manifest exists
            # to make impossible.
            e["split"] = None
            e["token_count"] = None
            continue
        e["token_count"] = s["token_count"]
        e["split"] = s["split"]
        e["sha256"] = s["sha256"]
        e["tokenizer"] = config.tokenizer_name
        updated += 1

    tmp = config.manifest_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    tmp.replace(config.manifest_path)
    print(f"  Backfilled token_count/split/sha256 for {updated} manifest entries")


def main():
    print(f"Loading {config.tokenizer_name} tokenizer (tiktoken)...")
    enc = tiktoken.get_encoding(config.tokenizer_name)
    print(f"  Tokenizer n_vocab: {enc.n_vocab}, padded vocab_size: {config.vocab_size}")
    assert enc.n_vocab <= config.vocab_size, (
        f"tokenizer n_vocab {enc.n_vocab} exceeds config.vocab_size {config.vocab_size}"
    )

    entries = load_manifest()
    books = resolve_books(entries)
    print(f"Corpus: {len(books)} books on disk "
          f"({sum(b['char_count'] for b in books) / 1e6:.1f} MB of text)")
    if not books:
        print("Nothing to tokenize. Run data/download_gutenberg.py first.")
        return

    # §5.5 orders the pipeline "by the order things cause damage", and dedup is
    # first. Candidate-cluster members are tier-1-scored just early enough to
    # prevent a dirty longest edition from displacing a clean copy; the full
    # quality-filter pass still happens after dedup.
    print(f"Near-duplicate detection (MinHash, >= {JACCARD_THRESHOLD} Jaccard)...")
    dup_drop, dup_clusters = find_duplicates(books)
    if dup_drop:
        dropped_chars = sum(b["char_count"] for b in books
                            if b["gutenberg_id"] in dup_drop)
        books = [b for b in books if b["gutenberg_id"] not in dup_drop]
        print(f"  dropped {len(dup_drop)} near-duplicates "
              f"({dropped_chars/1e6:.1f} MB); {len(books)} remain")

    print("Applying tier-1 quality filter...")
    books, decisions = apply_quality_filter(books)
    print(f"  {len(books)} documents remain "
          f"({sum(b['char_count'] for b in books) / 1e6:.1f} MB)")
    if not books:
        print("Filter removed everything — check THRESHOLDS in quality_filter.py")
        return

    # Slice classification (§5.1): records which documents are the register slice
    # so its share can be measured now and upweighted in Phase 1.
    overrides = slices.load_overrides()
    by_id = {e["gutenberg_id"]: e for e in entries}
    slice_of = {b["gutenberg_id"]:
                slices.classify(by_id.get(b["gutenberg_id"], b), overrides)
                for b in books}
    n_reg = sum(1 for v in slice_of.values() if v == slices.REGISTER)
    print(f"  slices: {len(books) - n_reg} backbone, {n_reg} register "
          f"({len(overrides)} manual override(s))")

    print("Selecting validation books...")
    val_ids = select_val_books(books)

    print(f"Tokenizing {len(books)} books...")
    train_arrays, val_arrays, stats = tokenize_books(books, enc, val_ids)

    train_tokens = concat(train_arrays)
    val_tokens = concat(val_arrays)
    print(f"Train: {len(train_tokens):,} tokens across {len(train_arrays)} books")
    print(f"Val:   {len(val_tokens):,} tokens across {len(val_arrays)} books")

    if len(val_tokens) == 0:
        print("  WARNING: empty validation split — check val_frac and corpus size")

    # Per-slice token accounting, from the same stats the shards were built from.
    slice_tokens = {slices.BACKBONE: 0, slices.REGISTER: 0}
    for gid, s in stats.items():
        if s["split"] == "train":
            slice_tokens[slice_of.get(gid, slices.BACKBONE)] += s["token_count"]
    _tt = sum(slice_tokens.values()) or 1
    print(f"Slice shares of training tokens:")
    for name, tk in slice_tokens.items():
        print(f"  {name:9s} {tk/1e6:8.2f}M  {tk/_tt:6.1%}")
    if slice_tokens[slices.REGISTER] / _tt < 0.10:
        print(f"  NOTE: register slice below §5.1's 10-25% band; upweight it in the "
              f"mix or fetch more register authors "
              f"(data/download_gutenberg.py --canon)")

    train_shards = write_shards(train_tokens, "gutenberg_train")
    val_shards = write_shards(val_tokens, "gutenberg_val")

    shard_manifest = config.shard_dir / "manifest.json"
    with open(shard_manifest, "w", encoding="utf-8") as f:
        json.dump({
            "train_shards": train_shards,
            "val_shards": val_shards,
            "total_train_tokens": int(len(train_tokens)),
            "total_val_tokens": int(len(val_tokens)),
            "num_train_books": len(train_arrays),
            "num_val_books": len(val_arrays),
            "tokenizer": config.tokenizer_name,
            "vocab_size": config.vocab_size,
            "eot_token": config.eot_token,
            "doc_separator": "eot_prefix_per_document",
            "val_split": "whole_book_holdout",
            "val_books_path": str(config.val_books_path.name),
            "quality_filter": "tier1_heuristic",
            "quality_thresholds": THRESHOLDS,
            "dedup": {
                "method": "minhash_lsh_plus_blockwise_prefilter",
                "verification": "full_word_5gram_jaccard",
                "jaccard_threshold": JACCARD_THRESHOLD,
                "minhash_candidate_floor": MINHASH_CANDIDATE_FLOOR,
                "num_permutations": N_PERM,
                "bands": BANDS,
                "rows_per_band": ROWS,
                "shingle_words": SHINGLE,
                "content_sample_modulus": KEEP_MOD,
                "sparse_exact_candidate_below": MIN_SAMPLED_SHINGLES,
                "clusters": len(dup_clusters),
                "documents_dropped": len(dup_drop),
            },
            "slices": slice_tokens,
        }, f, indent=2)

    print(f"\nShard manifest written to {shard_manifest}")
    update_manifest(stats, decisions, dup_drop, dup_clusters, slice_of)

    # A corpus this small relative to the training budget is the real gate on
    # Milestone 1 — surface it rather than letting it be discovered on a GPU.
    full_run = config.train_steps * config.batch_size
    if len(train_tokens) > 0:
        epochs = full_run / len(train_tokens)
        print(f"\nA full baseline run is {full_run:,} tokens = {epochs:.1f} epochs "
              f"over this corpus.")
        if epochs > 4:
            print(f"  NOTE: past the ~4-epoch point where repetition stops paying "
                  f"(PLAN.md §5.6). Want ~{full_run / 4 / 1e6:.0f}M+ unique tokens; "
                  f"download more books.")


if __name__ == "__main__":
    main()
