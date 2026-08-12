"""
Near-duplicate detection: MinHash + LSH banding (PLAN.md §5.5.1).

This runs BEFORE the quality filter, because dedup is step one of the pipeline
"in the order things cause damage" -- duplicates waste compute and inflate
memorisation, and they do it on the most famous texts in the corpus, which is
where memorisation is most visible and least excusable.

Exact hashing does not find these. The 3,431-book Gutenberg corpus had zero
byte-identical files but three separate editions each of Romeo and Juliet,
Frankenstein and Dorian Gray. Different transcriptions of the same work differ in
front matter, line breaks and editorial notes while being ~the same novel, which
is precisely the regime MinHash covers and SHA-256 does not.

Implementation notes, since the cheap version of this is subtly wrong:

  - Shingles are word 5-grams, not character n-grams at a stride. Strided
    sampling is offset-sensitive: two editions with different front-matter
    lengths land on different sample positions and the duplicate is missed.
  - Subsampling is CONTENT-DEFINED (keep shingles where hash % KEEP_MOD == 0).
    The same shingle always survives or always does not, independent of where it
    sits in the document, so the estimate stays offset-invariant.
  - Sparse documents never switch to a different signature universe. Instead,
    a cardinality bound sends every size-compatible sparse pair to exact checking.
  - A representative must survive tier 1 if any member can; among eligible
    editions, keep the longest on the assumption that it is the fuller text.
  - MinHash only proposes candidates. Removal requires >= JACCARD_THRESHOLD on
    the complete word-5-gram sets, and every loser is checked directly against
    its recorded representative (similarity is not transitive).
"""

import re
import sys
import json
import zlib
import math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config

N_PERM = 120           # signature length
BANDS = 12             # b * r == N_PERM; threshold ~= (1/b)^(1/r)
ROWS = 10              # (1/12)^(1/10) ~= 0.784, close to the ~0.8 in §5.5.1
SHINGLE = 5            # words per shingle
KEEP_MOD = 64          # content-defined subsample rate
JACCARD_THRESHOLD = 0.80
# LSH is a useful cheap candidate source, but 12x10 banding only proposes about
# 74% of pairs whose signature agreement is exactly 0.80.  Scan all signature
# pairs as well, at a deliberately lower floor; exact full-set verification below
# makes the extra candidates safe.
MINHASH_CANDIDATE_FLOOR = 0.50
CANDIDATE_BLOCK_SIZE = 128
# Documents below this content-sample size get a cardinality-bounded exact
# candidate path. They do NOT switch to a different signature universe.
MIN_SAMPLED_SHINGLES = 8
MERSENNE = (1 << 61) - 1

WORD_RE = re.compile(r"[a-z0-9']+")


def _shingle_hashes(text: str) -> np.ndarray | None:
    """Sorted unique hashes for every word 5-gram, or None when too short."""
    words = WORD_RE.findall(text.lower())
    if len(words) < SHINGLE * 4:
        return None

    # Words must map to GLOBALLY stable ids. Using np.unique's per-document
    # inverse indices here is a silent correctness bug: the same 5-gram gets
    # different ids in different documents, so no two signatures are comparable
    # and the whole pass reports zero duplicates. Hash the word STRING instead,
    # with crc32 rather than hash() -- the latter is salted per process, so it
    # would not even be stable across runs.
    arr = np.asarray(words)
    uniq, inv = np.unique(arr, return_inverse=True)
    uniq_h = np.fromiter((zlib.crc32(w.encode("utf-8")) for w in uniq),
                         dtype=np.uint64, count=len(uniq))
    ids = uniq_h[inv] + np.uint64(1)

    n = len(ids) - SHINGLE + 1
    h = np.zeros(n, dtype=np.uint64)
    base = np.uint64(1000003)
    for i in range(SHINGLE):
        h = h * base + ids[i:i + n]

    return np.unique(h)


def _signature_shingles(full_h: np.ndarray) -> np.ndarray:
    """The one shared content-defined universe used by every document."""
    return full_h[(full_h % np.uint64(KEEP_MOD)) == 0]


def _signature_from_hashes(full_h: np.ndarray, coeffs) -> np.ndarray:
    """Build one MinHash signature from a full sorted shingle-hash set."""
    h = _signature_shingles(full_h)

    # MERSENNE is outside the permutation output range [0, MERSENNE), so it is
    # an unambiguous +infinity sentinel for an empty content sample. Sparse docs
    # also receive a safe exact-candidate path in _sparse_candidates().
    if h.size == 0:
        return np.full(N_PERM, np.uint64(MERSENNE), dtype=np.uint64)

    # Content-defined subsample: keeps the estimate offset-invariant.
    a, b = coeffs
    # (a * h + b) mod Mersenne, min over shingles -> one value per permutation
    hh = (h.astype(object)[None, :] * a[:, None] + b[:, None]) % MERSENNE
    return np.array([int(x) for x in hh.min(axis=1)], dtype=np.uint64)


def _signature(text: str, rng: np.random.Generator, coeffs) -> np.ndarray | None:
    """120-length MinHash signature, or None if the document is too short."""
    del rng  # retained in the API for callers/tests written against the first version
    full_h = _shingle_hashes(text)
    if full_h is None or full_h.size == 0:
        return None
    return _signature_from_hashes(full_h, coeffs)


def _lsh_candidates(S: np.ndarray) -> set[tuple[int, int]]:
    """Candidate pairs sharing at least one complete LSH band."""
    candidates: set[tuple[int, int]] = set()
    n = len(S)
    for band in range(BANDS):
        rows = S[:, band * ROWS:(band + 1) * ROWS]
        buckets: dict[bytes, list[int]] = {}
        for i in range(n):
            buckets.setdefault(rows[i].tobytes(), []).append(i)
        for group in buckets.values():
            if len(group) > 1:
                for x in range(len(group)):
                    for y in range(x + 1, len(group)):
                        candidates.add((group[x], group[y]))
    return candidates


def _blockwise_candidates(
    S: np.ndarray,
    floor: float = MINHASH_CANDIDATE_FLOOR,
    block_size: int = CANDIDATE_BLOCK_SIZE,
) -> set[tuple[int, int]]:
    """Deterministically scan every signature pair without an NxN allocation."""
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"candidate floor must be in [0, 1], got {floor}")
    min_matches = math.ceil(floor * S.shape[1])
    candidates: set[tuple[int, int]] = set()
    n = len(S)
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        left = S[i0:i1]
        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            right = S[j0:j1]
            matches = np.count_nonzero(
                left[:, None, :] == right[None, :, :], axis=2
            )
            for li, rj in np.argwhere(matches >= min_matches):
                i, j = i0 + int(li), j0 + int(rj)
                if i < j:
                    candidates.add((i, j))
    return candidates


def _sparse_candidates(
    full_sizes: list[int], sample_sizes: list[int]
) -> set[tuple[int, int]]:
    """
    Safely propose sparse-sample pairs using the exact-set cardinality bound.

    J(A, B) cannot exceed min(|A|, |B|) / max(|A|, |B|). Pair every document
    with a sparse content sample against every size-compatible document, so a
    7-vs-8 (or 0-vs-N) sample boundary cannot hide a true >= threshold pair.
    """
    if len(full_sizes) != len(sample_sizes):
        raise ValueError("full_sizes and sample_sizes must have equal length")
    candidates: set[tuple[int, int]] = set()
    n = len(full_sizes)
    for i, sample_size in enumerate(sample_sizes):
        if sample_size >= MIN_SAMPLED_SHINGLES:
            continue
        for j in range(n):
            if i == j:
                continue
            lo, hi = sorted((full_sizes[i], full_sizes[j]))
            if hi and lo / hi >= JACCARD_THRESHOLD:
                candidates.add((min(i, j), max(i, j)))
    return candidates


def _candidate_pairs(
    S: np.ndarray,
    full_sizes: list[int] | None = None,
    sample_sizes: list[int] | None = None,
) -> set[tuple[int, int]]:
    """High-recall union of LSH and a conservative all-pairs prefilter."""
    candidates = _lsh_candidates(S) | _blockwise_candidates(S)
    if full_sizes is not None or sample_sizes is not None:
        if full_sizes is None or sample_sizes is None:
            raise ValueError("full_sizes and sample_sizes must be provided together")
        if len(full_sizes) != len(S):
            raise ValueError("signature and shingle-size counts must align")
        candidates |= _sparse_candidates(full_sizes, sample_sizes)
    return candidates


def _exact_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard over the complete (not content-sampled) shingle-hash sets."""
    intersection = int(np.intersect1d(a, b, assume_unique=True).size)
    union = int(a.size + b.size - intersection)
    return intersection / union if union else 1.0


def _book_quality_eligible(book: dict) -> bool:
    """Whether a book can survive tier 1; explicit test fixtures may override."""
    if "quality_eligible" in book:
        return bool(book["quality_eligible"])
    # Local import avoids making the standalone quality-report path depend on
    # dedup at module import time.
    from data.quality_filter import score_document, is_prose
    text = book["path"].read_text(encoding="utf-8", errors="replace")
    return is_prose(score_document(text))[0]


def _select_clusters(
    books: list[dict], confirmed_edges: set[tuple[int, int]]
) -> tuple[set[int], dict[int, list[int]]]:
    """
    Select disjoint star clusters from exact-similarity edges.

    A connected component is not itself safe: Jaccard is not transitive.  Each
    loser here is therefore a direct exact-verified neighbour of its recorded
    representative.  Prefer a tier-1-eligible representative, then the longest.
    """
    adjacency: dict[int, set[int]] = {}
    for i, j in confirmed_edges:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    if not adjacency:
        return set(), {}

    eligible = {i: _book_quality_eligible(books[i]) for i in adjacency}
    unseen = set(adjacency)
    components: list[set[int]] = []
    while unseen:
        start = min(unseen)
        component, stack = set(), [start]
        while stack:
            i = stack.pop()
            if i in component:
                continue
            component.add(i)
            unseen.discard(i)
            stack.extend(sorted(adjacency[i] - component, reverse=True))
        components.append(component)

    drop: set[int] = set()
    clusters: dict[int, list[int]] = {}
    for component in components:
        remaining = set(component)
        while remaining:
            # With no tier-1-eligible text there is nothing for dedup to preserve;
            # leave every member to the quality filter rather than recording one
            # unrelated metadata stub as another stub's representative.
            if not any(eligible[i] for i in remaining):
                break
            rep_i = min(
                remaining,
                key=lambda i: (
                    not eligible[i],
                    -books[i]["char_count"],
                    books[i]["gutenberg_id"],
                ),
            )
            losers_i = sorted(
                adjacency[rep_i] & remaining,
                key=lambda i: books[i]["gutenberg_id"],
            )
            remaining.remove(rep_i)
            remaining.difference_update(losers_i)
            if not losers_i:
                continue
            rep = books[rep_i]["gutenberg_id"]
            losers = [books[i]["gutenberg_id"] for i in losers_i]
            clusters[rep] = losers
            drop.update(losers)
    return drop, clusters


def find_duplicates(books: list[dict], verbose: bool = True) -> tuple[set[int], dict]:
    """
    Returns (ids_to_drop, clusters) where clusters maps
    representative_id -> [dropped_id, ...].

    `books` entries need "gutenberg_id", "path" and "char_count".
    """
    rng = np.random.default_rng(config.val_split_seed)
    a = rng.integers(1, MERSENNE, size=N_PERM, dtype=np.int64).astype(object)
    b = rng.integers(0, MERSENNE, size=N_PERM, dtype=np.int64).astype(object)

    sigs, kept_books = [], []
    full_sizes, sample_sizes = [], []
    for i, bk in enumerate(books, 1):
        text = bk["path"].read_text(encoding="utf-8", errors="replace")
        full_h = _shingle_hashes(text)
        if full_h is not None and full_h.size:
            sigs.append(_signature_from_hashes(full_h, (a, b)))
            kept_books.append(bk)
            full_sizes.append(int(full_h.size))
            sample_sizes.append(int(_signature_shingles(full_h).size))
        if verbose and (i % 500 == 0 or i == len(books)):
            print(f"    hashed {i}/{len(books)}")

    if not sigs:
        return set(), {}
    S = np.vstack(sigs)
    n = len(kept_books)

    candidates = _candidate_pairs(S, full_sizes, sample_sizes)

    # MinHash only proposes. Re-read the small candidate subset and make every
    # removal decision on complete word-5-gram sets.
    full_cache: dict[int, np.ndarray] = {}

    def full_hashes(i: int) -> np.ndarray:
        if i not in full_cache:
            text = kept_books[i]["path"].read_text(encoding="utf-8", errors="replace")
            h = _shingle_hashes(text)
            if h is None:
                h = np.empty(0, dtype=np.uint64)
            full_cache[i] = h
        return full_cache[i]

    confirmed_edges: set[tuple[int, int]] = set()
    for i, j in sorted(candidates):
        if _exact_jaccard(full_hashes(i), full_hashes(j)) >= JACCARD_THRESHOLD:
            confirmed_edges.add((i, j))

    drop, clusters = _select_clusters(kept_books, confirmed_edges)

    if verbose:
        print(f"    {len(candidates)} candidate pairs, {len(confirmed_edges)} exact-confirmed "
              f">= {JACCARD_THRESHOLD} full-set Jaccard")
        print(f"    {len(clusters)} duplicate clusters, {len(drop)} documents to drop")
    return drop, clusters


def resolve_manifest_books(entries: list[dict], verbose: bool = False) -> list[dict]:
    """Resolve the exact document population used by tokenization and dedup."""
    # Collapse first so the final record controls every field, including a final
    # failed/copyright status. Otherwise an older usable record can leak back in.
    by_id: dict[int, dict] = {}
    for e in entries:
        by_id[e["gutenberg_id"]] = e

    excluded_copyright = [
        (gid, e.get("title"))
        for gid, e in sorted(by_id.items())
        if e.get("status") != "failed" and e.get("copyright") is True
    ]

    if verbose and excluded_copyright:
        print(f"  EXCLUDED {len(excluded_copyright)} in-copyright work(s) per §6:")
        for gid, title in excluded_copyright:
            print(f"    {gid}: {title}")

    books, missing = [], 0
    for gid, e in sorted(by_id.items()):
        if e.get("status") == "failed" or e.get("copyright") is True:
            continue
        p = Path(e.get("cleaned_path") or e.get("local_path") or "")
        if not p.exists():
            p = config.gutenberg_raw / f"{gid}.txt"
        if not p.exists():
            missing += 1
            continue
        books.append({"gutenberg_id": gid, "title": e.get("title") or "?",
                      "path": p, "char_count": p.stat().st_size})
    if verbose and missing:
        print(f"  WARNING: {missing} manifest entries have no file on disk")
    return books


def _load_books_from_manifest() -> list[dict]:
    entries = []
    with open(config.manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return resolve_manifest_books(entries)


if __name__ == "__main__":
    books = _load_books_from_manifest()
    print(f"MinHash dedup over {len(books)} documents "
          f"(threshold {JACCARD_THRESHOLD}, {N_PERM} perms, {BANDS}x{ROWS} bands)")
    drop, clusters = find_duplicates(books)
    by_id = {b["gutenberg_id"]: b for b in books}
    wasted = sum(by_id[i]["char_count"] for i in drop)
    print(f"\nClusters ({len(clusters)}):")
    for rep, losers in sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:30]:
        print(f"  keep {rep:>6} {by_id[rep]['title'][:44]:44s}")
        for l in losers:
            print(f"    drop {l:>6} {by_id[l]['title'][:44]}")
    print(f"\nWould drop {len(drop)} documents, {wasted/1e6:.1f} MB of text")
