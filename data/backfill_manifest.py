"""
Backfill provenance fields into data/manifest.jsonl.

PLAN.md §6 requires, per document: source, license, publication date, cleaning
steps applied, and a hash. The first pass of the downloader recorded source and
paths only. This script fills in the rest.

What it can and cannot establish, stated plainly:

  - `copyright`         Gutenberg's own determination (false = they assert PD).
                        Authoritative enough for a Gutenberg-sourced backbone.
  - `author_*_year`     From Gutendex, useful as a coarse date sanity check.
  - `pub_year`          NOT AVAILABLE from Gutendex. Left null on purpose.
                        PLAN.md §6's "publication date <= 1930, checked per
                        work" rule therefore cannot be automated from this
                        source. That is fine for Gutenberg (everything there is
                        already PD) but it means the weird & eerie slice in
                        §5.2.3 needs per-work manual verification, because that
                        is exactly the slice where the 1930 line actually bites.
  - `cleaning_steps`    What was applied. Note the downloader overwrote the raw
                        download with the cleaned text, so the pre-clean bytes
                        were not retained; `local_path == cleaned_path` reflects
                        that rather than meaning "no cleaning happened".
  - `sha256`            Of the on-disk (cleaned) text.

Run after download_gutenberg.py, before tokenize_corpus.py.
Network access is only needed for the Gutendex fields; hashes and cleaning
metadata are filled in offline.
"""

import sys
import json
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config

GUTENDEX_URL = "https://gutendex.com/books/"
BATCH = 32          # gutendex accepts ?ids=comma,separated
CLEANING_STEPS = ["strip_gutenberg_boilerplate_header_footer", "utf8_normalize", "strip"]


def load_manifest() -> list[dict]:
    entries = []
    if config.manifest_path.exists():
        with open(config.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def dedupe(entries: list[dict]) -> list[dict]:
    """Collapse duplicate gutenberg_ids, last wins, preserving first-seen order."""
    seen, out = {}, []
    for e in entries:
        gid = e["gutenberg_id"]
        if gid in seen:
            out[seen[gid]] = e
        else:
            seen[gid] = len(out)
            out.append(e)
    if len(out) != len(entries):
        print(f"  Collapsed {len(entries) - len(out)} duplicate manifest entries")
    return out


def resolve_path(e: dict) -> Path | None:
    for key in ("cleaned_path", "local_path"):
        p = e.get(key)
        if p and Path(p).exists():
            return Path(p)
    p = config.gutenberg_raw / f"{e['gutenberg_id']}.txt"
    return p if p.exists() else None


def fetch_gutendex(ids: list[int], session) -> dict[int, dict]:
    """Fetch metadata for a batch of ids. Returns {} on failure (non-fatal)."""
    url = f"{GUTENDEX_URL}?ids={','.join(str(i) for i in ids)}"
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            return {b["id"]: b for b in resp.json().get("results", [])}
        except Exception as e:
            if attempt == 2:
                print(f"  WARNING: gutendex batch failed ({e}); "
                      f"leaving those fields unset")
                return {}
            time.sleep(2 ** attempt)
    return {}


def main(online: bool = True):
    entries = dedupe(load_manifest())
    if not entries:
        print("Manifest is empty. Run data/download_gutenberg.py first.")
        return
    print(f"Backfilling {len(entries)} manifest entries...")

    meta: dict[int, dict] = {}
    if online:
        try:
            import requests
        except ImportError:
            print("  requests not installed; skipping Gutendex fields")
            online = False
        if online:
            session = requests.Session()
            session.headers.update({"User-Agent": "k3-mini-corpus-builder/0.1"})
            ids = [e["gutenberg_id"] for e in entries]
            for i in range(0, len(ids), BATCH):
                batch = ids[i:i + BATCH]
                meta.update(fetch_gutendex(batch, session))
                print(f"  Gutendex {min(i + BATCH, len(ids))}/{len(ids)}")
                time.sleep(0.5)   # be polite

    hashed = missing = 0
    for e in entries:
        gid = e["gutenberg_id"]

        # --- offline fields ---
        path = resolve_path(e)
        if path is None:
            e["status"] = "missing_on_disk"
            missing += 1
        else:
            e["cleaned_path"] = str(path)
            e["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            e["char_count"] = path.stat().st_size
            hashed += 1

        e["cleaning_steps"] = CLEANING_STEPS
        e["raw_retained"] = False
        e["license"] = "public_domain_us_gutenberg"
        e.setdefault("token_count", None)   # filled by tokenize_corpus.py

        # publication date is not derivable from Gutendex — say so explicitly
        e.setdefault("pub_year", None)
        e["pub_year_source"] = "unavailable_from_gutendex"

        # --- online fields ---
        b = meta.get(gid)
        if b:
            e["copyright"] = b.get("copyright")
            e["subjects"] = b.get("subjects", [])[:12]
            e["bookshelves"] = b.get("bookshelves", [])[:8]
            authors = b.get("authors", [])
            e["authors"] = [a["name"] for a in authors]
            e["author_birth_year"] = [a.get("birth_year") for a in authors]
            e["author_death_year"] = [a.get("death_year") for a in authors]
            e["pd_basis"] = (
                "gutenberg_copyright_false" if b.get("copyright") is False
                else f"gutenberg_copyright={b.get('copyright')}"
            )

    tmp = config.manifest_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    tmp.replace(config.manifest_path)

    n_meta = sum(1 for e in entries if "copyright" in e)
    flagged = [e["gutenberg_id"] for e in entries if e.get("copyright") is True]
    print(f"\nWrote {len(entries)} entries to {config.manifest_path}")
    print(f"  hashed on disk:        {hashed}")
    print(f"  missing on disk:       {missing}")
    print(f"  gutendex metadata:     {n_meta}")
    print(f"  pub_year:              0 (unavailable from this source, by design)")
    if flagged:
        print(f"  !! copyright=true, REVIEW BEFORE TRAINING: {flagged}")
    else:
        print(f"  copyright=true:        0")


if __name__ == "__main__":
    main(online="--offline" not in sys.argv)
