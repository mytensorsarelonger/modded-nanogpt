"""
Download Project Gutenberg books via the Gutendex API.

Fetches book metadata, filters for English-language public domain works,
and downloads the plain text. Strips license boilerplate headers.
"""

import os
import re
import sys
import time
import json
import requests
from pathlib import Path

# Windows defaults stdout to cp1252, so printing a title containing a macron,
# CJK glyph or anything else outside Latin-1 raises UnicodeEncodeError and kills
# the run. This corpus is multilingual by nature; force UTF-8 everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

GUTENDEX_URL = "https://gutendex.com/books/"
RAW_DIR = Path("data/gutenberg/raw")
MANIFEST_PATH = Path("data/manifest.jsonl")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# License boilerplate patterns to strip (Gutenberg header/footer)
# NOTE the \s* after the asterisks. 115 of the first 3000 files used the form
# "***START OF THE PROJECT GUTENBERG EBOOK DRACULA***" with NO space, which the
# original `\*\*\* START OF` (literal space) could not match. Because the old
# function only stripped when BOTH markers matched, those files silently kept
# their entire license text. Whitespace inside the marker is also variable.
START_PATTERN = re.compile(
    r"\*\*\*\s*START OF (THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
END_PATTERN = re.compile(
    r"\*\*\*\s*END OF (THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)

def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg license header and footer.

    Handles each marker independently: a missing footer should still let us drop
    the header. The old all-or-nothing version returned the text untouched
    whenever either marker failed to match, which is how the license text ended
    up in the corpus.
    """
    start_match = START_PATTERN.search(text)
    end_match = END_PATTERN.search(text)

    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    if end <= start:          # markers out of order or degenerate; keep as-is
        start, end = 0, len(text)
    return text[start:end].strip()

def get_plain_text_url(book: dict) -> str | None:
    """Find the best plain text download URL for a book."""
    formats = book.get("formats", {})
    # Prefer UTF-8 plain text
    for key in ["text/plain; charset=utf-8", "text/plain"]:
        if key in formats:
            return formats[key]
    # Fallback: any text/plain
    for key, url in formats.items():
        if "text/plain" in key:
            return url
    return None

def download_book(book: dict, session: requests.Session) -> dict | None:
    """Download a single book and return manifest entry."""
    book_id = book["id"]
    title = book["title"]
    if not title:
        return None

    text_url = get_plain_text_url(book)
    if not text_url:
        return None

    # Check if already downloaded
    raw_path = RAW_DIR / f"{book_id}.txt"
    if raw_path.exists():
        return {
            "gutenberg_id": book_id,
            "title": title,
            "authors": [a["name"] for a in book.get("authors", [])],
            "language": book.get("languages", []),
            "source_url": text_url,
            "local_path": str(raw_path),
            "cleaned_path": str(raw_path),
            "status": "already_downloaded",
        }

    raw_text = None
    last_err = None
    for attempt in range(3):
        try:
            resp = session.get(text_url, timeout=60)
            resp.raise_for_status()
            raw_text = resp.text
            break
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    if raw_text is None:
        return {
            "gutenberg_id": book_id,
            "title": title,
            "error": str(last_err),
            "status": "failed",
        }

    try:

        # Strip boilerplate
        cleaned_text = strip_gutenberg_boilerplate(raw_text)
        if len(cleaned_text) < 100:
            # Didn't strip properly, use raw
            cleaned_text = raw_text

        # Save cleaned text
        raw_path.write_text(cleaned_text, encoding="utf-8")

        return {
            "gutenberg_id": book_id,
            "title": title,
            "authors": [a["name"] for a in book.get("authors", [])],
            "language": book.get("languages", []),
            "source_url": text_url,
            "local_path": str(raw_path),
            "cleaned_path": str(raw_path),
            "status": "downloaded",
            "token_count": None,  # filled later
        }
    except Exception as e:
        return {
            "gutenberg_id": book_id,
            "title": title,
            "error": str(e),
            "status": "failed",
        }

def main(max_books: int = 500):
    session = requests.Session()
    session.headers.update({"User-Agent": "k3-mini-corpus-builder/0.1"})

    # Resume from what is actually on disk, not from a status string. The status
    # field has several values ("downloaded", "already_downloaded") and can be
    # rewritten by backfill_manifest.py; the files are the ground truth.
    downloaded_ids = {int(p.stem) for p in RAW_DIR.glob("*.txt") if p.stem.isdigit()}
    print(f"Resuming: {len(downloaded_ids)} books already on disk")

    # Open manifest in append mode
    manifest_file = open(MANIFEST_PATH, "a", encoding="utf-8")
    downloaded = len(downloaded_ids)
    page = 1

    while downloaded < max_books:
        url = f"{GUTENDEX_URL}?page={page}&languages=en"
        print(f"Fetching page {page}...")
        # Retry with backoff. The previous version's `continue` targeted the
        # attempt loop rather than the page loop, so after three failures it fell
        # through and called resp.json() on the *previous* page's response --
        # silently re-processing stale data (or NameError on page 1).
        data = None
        for attempt in range(5):
            try:
                resp = session.get(url, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                if attempt < 4:
                    print(f"  Request failed ({type(e).__name__}), retrying in {wait}s"
                          f" ({attempt+1}/5)")
                    time.sleep(wait)
                else:
                    print(f"  Giving up on page {page} after 5 attempts: {e}")
        if data is None:
            page += 1          # skip this page, keep going
            continue           # <- the while loop, which is what we mean
        if not data.get("results"):
            print(f"  Page {page} returned no results; stopping")
            break

        for book in data["results"]:
            # Filter: English, has text, has title
            if "en" not in book.get("languages", []):
                continue
            if not get_plain_text_url(book):
                continue
            if not book.get("title"):
                continue

            # Skip already downloaded
            if book["id"] in downloaded_ids:
                continue

            entry = download_book(book, session)
            if entry and entry.get("status") == "downloaded":
                manifest_file.write(json.dumps(entry) + "\n")
                manifest_file.flush()
                downloaded += 1
                downloaded_ids.add(book["id"])
                print(f"  [{downloaded}/{max_books}] {book['title'][:60]}")

            if downloaded >= max_books:
                break

        if not data.get("next"):
            break
        page += 1
        time.sleep(1)  # be polite

    manifest_file.close()
    print(f"\nDone! Downloaded {downloaded} books. Manifest at {MANIFEST_PATH}")

# --- Targeted sourcing (PLAN.md §5.2.3) --------------------------------------
# The bulk crawl pages Gutendex in download-count order, which is why it returned
# six Poe and zero Dunsany. A register slice has to be fetched by name instead:
# search-driven, not crawl-driven. Edit freely; this is a starting list, and taste
# is the point of the exercise.
CANON_QUERIES = [
    "Poe Edgar Allan",
    "Bierce Ambrose",
    "Machen Arthur",
    "Blackwood Algernon",
    "Dunsany",
    "Montague Rhodes James",
    "Hodgson William Hope",
    "Chambers Robert W",
    "Le Fanu Sheridan",
    "Lovecraft",
    "Shiel M P",
    "Marsh Richard",
    "Benson E F",
    "Onions Oliver",
    "Hearn Lafcadio",
    "Hoffmann E T A",
    "Gautier Theophile",
    "Radcliffe Ann",
    "Maturin Charles Robert",
    "Beckford William",
]


def iter_search_results(session, params: dict, max_pages: int = 40):
    """Page through a Gutendex query, yielding book dicts."""
    page = 1
    while page <= max_pages:
        q = dict(params, page=page)
        url = GUTENDEX_URL + "?" + "&".join(f"{k}={v}" for k, v in q.items())
        data = None
        for attempt in range(5):
            try:
                resp = session.get(url, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  Giving up on {url}: {e}")
        if not data or not data.get("results"):
            return
        for book in data["results"]:
            yield book
        if not data.get("next"):
            return
        page += 1
        time.sleep(0.5)


def fetch_targeted(queries: list[str] = None, ids: list[int] = None,
                   max_per_query: int = 200, dry_run: bool = False,
                   include_copyright: bool = False) -> int:
    """
    Fetch specific authors/works by name or id rather than by popularity.

    In-copyright works are skipped by default: PLAN.md §6 is public-domain only,
    and this session found 27 of them in the bulk crawl -- mostly copyrighted
    *translations and editions* of long-PD works, which is why the check has to
    happen per edition rather than per author.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "k3-mini-corpus-builder/0.1"})
    on_disk = {int(p.stem) for p in RAW_DIR.glob("*.txt") if p.stem.isdigit()}

    seen, new, skipped_cr, skipped_have = set(), 0, [], 0
    manifest_file = None if dry_run else open(MANIFEST_PATH, "a", encoding="utf-8")

    tasks = []
    if ids:
        tasks.append(("ids", {"ids": ",".join(str(i) for i in ids)}))
    for q in (queries or []):
        tasks.append((q, {"search": q.replace(" ", "%20"), "languages": "en"}))

    try:
        for label, params in tasks:
            found = kept = 0
            for book in iter_search_results(session, params):
                if found >= max_per_query:
                    break
                found += 1
                bid = book["id"]
                if bid in seen:
                    continue
                seen.add(bid)

                if not get_plain_text_url(book) or not book.get("title"):
                    continue
                if book.get("copyright") is True and not include_copyright:
                    skipped_cr.append((bid, book["title"][:44]))
                    continue
                if bid in on_disk:
                    skipped_have += 1
                    continue

                kept += 1
                authors = ", ".join(a["name"] for a in book.get("authors", []))
                if dry_run:
                    print(f"  WOULD FETCH {bid:>6}  {authors[:28]:28s}  "
                          f"{book['title'][:44]}")
                    continue
                entry = download_book(book, session)
                if entry and entry.get("status") == "downloaded":
                    manifest_file.write(json.dumps(entry) + "\n")
                    manifest_file.flush()
                    on_disk.add(bid)
                    new += 1
                    print(f"  [{new}] {bid:>6}  {book['title'][:52]}")
            print(f"{label!r}: {found} results, {kept} new")
    finally:
        if manifest_file:
            manifest_file.close()

    print(f"\n{'Would fetch' if dry_run else 'Fetched'} {new if not dry_run else len(seen)} "
          f"new book(s); {skipped_have} already on disk")
    if skipped_cr:
        print(f"Skipped {len(skipped_cr)} in-copyright edition(s) per §6 "
              f"(pass --include-copyright to override):")
        for bid, title in skipped_cr[:15]:
            print(f"    {bid:>6}  {title}")
    if not dry_run and new:
        print("\nNow re-run: data/backfill_manifest.py then data/tokenize_corpus.py")
    return new


def reclean(dry_run: bool = False) -> int:
    """
    Re-apply boilerplate stripping to every file already on disk.

    Needed whenever strip_gutenberg_boilerplate changes. Safe and idempotent: on
    an already-clean file the markers are gone, both searches miss, and the text
    is returned unchanged. Without this, manifest.jsonl's cleaning_steps claim is
    false for any file the previous stripper missed -- a provenance defect, not
    just wasted tokens (PLAN.md §6).
    """
    files = sorted(RAW_DIR.glob("*.txt"))
    changed = 0
    for f in files:
        before = f.read_text(encoding="utf-8", errors="replace")
        after = strip_gutenberg_boilerplate(before)
        if after != before:
            changed += 1
            saved = len(before) - len(after)
            print(f"  {f.stem}: {len(before):,} -> {len(after):,} ch (-{saved:,})")
            if not dry_run:
                f.write_text(after, encoding="utf-8")
    verb = "would change" if dry_run else "re-cleaned"
    print(f"{verb} {changed}/{len(files)} files")
    if changed and not dry_run:
        print("Now re-run: data/backfill_manifest.py then data/tokenize_corpus.py")
    return changed


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Fetch Gutenberg texts: bulk by popularity, or targeted by name.",
        epilog="examples:\n"
               "  %(prog)s 3000                        bulk crawl to 3000 books\n"
               "  %(prog)s --canon --dry-run           preview the §5.2.3 register slice\n"
               "  %(prog)s --canon                     fetch it\n"
               "  %(prog)s --author Dunsany --author 'Blackwood Algernon'\n"
               "  %(prog)s --ids 10662,42848\n"
               "  %(prog)s --reclean --dry-run         re-strip boilerplate\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("max_books", nargs="?", type=int, default=None,
                    help="bulk crawl target (popularity order)")
    ap.add_argument("--author", action="append", default=[], metavar="QUERY",
                    help="search term, repeatable (matches author names and titles)")
    ap.add_argument("--ids", default=None,
                    help="comma-separated Gutenberg ids")
    ap.add_argument("--canon", action="store_true",
                    help="fetch the built-in weird & eerie canon list")
    ap.add_argument("--max-per-query", type=int, default=200)
    ap.add_argument("--include-copyright", action="store_true",
                    help="do not skip in-copyright editions (§6 says don't)")
    ap.add_argument("--reclean", action="store_true",
                    help="re-apply boilerplate stripping to files on disk")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.reclean:
        reclean(dry_run=args.dry_run)
    elif args.canon or args.author or args.ids:
        queries = list(args.author)
        if args.canon:
            queries = CANON_QUERIES + queries
        ids = [int(x) for x in args.ids.split(",")] if args.ids else None
        fetch_targeted(queries=queries, ids=ids,
                       max_per_query=args.max_per_query,
                       dry_run=args.dry_run,
                       include_copyright=args.include_copyright)
    else:
        main(args.max_books if args.max_books is not None else 500)
