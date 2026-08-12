"""
Tier-1 heuristic quality filter (PLAN.md §5.5.2).

Cheap, transparent, document-level. This is the first of two tiers: it exists to
drop material that is obviously not prose — reference tables, almanacs, indices,
gazetteers, catalogues — before any model-based classifier is trained. It makes
no attempt at literary judgement; that is tier 2's job, and taste is §7's.

Design notes:

  - Every threshold lives in THRESHOLDS and every rejection reports which one
    fired, so the filter is auditable and tunable by looking at what it caught
    rather than by argument.
  - Metrics are SENTENCE-based, not line- or paragraph-based. Gutenberg texts
    are hard-wrapped at ~70 columns and their blank-line conventions vary wildly
    (Moby Dick is 57% blank lines), so both line and paragraph statistics are
    measuring typesetting rather than content. Sentences are immune to both.
  - Verse and drama are KEPT. Short lines, repeated stage directions and low
    sentence counts are properties of poetry and plays, not of junk; the target
    is reference material — almanacs, gazetteers, biographical dictionaries,
    roll calls, catalogues, audiobook indices. digit_ratio and dup_line_frac
    are what actually separate those from literature.
  - Run with --report to score the corpus without changing anything. Do that
    first, and read the tail, before trusting any cutoff.

Usage:
    python data/quality_filter.py --report            # score all books, show outliers
    python data/quality_filter.py --report --top 40   # show more
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config

import re

SENTENCE_END = re.compile(r'[.!?]+["\'”’)\]]*(?=\s|$)')

# Residual Gutenberg license text. This is a SAFETY NET for cleaning failures:
# strip_gutenberg_boilerplate once silently no-opped on 115 files whose marker was
# "***START OF THE..." with no space, and the license text went straight into the
# corpus. Catching it here means the next stripping failure is loud, not silent.
RESIDUAL_BOILERPLATE = re.compile(
    r'gutenberg\.org/license|Section 1\.\s+General Terms of Use'
    r'|This eBook is for the use of anyone anywhere', re.I)

# LibriVox audiobook metadata: a track listing with timings and reader credits,
# no book text at all. Matched on the front-matter phrase rather than the bare
# word "librivox", which also appears in proofreaders' email addresses inside
# perfectly good books (e.g. 27481, a 495 KB travelogue).
AUDIOBOOK_METADATA = re.compile(r'This audio reading of', re.I)

# Tuned against the 200-book Gutenberg sample; run --report before changing any
# of these. Rejection criteria only — short_line_frac and the paragraph stats are
# reported for inspection but deliberately NOT used to reject, because they
# reject verse and drama.
THRESHOLDS = {
    "min_chars": 5_000,           # a short story is canon (§5.2.3), not a stub
    # Real reference works in this sample sit at 0.062-0.166. Legitimate
    # literature that merely contains dates, page or line numbers (Chaucer's
    # Troilus, the Anabasis, Napoleon's letters, a Nightingale biography) sits at
    # 0.031-0.039. 0.030 rejected all four; 0.050 separates them cleanly.
    "max_digit_ratio": 0.050,
    "min_alpha_ratio": 0.800,     # symbol soup, OCR wreckage, markup
    "max_dup_line_frac": 0.320,   # repeated field labels; loose enough for plays
    "max_upper_ratio": 0.140,     # all-caps catalogues and headers
    # A floor for degenerate one-word records only. Anything higher rejects
    # verse and drama (Chaucer 5.8w, Romeo and Juliet 8.4w), which we want.
    # Reference material is caught by digit_ratio/alpha_ratio instead.
    "min_sentence_words": 5.0,
    "max_sentence_words": 90.0,   # no sentence structure detected at all
}


def score_document(text: str) -> dict:
    """Compute cheap structural metrics for one document."""
    n_chars = len(text)
    if n_chars == 0:
        return {"n_chars": 0}

    non_space = [c for c in text if not c.isspace()]
    n_ns = len(non_space) or 1
    n_alpha = sum(c.isalpha() for c in non_space)
    n_digit = sum(c.isdigit() for c in non_space)
    n_upper = sum(c.isupper() for c in non_space if c.isalpha())

    lines = text.split("\n")
    nonblank = [ln.strip() for ln in lines if ln.strip()]
    n_nb = len(nonblank) or 1
    n_short = sum(len(ln) < 40 for ln in nonblank)
    n_uniq = len(set(nonblank))

    # Sentences, not paragraphs: independent of wrapping and blank-line style.
    words = text.split()
    n_sentences = len(SENTENCE_END.findall(text))

    head = text[:4000]
    return {
        "residual_boilerplate": bool(RESIDUAL_BOILERPLATE.search(text)),
        "audiobook_metadata": bool(AUDIOBOOK_METADATA.search(head)),
        "n_chars": n_chars,
        "alpha_ratio": n_alpha / n_ns,
        "digit_ratio": n_digit / n_ns,
        "upper_ratio": n_upper / (n_alpha or 1),
        "short_line_frac": n_short / n_nb,          # reported, not a reject criterion
        "dup_line_frac": 1.0 - (n_uniq / n_nb),
        "n_words": len(words),
        "n_sentences": n_sentences,
        "sentence_words": (len(words) / n_sentences) if n_sentences else float("inf"),
        "mean_word_len": (sum(len(w) for w in words) / len(words)) if words else 0.0,
    }


def is_prose(m: dict, thresholds: dict = THRESHOLDS) -> tuple[bool, list[str]]:
    """Return (keep, reasons_for_rejection). Empty reasons means kept."""
    t, reasons = thresholds, []
    # Content rejections come first: these are "this document is the wrong kind
    # of thing", not "this document scores badly".
    if m.get("residual_boilerplate"):
        reasons.append("residual_gutenberg_license")
    if m.get("audiobook_metadata"):
        reasons.append("audiobook_metadata")
    if m.get("n_chars", 0) < t["min_chars"]:
        reasons.append(f"too_short({m.get('n_chars', 0)})")
    if m.get("digit_ratio", 0) > t["max_digit_ratio"]:
        reasons.append(f"digit_ratio({m['digit_ratio']:.3f})")
    if m.get("alpha_ratio", 1) < t["min_alpha_ratio"]:
        reasons.append(f"alpha_ratio({m['alpha_ratio']:.3f})")
    if m.get("dup_line_frac", 0) > t["max_dup_line_frac"]:
        reasons.append(f"dup_lines({m['dup_line_frac']:.3f})")
    if m.get("upper_ratio", 0) > t["max_upper_ratio"]:
        reasons.append(f"upper_ratio({m['upper_ratio']:.3f})")
    sw = m.get("sentence_words", 0.0)
    if sw < t["min_sentence_words"]:
        reasons.append(f"short_sentences({sw:.1f}w)")
    if sw > t["max_sentence_words"]:
        reasons.append(f"no_sentences({sw:.1f}w)")
    return (len(reasons) == 0), reasons


def _load_books() -> list[dict]:
    books = []
    if not config.manifest_path.exists():
        return books
    seen = set()
    with open(config.manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            gid = e["gutenberg_id"]
            if gid in seen or e.get("status") == "failed":
                continue
            seen.add(gid)
            p = Path(e.get("cleaned_path") or e.get("local_path") or "")
            if not p.exists():
                p = config.gutenberg_raw / f"{gid}.txt"
            if p.exists():
                books.append({"gutenberg_id": gid, "title": e.get("title") or "?",
                              "path": p, "copyright": e.get("copyright")})
    return books


def report(top: int = 25):
    books = _load_books()
    if not books:
        print("No books found. Run data/download_gutenberg.py first.")
        return

    print(f"Scoring {len(books)} books...\n")
    rows = []
    for b in books:
        text = b["path"].read_text(encoding="utf-8", errors="replace")
        m = score_document(text)
        keep, reasons = is_prose(m)
        rows.append({**b, **m, "keep": keep, "reasons": reasons})

    kept = [r for r in rows if r["keep"]]
    dropped = [r for r in rows if not r["keep"]]

    def pct(key, ps):
        vals = sorted(v for v in (r.get(key, 0) for r in rows) if v != float("inf"))
        out = []
        for p in ps:
            i = min(int(p * (len(vals) - 1)), len(vals) - 1)
            out.append(f"p{int(p*100)}={vals[i]:.3f}")
        return "  ".join(out)

    HI = (0.5, 0.9, 0.99, 1.0)    # metrics we reject for being too HIGH
    LO = (0.0, 0.01, 0.1, 0.5)    # metrics we reject for being too LOW
    print("Distribution across corpus (tail shown is the one that triggers rejection):")
    for key, ps in (("digit_ratio", HI), ("dup_line_frac", HI), ("upper_ratio", HI),
                    ("alpha_ratio", LO), ("sentence_words", LO),
                    ("short_line_frac", HI)):
        tag = "reported only" if key == "short_line_frac" else ""
        print(f"  {key:18s} {pct(key, ps):52s} {tag}")

    print(f"\nKept:    {len(kept)}/{len(rows)}")
    print(f"Dropped: {len(dropped)}/{len(rows)}")
    if dropped:
        drop_chars = sum(r["n_chars"] for r in dropped)
        all_chars = sum(r["n_chars"] for r in rows) or 1
        print(f"         ({drop_chars/all_chars:.1%} of corpus characters)\n")
        print(f"Dropped documents (worst first by reason count):")
        for r in sorted(dropped, key=lambda r: -len(r["reasons"]))[:top]:
            print(f"  {r['gutenberg_id']:>6}  {r['title'][:48]:48s}  "
                  f"{', '.join(r['reasons'])}")

    # The borderline cases are where thresholds are actually decided.
    print(f"\nClosest kept documents to rejection (inspect these):")
    def margin(r):
        t = THRESHOLDS
        return min(
            (t["max_digit_ratio"] - r.get("digit_ratio", 0)) / t["max_digit_ratio"],
            (r.get("alpha_ratio", 1) - t["min_alpha_ratio"]) / t["min_alpha_ratio"],
            (t["max_dup_line_frac"] - r.get("dup_line_frac", 0)) / t["max_dup_line_frac"],
            (t["max_upper_ratio"] - r.get("upper_ratio", 0)) / t["max_upper_ratio"],
        )
    for r in sorted(kept, key=margin)[:10]:
        print(f"  {r['gutenberg_id']:>6}  {r['title'][:44]:44s}  "
              f"digit={r['digit_ratio']:.3f} dup={r['dup_line_frac']:.3f} "
              f"alpha={r['alpha_ratio']:.3f} upper={r['upper_ratio']:.3f} "
              f"sw={r['sentence_words']:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    if args.report:
        report(args.top)
    else:
        print(__doc__)
