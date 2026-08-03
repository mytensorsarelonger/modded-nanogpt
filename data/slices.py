"""
Corpus slice classification (PLAN.md §5.1, §5.2.3).

§5.1 wants the register slice held at 10-25% of tokens and §4.3 wants it
upweighted during cooldown. Neither is possible unless something records which
documents ARE the register slice -- and after tokenisation the corpus is one
undifferentiated pile of shards. This module is that record.

Classification is by AUTHOR, deliberately, because that is the granularity
§5.2.3 works at ("author by author"). It is a heuristic with a known weakness:
a prolific author's whole catalogue comes along, and E. F. Benson wrote roughly
a hundred books of which the Mapp and Lucia social comedies are not register
material at all. So the author rule is overridable per document via
data/slice_overrides.json:

    {"12345": "backbone", "678": "register"}

Author-level classification plus hand overrides beats recording "which command
fetched this", because it is re-derivable from the manifest alone and survives
a corpus rebuild.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config

BACKBONE = "backbone"
REGISTER = "register"

# Substring match against Gutenberg's "Surname, Forenames" author strings.
# Note "James, M. R." matches "James, M. R. (Montague Rhodes)" -- Gutenberg's
# actual form, which a naive "James, Montague" pattern misses entirely.
REGISTER_AUTHOR_PATTERNS = [
    "Poe, Edgar Allan",
    "Bierce, Ambrose",
    "Machen, Arthur",
    "Blackwood, Algernon",
    "Dunsany",
    "James, M. R.",
    "Hodgson, William Hope",
    "Chambers, Robert W",
    "Le Fanu",
    "Lovecraft",
    "Shiel, M. P",
    "Marsh, Richard",
    "Benson, E. F",
    "Onions, Oliver",
    "Hearn, Lafcadio",
    "Hoffmann, E. T. A",
    "Gautier, Th",
    "Radcliffe, Ann",
    "Maturin, Charles Robert",
    "Beckford, William",
    "Shelley, Mary Wollstonecraft",
    "Stoker, Bram",
    "Walpole, Horace",
    "Riddell, Mrs. J. H",
    "Crawford, F. Marion",
    "Wells, H. G",
]

OVERRIDES_PATH = config.data_dir / "slice_overrides.json"


def load_overrides() -> dict[int, str]:
    if not OVERRIDES_PATH.exists():
        return {}
    raw = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items() if v in (BACKBONE, REGISTER)}


def classify(entry: dict, overrides: dict[int, str] | None = None) -> str:
    """Return BACKBONE or REGISTER for one manifest entry."""
    overrides = overrides if overrides is not None else {}
    gid = entry.get("gutenberg_id")
    if gid in overrides:
        return overrides[gid]
    authors = entry.get("authors") or []
    for a in authors:
        for pat in REGISTER_AUTHOR_PATTERNS:
            if pat in a:
                return REGISTER
    return BACKBONE


def summarize(entries: list[dict], split: str = "train") -> dict:
    """
    Token totals per slice.

    Defaults to the TRAIN split only, matching the `slices` block in
    shards/manifest.json. Counting val tokens here too would report a different
    share than the shard manifest for the same corpus -- two denominators for one
    number, in the artifact whose whole job is being unambiguous.
    """
    overrides = load_overrides()
    out = {BACKBONE: {"books": 0, "tokens": 0}, REGISTER: {"books": 0, "tokens": 0}}
    for e in entries:
        if split is None:
            if not e.get("split"):
                continue
        elif e.get("split") != split:
            continue
        s = e.get("slice") or classify(e, overrides)
        out[s]["books"] += 1
        out[s]["tokens"] += e.get("token_count") or 0
    total = sum(v["tokens"] for v in out.values()) or 1
    for v in out.values():
        v["share"] = v["tokens"] / total
    return out


if __name__ == "__main__":
    entries, seen = [], set()
    with open(config.manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                if e["gutenberg_id"] not in seen:
                    seen.add(e["gutenberg_id"])
                    entries.append(e)
    ov = load_overrides()
    print(f"{len(entries)} manifest entries, {len(ov)} override(s) "
          f"from {OVERRIDES_PATH.name if OVERRIDES_PATH.exists() else '(none)'}")
    s = summarize(entries)
    for name, v in s.items():
        print(f"  {name:9s} {v['books']:5d} books  {v['tokens']/1e6:8.2f}M tokens  "
              f"{v['share']:6.1%}")
    print(f"\n§5.1 target for register: 10-25%")
