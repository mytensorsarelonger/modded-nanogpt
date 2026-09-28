"""
evals/exemplar.py — the §7.2 exemplar / contrast-pair harness.

Three tracked quantities, wired into train_baseline.py's checkpoint loop and
runnable retroactively over checkpoints already on the runs Volume
(PLAN.md §7.2, RL_STRATEGY.md §3):

1. Perplexity of hand-written exemplars — the owner's eval set.
2. Contrast-pair loss gap: target writing vs. superficially-similar-but-wrong
   writing. Should WIDEN if the mix works. This is the known-curve instrument
   for the actual goal; a mix ablation whose curves cannot be scored on this
   axis cannot be scored at all.
3. Craft-essay / deliberation perplexity, tracked separately (§4.2 job 2).

Design decisions that are not negotiable without a CHANGELOG entry:

- WINDOWED loss, EOT-anchored, not whole-text. Every document in the corpus is
  prefixed with EOT (50256), and training windows are seq_len-long slices of
  a concatenated stream. A whole-text loss under a causal model weights the
  first tokens (near-uninformative conditioning) far more than an in-training
  window does, so exemplar PPL and val PPL would not be comparable and the
  "is the register actually landing" question would be answered in the wrong
  units. Instead: each passage is scored over overlapping seq_len windows
  with a fresh EOT prefix, exactly the shape the model trains on.
- The CONTAMINATION GUARD is part of the harness, not a one-off check. The
  Philosophy of Composition is in the training corpus via collected-works
  editions (10031, 76996) while the standalone book (55749) was deduped out:
  manifest-level absence says "not downloaded", which is not "not trained".
  The guard tokenizes every eval passage with the same gpt2 encoder and
  searches the actual training shards for a 12-token exact match at every
  offset. A hit fails the run loudly rather than scoring memorization as if
  it were register affinity. Craft passages whose source is corpus-absent
  (Hazlitt, Lamb) pass; any future passage pasted from a corpus book fails.
- Per-passage numbers, then aggregates. A single aggregate exemplar PPL
  hides the mundane/cold-open split that §7.1 already established matters;
  the JSONL log keeps per-passage losses so a later re-aggregation costs
  nothing.
- Strict JSON, one line per evaluation, append-only — same contract as
  samples.log and the run registry, for the same reasons (index.jsonl loses
  rows when runs overlap; run.json is the source of truth).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Eval-set loading
# ---------------------------------------------------------------------------

_SECTION = re.compile(r"(?m)^##\s+(\S+)[^\n]*$")
_HEADER = re.compile(r"(?m)^#[^\n]*$")


@dataclass(frozen=True)
class Passage:
    id: str
    text: str


def _parse_sections(text: str) -> dict[str, str]:
    """`## id` sections, `---`-terminated. Prose only; comment/paragraph prose
    before the first `##` header is ignored (that is where file-level notes
    live). A `---` line ends its section — the separator itself is never
    scored as passage text."""
    sections: dict[str, str] = {}
    matches = list(_SECTION.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end]
        # a --- terminator ends the section early; text after it up to the
        # next header is ignored (notes between sections)
        lines = body.splitlines()
        kept: list[str] = []
        for line in lines:
            if line.strip() == "---":
                break
            kept.append(line)
        # drop any embedded file-level comment lines the way _HEADER would
        kept = [line for line in kept if not line.lstrip().startswith("#")]
        sections[m.group(1)] = "\n".join(kept).strip()
    return sections


def load_passages(path: Path) -> list[Passage]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — the §7.2 eval set ships with the repo "
            f"(evals/exemplar_data/)"
        )
    sections = _parse_sections(path.read_text(encoding="utf-8"))
    if not sections:
        raise ValueError(f"No '## <id>' sections in {path}")
    return [Passage(sid, body) for sid, body in sections.items()]


def split_pairs(passages: list[Passage]) -> tuple[list[Passage], list[tuple[Passage, Passage]]]:
    """Separate plain exemplars from contrast pairs (id.contrast convention)."""
    by_id = {p.id: p for p in passages}
    paired = {p.id[: -len(".contrast")] for p in passages if p.id.endswith(".contrast")}
    singles, pairs = [], []
    for p in passages:
        if p.id.endswith(".contrast"):
            base_id = p.id[: -len(".contrast")]
            if base_id not in by_id:
                raise ValueError(
                    f"contrast passage {p.id!r} has no target passage "
                    f"{base_id!r} in the same file"
                )
            pairs.append((by_id[base_id], p))
        elif p.id not in paired:
            singles.append(p)
    return singles, pairs


def eval_set_digest(paths: list[Path]) -> str:
    """Stable digest of the eval set — recorded so a sweep result names the
    exact texts it scored, and a later edit to the eval set is visible as a
    digest change rather than as a mysterious curve shift."""
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

EOT_TOKEN = 50256  # gpt2 eot; also config.eot_token — asserted at call sites


@dataclass
class ScoredPassage:
    id: str
    tokens: int            # scored token count (targets side)
    nll_sum: float         # sum of per-token NLL over all windows
    ppl: float


def _encode_windows(text: str, enc, seq_len: int, device) -> list[tuple[Tensor, Tensor]]:
    """EOT-anchored windows: [EOT] + text tokens, chunked at seq_len.

    The first window is full-width whenever the passage allows: the corpus
    packs documents with an EOT prefix into a continuous stream, and the
    model's training views are exactly seq_len-long slices. Overlapping by
    half a window bounds the conditioning deficit for passages longer than
    one window (every token except the first stride is scored with at least
    seq_len/2 of context).
    """
    ids = enc.encode(text, allowed_special=set())
    stream = [EOT_TOKEN] + ids
    windows: list[tuple[Tensor, Tensor]] = []
    stride = max(1, seq_len // 2)
    start = 0
    while start < len(stream) - 1:  # at least one target token
        chunk = stream[start : start + seq_len + 1]
        if len(chunk) < 2:
            break
        inputs = torch.tensor(chunk[:-1], dtype=torch.int64, device=device)
        targets = torch.tensor(chunk[1:], dtype=torch.int64, device=device)
        windows.append((inputs, targets))
        if start + seq_len + 1 >= len(stream):
            break
        start += stride
    return windows


def score_passage(model, enc, passage: Passage, seq_len: int, device) -> ScoredPassage:
    """Per-token NLL of `passage` under `model`, in training-window shape.

    Uses forward_logits (not forward): forward returns cross_entropy over the
    SOFTCAPPED logits the control model trains on — identical math, but
    keeping the loss here rather than in the model makes the harness usable
    with any forward_logits-bearing model (including future K3-mini files).
    """
    was_training = model.training
    model.eval()
    nll_sum, ntok = 0.0, 0
    with torch.no_grad():
        for inputs, targets in _encode_windows(passage.text, enc, seq_len, device):
            logits = model.forward_logits(inputs.unsqueeze(0))
            logits = 15 * logits * (logits.square() + 15 ** 2).rsqrt()
            nll = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets, reduction="sum"
            )
            nll_sum += float(nll)
            ntok += targets.numel()
    if was_training:
        model.train()
    if ntok == 0:
        raise ValueError(f"passage {passage.id!r} scored zero tokens")
    return ScoredPassage(passage.id, ntok, nll_sum, float(torch.exp(torch.tensor(nll_sum / ntok))))


def evaluate(
    model,
    enc,
    *,
    exemplar_file: Path,
    craft_file: Path,
    seq_len: int,
    device,
    max_batch_tokens: int = 64 * 1024,
) -> dict:
    """Full §7.2 read-out: per-passage losses + the three tracked aggregates."""
    singles, pairs = split_pairs(load_passages(exemplar_file))
    craft = load_passages(craft_file)

    scored_singles = [score_passage(model, enc, p, seq_len, device) for p in singles]
    pair_scores = [
        (score_passage(model, enc, tgt, seq_len, device),
         score_passage(model, enc, wrong, seq_len, device))
        for tgt, wrong in pairs
    ]
    craft_scores = [score_passage(model, enc, p, seq_len, device) for p in craft]

    def aggregate(scores: list[ScoredPassage]) -> float:
        nll = sum(s.nll_sum for s in scores)
        ntok = sum(s.tokens for s in scores)
        return float(nll / ntok) if ntok else float("nan")

    gaps = [
        {
            "pair": tgt.id,
            "target_ppl": tgt.ppl,
            "wrong_ppl": wrong.ppl,
            # loss gap in nats per token: wrong - target. Positive means the
            # model finds the wrong member LESS likely, i.e. the register
            # discrimination is present. Widening across checkpoints is the
            # §7.2 known curve.
            "loss_gap": (wrong.nll_sum / wrong.tokens) - (tgt.nll_sum / tgt.tokens),
            "target_nll_per_tok": tgt.nll_sum / tgt.tokens,
            "wrong_nll_per_tok": wrong.nll_sum / wrong.tokens,
            "target_tokens": tgt.tokens,
            "wrong_tokens": wrong.tokens,
        }
        for tgt, wrong in pair_scores
    ]

    return {
        # §7.2 quantity 1: perplexity of the hand-written exemplars — the
        # register targets ONLY. (First sweep shipped a target+contrast
        # average here; that diluted the register signal with the mundane
        # side and is not what §7.2 asks for. Contrast members are still
        # scored and reported per-pair in contrast_gaps.)
        "exemplar_ppl_nats_per_tok": aggregate(
            [tgt for tgt, wrong in pair_scores] + scored_singles
        ),
        "exemplar_passages": [vars(s) for s in scored_singles],
        "contrast_gaps": gaps,
        "mean_loss_gap": (sum(g["loss_gap"] for g in gaps) / len(gaps)) if gaps else None,
        "craft_ppl_nats_per_tok": aggregate(craft_scores),
        "craft_passages": [vars(s) for s in craft_scores],
        "eval_set_digest": eval_set_digest([exemplar_file, craft_file]),
        "seq_len": seq_len,
    }


# ---------------------------------------------------------------------------
# Contamination guard
# ---------------------------------------------------------------------------

_SHARD_MAGIC = 20240520


def _load_shard_tokens(shard: Path) -> Tensor:
    header = torch.from_file(str(shard), False, 256, dtype=torch.int32)
    if int(header[0]) != _SHARD_MAGIC:
        raise ValueError(f"{shard}: bad magic {int(header[0])}")
    n = int(header[2])
    raw = torch.from_file(str(shard), False, 256 * 4 + 2 * n, dtype=torch.uint8)
    return raw[1024:].view(torch.uint16)[:n].long()


def _find_matches(query: Tensor, corpus: Tensor, match_len: int,
                  corpus_index: list[tuple[Tensor, Tensor]] | None = None):
    """Exact-match offsets of every match_len-run of `query` in `corpus`.

    Single pass over the corpus regardless of query length: both sides are
    viewed as packed lanes of 4 tokens (int64), the corpus lanes are sorted,
    and every query window is looked up by binary search. A naive scan of
    every query offset against every corpus offset is O(len(q)·len(c)) —
    minutes per passage over 462M tokens; this is O(len(c) log len(c)).

    `corpus_index` (optional, from _build_corpus_index) reuses the sorted
    corpus lanes across many queries — the per-shard sort is the dominant
    cost and must not be repeated per passage.
    """
    LANE = 4
    if match_len < LANE:
        raise ValueError(f"match_len must be >= {LANE}")
    if len(corpus) < match_len or len(query) < match_len:
        return []

    if corpus_index is None:
        corpus_index = _build_corpus_index(corpus)

    hits: list[tuple[int, int]] = []
    for c_sorted, c_offs in corpus_index:
        for q_start in range(0, len(query) - match_len + 1):
            q_lane = _pack_lane(query, q_start)
            pos = torch.searchsorted(c_sorted, q_lane)
            # exact-hit check + full match_len verification
            while pos < len(c_sorted) and int(c_sorted[pos]) == int(q_lane):
                cand = int(c_offs[pos])
                if cand + match_len <= len(corpus) and torch.equal(
                    corpus[cand : cand + match_len], query[q_start : q_start + match_len]
                ):
                    hits.append((q_start, cand))
                pos += 1
    return hits


def _pack_lane(t: Tensor, offset: int) -> Tensor:
    """One packed 4-token lane starting at `offset`, as a 1-element tensor."""
    a = t[offset : offset + 4].to(torch.int64)
    return (a * (2 ** torch.arange(4, dtype=torch.int64))).sum()


def _build_corpus_index(corpus: Tensor) -> list[tuple[Tensor, Tensor]]:
    """Sorted packed-lane index of the corpus, one per alignment shift.

    To catch matches at ANY alignment (not just multiples of 4) the lane
    trick runs once per shift r in 0..3 — i.e. the corpus viewed with every
    offset-0/1/2/3 phase. 4x the corpus work, still linear, still exact,
    and done ONCE per shard for all passages.
    """
    LANE = 4
    index = []
    for r in range(LANE):
        if len(corpus) - r < LANE:
            break
        n_lanes = (len(corpus) - r - LANE) // LANE + 1
        a = corpus[r : r + n_lanes * LANE].to(torch.int64)
        packed = a.view(n_lanes, LANE)
        lanes = (packed * (2 ** torch.arange(LANE, dtype=torch.int64))).sum(dim=1)
        offs = torch.arange(n_lanes, dtype=torch.int64) * LANE + r
        order = torch.argsort(lanes)
        index.append((lanes[order], offs[order]))
    return index


def check_contamination(
    passages: list[Passage],
    enc,
    shard_glob: str,
    *,
    match_len: int = 12,
    root: Path | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Fail-loud corpus-overlap check for every eval passage.

    Exact token match of `match_len` consecutive tokens anywhere in the
    training shards, at ANY offset on either side. 12 gpt2 tokens is ~8-10
    words: an exact 12-token collision between held-out prose and a 462M-token
    corpus is contamination, not coincidence — especially since passages
    share register vocabulary with the corpus. Shorter would flag stock
    phrases; longer would let a mid-length verbatim borrow through. Calibrated
    against the corpus: no shipped passage has ANY 12-token match (verified
    2026-09-27 by running this guard; the Poe Philosophy-of-Composition probe
    DID hit, in train_000/train_004, which is why the guard exists).

    `root` overrides the glob base (tests); default is the process CWD, which
    is the repo root in training (the shards volume is mounted at
    data/shards) — the same cwd contract the dataloader's glob uses.
    """
    base = Path(root) if root is not None else Path.cwd()
    shard_files = sorted(base.glob(shard_glob))
    if not shard_files:
        raise FileNotFoundError(f"no shards match {shard_glob!r} (cwd={Path.cwd()})")
    hits: list[dict] = []
    for shard in shard_files:
        toks = _load_shard_tokens(shard)
        # The per-shard sort is the dominant cost: build the lane index ONCE
        # per shard and share it across every passage.
        corpus_index = _build_corpus_index(toks)
        for passage in passages:
            ids = torch.tensor(
                enc.encode(passage.text, allowed_special=set()), dtype=torch.long
            )
            for q_start, c_off in _find_matches(ids, toks, match_len,
                                                corpus_index=corpus_index):
                snippet = enc.decode(ids[q_start : q_start + match_len].tolist())
                hits.append({
                    "passage": passage.id,
                    "shard": shard.name,
                    "passage_offset": q_start,
                    "corpus_offset": c_off,
                    "snippet": snippet,
                })
                if verbose:
                    print(f"[guard] HIT {passage.id} @ {shard.name}:{c_off}: {snippet!r}")
    return hits


def assert_clean(passages: list[Passage], enc, shard_glob: str, **kw) -> None:
    hits = check_contamination(passages, enc, shard_glob, **kw)
    if hits:
        sample = hits[0]
        raise RuntimeError(
            f"contamination guard: {len(hits)} training-corpus match(es) for "
            f"eval passages. First: {sample['passage']!r} in "
            f"{sample['shard']} @ corpus offset {sample['corpus_offset']}: "
            f"{sample['snippet']!r}. "
            "Eval passages must be held-out; scoring in-train text measures "
            "memorization, not register affinity (PLAN.md §5.5)."
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _json_safe(value):
    """Non-finite floats -> None + flag, recursively. A diverged model yields
    inf/nan perplexities; strict JSON cannot carry them, and crashing the run
    at the logging line to preserve strictness would lose the measurement of
    the divergence itself — the one thing that diagnosis needs. Same
    convention as completion_metadata's final_val_loss_nonfinite."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(v, float) and not math.isfinite(v):
                out[k] = None
                out[f"{k}_nonfinite"] = True
            else:
                out[k] = _json_safe(v)
        return out
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")
