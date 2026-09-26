"""Eerie cases — passages from the weird & eerie register as RL episodes.

A ``Case`` is one passage from the K3-mini register corpus plus a task
type (continuation, restoration, or infilling) and the ground-truth text
that the model must produce.  The reward is mechanically checkable:

* **continuation** — given the first N tokens, produce the next M.  Reward
  = n-gram precision (fraction of model n-grams that appear in the
  held-out continuation).
* **restoration** — a real passage is flattened / modernised; the model
  restores the original.  Reward = n-gram recall against the original.
* **infilling** — the middle of a passage is deleted; the model fills it.
  Reward = n-gram F1 between model output and the deleted middle.

All three are deterministic string/set operations — no judge, no model,
no I/O.  This is the ``foundermath.checks.all_hard_pass`` analogue: pure
functions over the case and the model's output.

The case shape is deliberately simpler than ``foundermath.case.Case``
because the domain is text-on-text, not math-on-math: there is no
``answer_kind`` to dispatch on, no numeric grader, no spec dict.  What
there *is* is a passage, a task, and a target — and a BLEU-style score
that is cheap, deterministic, and unhackable against a reward model.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence


class Task(str, Enum):
    continuation = "continuation"
    restoration = "restoration"
    infilling = "infilling"


@dataclass(frozen=True)
class Case:
    """One passage from the register, one task, one target.

    ``prompt`` is what the model sees.  ``target`` is the held-out text
    the reward is measured against.  For continuation, ``prompt`` is the
    prefix and ``target`` is what follows; for infilling, ``prompt``
    contains the surrounding text with a ``[FILL]`` marker and ``target``
    is what was removed; for restoration, ``prompt`` is the degraded
    version and ``target`` is the original.
    """

    id: str
    task: Task
    prompt: str
    target: str
    source_title: str
    source_author: str
    source_id: int  # gutenberg_id


def _ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser.

    Good enough for n-gram overlap on literary text — we are measuring
    surface-form similarity, not semantic equivalence.
    """
    import re

    return [t for t in re.split(r"\s+", text.strip()) if t]


def _word_set(text: str) -> set[str]:
    """Lowercased word set — for bag-of-words / diction overlap."""
    return {w.strip(".,;:!?'\"()[]—–").lower() for w in _tokenize(text) if w}


def ngram_precision(
    generated: str, target: str, n: int = 3
) -> float:
    """Fraction of model n-grams that appear in the target (precision)."""
    gen_tokens = _tokenize(generated)
    tgt_tokens = _tokenize(target)
    gen_ngrams = _ngrams(gen_tokens, n)
    tgt_ngrams = _ngrams(tgt_tokens, n)
    if not gen_ngrams:
        return 0.0
    return len(gen_ngrams & tgt_ngrams) / len(gen_ngrams)


def ngram_recall(
    generated: str, target: str, n: int = 3
) -> float:
    """Fraction of target n-grams recovered by the model (recall)."""
    gen_tokens = _tokenize(generated)
    tgt_tokens = _tokenize(target)
    gen_ngrams = _ngrams(gen_tokens, n)
    tgt_ngrams = _ngrams(tgt_tokens, n)
    if not tgt_ngrams:
        return 0.0
    return len(gen_ngrams & tgt_ngrams) / len(tgt_ngrams)


def ngram_f1(
    generated: str, target: str, n: int = 3
) -> float:
    """F1 of n-gram precision and recall."""
    p = ngram_precision(generated, target, n)
    r = ngram_recall(generated, target, n)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def word_overlap(generated: str, target: str) -> float:
    """Jaccard overlap of word sets — diction / vocabulary match.

    This is the softer, learnable signal: what fraction of the model's
    vocabulary appears in the target?  It rewards using the same diction
    (archaic, atmospheric) without requiring the exact word order that
    n-gram precision demands.  A model that writes in the register even
    if it doesn't reproduce the passage will score above zero here.
    """
    gen_words = _word_set(generated)
    tgt_words = _word_set(target)
    if not gen_words or not tgt_words:
        return 0.0
    return len(gen_words & tgt_words) / len(gen_words | tgt_words)


def score_text(case: Case, text: str) -> tuple[float, dict[str, Any]]:
    """(reward, detail) for one model reply — the single scoring path.

    Used by both the env's ``step`` and any offline scorer, so the smoke
    test and the RL loop see identical numbers.  The reward blends two
    signals:

    * **Sequence match** (n-gram precision/recall/F1) — the exact-text
      component.  Near-zero at baseline because the base model cannot
      guess the held-out continuation; this is the ceiling reward that
      RL pushes toward.
    * **Diction match** (word-overlap Jaccard) — the softer, learnable
      component.  Rewards writing in the same vocabulary and register
      even if the exact sequence differs.  This is the gradient the
      LoRA can actually learn from at this scale.

    The blend is 50/50 so the model gets signal from either path.  Both
    are mechanically checkable and judge-free (RL_STRATEGY.md §1: no
    online RL against a judge — mode collapse).
    """
    seq_p = ngram_precision(text, case.target, n=3)
    seq_r = ngram_recall(text, case.target, n=3)
    seq_f = ngram_f1(text, case.target, n=3)
    word = word_overlap(text, case.target)

    if case.task == Task.continuation:
        # Precision-weighted: did the model produce text that fits?
        seq_component = seq_p
    elif case.task == Task.restoration:
        # Recall-weighted: did the model recover the original?
        seq_component = seq_r
    elif case.task == Task.infilling:
        # F1: balanced fill
        seq_component = seq_f
    else:
        seq_component = 0.0

    reward = 0.5 * seq_component + 0.5 * word

    detail = {
        "task": case.task.value,
        "precision": seq_p,
        "recall": seq_r,
        "f1": seq_f,
        "word_overlap": word,
        "reward": reward,
        "gen_len": len(_tokenize(text)),
        "tgt_len": len(_tokenize(case.target)),
    }
    return reward, detail


def oracle_reply(case: Case) -> str:
    """The target itself — scores 1.0 by construction.

    ``score_text(case, oracle_reply(case))`` must be 1.0 for every case,
    or the env is asking for something its own grader cannot accept.
    """
    return case.target


# --- Case generation from the corpus -----------------------------------------


def _extract_passages(
    text: str,
    passage_min_words: int = 80,
    passage_max_words: int = 200,
) -> list[str]:
    """Split a book into paragraph-sized passages.

    We want passages long enough to carry a cadence (80+ words) but short
    enough to fit comfortably in a single-turn RL episode (200 cap).  We
    split on double-newlines (paragraph breaks) and merge short adjacent
    paragraphs until we hit the range.
    """
    import re

    raw_paras = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip() and len(p.strip().split()) >= 10
    ]

    passages: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for para in raw_paras:
        w = len(para.split())
        if buf_words + w <= passage_max_words:
            buf.append(para)
            buf_words += w
        else:
            if buf and buf_words >= passage_min_words:
                passages.append(" ".join(buf))
            buf = [para]
            buf_words = w
    if buf and buf_words >= passage_min_words:
        passages.append(" ".join(buf))
    return passages


def make_continuation_case(passage: str, case_id: str, source: dict) -> Case:
    """Split a passage at the 60% mark; prefix → prompt, suffix → target."""
    tokens = passage.split()
    split = max(int(len(tokens) * 0.6), 20)
    prompt = " ".join(tokens[:split])
    target = " ".join(tokens[split:])
    return Case(
        id=case_id,
        task=Task.continuation,
        prompt=prompt,
        target=target,
        source_title=source["title"],
        source_author=source["authors"][0] if source.get("authors") else "?",
        source_id=source["gutenberg_id"],
    )


def make_infilling_case(passage: str, case_id: str, source: dict) -> Case:
    """Delete the middle third of a passage; model fills it."""
    tokens = passage.split()
    n = len(tokens)
    start = n // 3
    end = 2 * n // 3
    prefix = " ".join(tokens[:start])
    suffix = " ".join(tokens[end:])
    target = " ".join(tokens[start:end])
    prompt = f"{prefix} [FILL] {suffix}"
    return Case(
        id=case_id,
        task=Task.infilling,
        prompt=prompt,
        target=target,
        source_title=source["title"],
        source_author=source["authors"][0] if source.get("authors") else "?",
        source_id=source["gutenberg_id"],
    )


def _degrade(text: str) -> str:
    """Flatten a passage toward plain modern English.

    A deliberately crude degradation that strips the markers of the
    weird & eerie register: complex sentences → short declaratives,
    archaic constructions → plain ones, atmosphere → information.  Not
    sophisticated — it just needs to be *worse* so the model has
    something to restore.  The target is the original.
    """
    import re

    t = text
    # Split complex sentences on semicolons, colons, em-dashes
    t = t.replace(";", ". ").replace(":", ". ")
    t = re.sub(r"\s+[–—]\s+", ". ", t)
    # Remove parenthetical asides
    t = re.sub(r"\(.*?\)", "", t)
    # Remove comma-separated subordinate clauses (rough: strip everything
    # between commas that starts with a participle or preposition)
    # Keep it simple: just remove trailing subordinate clauses
    t = re.sub(r",\s+(?:who|which|where|when|while|though|although)\b[^.]*", "", t, flags=re.IGNORECASE)
    # Lowercase all sentence starts (kills the archaic tone)
    t = re.sub(r"([.!?]\s+)([A-Z])", lambda m: m.group(1) + m.group(2).lower(), t)
    # Replace archaic words with plain ones
    replacements = {
        r"\bperchance\b": "maybe", r"\bwhelmed\b": "covered",
        r"\bbetook\b": "went", r"\bforsooth\b": "indeed",
        r"\bverily\b": "truly", r"\bwhence\b": "from where",
        r"\bthither\b": "there", r"\bhither\b": "here",
        r"\begad\b": "wow", r"\btwas\b": "it was",
        r"\bmayhap\b": "maybe", r"\bnigh\b": "near",
    }
    for pat, repl in replacements.items():
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    # Collapse multiple spaces
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def make_restoration_case(passage: str, case_id: str, source: dict) -> Case:
    """Degrade a passage; the model restores the original."""
    degraded = _degrade(passage)
    return Case(
        id=case_id,
        task=Task.restoration,
        prompt=degraded,
        target=passage,
        source_title=source["title"],
        source_author=source["authors"][0] if source.get("authors") else "?",
        source_id=source["gutenberg_id"],
    )


def load_cases_from_corpus(
    manifest_path: Path | str,
    raw_dir: Path | str,
    n_cases: int = 18,
    seed: int = 1899,
    task_weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    max_per_book: int = 2,
) -> list[Case]:
    """Build ``n_cases`` cases from the register slice of the corpus.

    Reads the K3-mini manifest, filters to the ``register`` slice, loads
    raw text from the Gutenberg raw directory, extracts passages, and
    generates cases with a weighted random task type.  The seed is fixed
    so the case set is reproducible across runs.

    ``max_per_book`` limits how many cases come from a single book, so
    the case set draws from multiple authors and voices rather than
    burning through one book's passages.  Default 2 → 18 cases from
    ≥9 books.
    """
    manifest_path = Path(manifest_path)
    raw_dir = Path(raw_dir)
    entries = [
        json.loads(l)
        for l in manifest_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    register = [e for e in entries if e.get("slice") == "register"]
    if not register:
        raise ValueError(
            f"no register-slice books in {manifest_path}. "
            "Check the manifest — the slice field may be missing."
        )

    rng = random.Random(seed)
    rng.shuffle(register)

    cases: list[Case] = []
    per_book: dict[int, int] = {}

    def _pick_task_fn() -> callable:
        r = rng.random()
        if r < task_weights[0]:
            return make_continuation_case
        elif r < task_weights[0] + task_weights[1]:
            return make_infilling_case
        else:
            return make_restoration_case

    for entry in register:
        if len(cases) >= n_cases:
            break
        gid = entry["gutenberg_id"]
        if per_book.get(gid, 0) >= max_per_book:
            continue
        raw_path = raw_dir / f"{gid}.txt"
        if not raw_path.exists():
            continue
        text = raw_path.read_text(encoding="utf-8")
        passages = _extract_passages(text)
        rng.shuffle(passages)
        for passage in passages:
            if len(cases) >= n_cases:
                break
            if per_book.get(gid, 0) >= max_per_book:
                break
            task_fn = _pick_task_fn()
            case_id = f"{gid}-{len(cases):03d}"
            cases.append(task_fn(passage, case_id, entry))
            per_book[gid] = per_book.get(gid, 0) + 1

    # If we still don't have enough, relax the per-book limit
    if len(cases) < n_cases:
        for entry in register:
            if len(cases) >= n_cases:
                break
            gid = entry["gutenberg_id"]
            raw_path = raw_dir / f"{gid}.txt"
            if not raw_path.exists():
                continue
            text = raw_path.read_text(encoding="utf-8")
            passages = _extract_passages(text)
            rng.shuffle(passages)
            for passage in passages:
                if len(cases) >= n_cases:
                    break
                task_fn = _pick_task_fn()
                case_id = f"{gid}-{len(cases):03d}"
                cases.append(task_fn(passage, case_id, entry))

    return cases[:n_cases]
