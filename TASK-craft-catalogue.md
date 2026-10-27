# TASK-craft-catalogue.md — side-quest brief: assemble craft-essay slice candidates

Handoff brief for a cheaper/assistant agent. The owner vetoes everything; your
job is to make the veto fast. Do not edit anything outside this task's output
file. Do not commit. Read AGENTS.md first, then this file.

## The task

PLAN.md §5.1.2 puts the craft-essay slice's "genuinely by hand" work in the
owner's column: *which* prefaces, craft essays, criticism, and writers'-letters
volumes exist as separate public-domain texts is a catalogue-reading problem.
You cannot do the taste part. You CAN assemble a well-structured candidate
list the owner marks up in minutes instead of spending days in the catalogue.

**Deliverable: `craft_candidates.md`** (repo root, uncommitted) — a table of
candidate volumes, one row per candidate, with exactly these columns:

| pg_id | title | author | category | est_tokens | in_corpus_author | conflict | one_line_pitch |
|-------|-------|--------|----------|------------|------------------|----------|-----------------|

- `category` ∈ {preface, craft_essay, criticism, letters_on_writing}
  (PLAN.md §4.2 job 2 wants "author thinks aloud, then writes" — deliberation-
  shaped text. A preface explaining the book's making is the archetype.)
- `est_tokens` — rough is fine (chars/4). Use the PG text length.
- `in_corpus_author` — TRUE/FALSE: is the author present in
  `data/manifest.jsonl` (grep the `authors` field)? Both are acceptable; the
  column exists so the owner sees which candidates are new voices vs
  reweighting existing ones.
- `conflict` — see hard constraints below; write "clean", "eval-holdout",
  "already-in-corpus", or "val-book".
- `one_line_pitch` — one sentence on why it belongs in a craft-essay slice.
  No marketing language; state what the text IS.

Target: **60–120 candidates**, spanning all four categories. Quality over
quantity within that range.

## Hard constraints (a candidate that violates these is worthless)

1. **Exclude PG 3020 (Hazlitt, Table-Talk) and PG 4037 (Pater,
   Appreciations)** entirely — passages from both are the §7.2 held-out eval
   set (`evals/exemplar_data/craft_essays.txt`); training on them destroys
   the eval. Mark any other Hazlitt/Pater volume `conflict=eval-holdout`
   rather than including it (same-author risk of near-duplicate passages —
   the 12-token contamination guard will catch literal overlap, but the
   owner should decide on same-author volumes).
2. **Exclude anything whose PG id appears in `data/manifest.jsonl`**
   (`gutenberg_id` field) — already in the training corpus. Mark it
   `already-in-corpus` in the table (still list it: the owner may want it
   *moved* to the craft slice).
3. **Exclude the 30 holdout books in `data/val_books.json`** — those are
   never training data.
4. Public domain, English, on Project Gutenberg (use the gutendex API:
   `https://gutendex.com/books?search=...` — it exists for exactly this).
5. Whole volumes only (a letters collection, a criticism volume), not single
   essays you'd have to extract from misc collections — the slice pipeline
   (`download_gutenberg.py --author`, `tokenize_corpus.py`) works per-book.

## Where to look (seed directions, not a limit)

- Letters-on-writing: Keats's letters (collected editions), Stevenson's
  essays (NOT "A Humble Remonstrance" — it rides in collected-works
  editions already in corpus; mark those `already-in-corpus`), Twain's
  "How to Write" / letters, Henry James's prefaces (the NY Edition prefaces
  if separately published), Coleridge's Biographia Literaria (check corpus
  first — likely present), Poe's marginalia volumes (NOT "Philosophy of
  Composition" — same collected-works contamination).
- Craft essays / criticism: De Quincey's "Letters to a Young Man" and
  essays on style/knowledge, Sidney's "Defense of Poesy", Dryden's essays,
  Johnson's "Rambler"/"Lives of the Poets" prefaces, Leigh Hunt's essays
  (NOT Charles Lamb — Symons 21407 quotes him at length and is in-corpus),
  George Orwell's essays (check PD status carefully — most are NOT PD),
  Woolf's "Common Reader" (check PD status), Brander Matthews, Hamilton
  Wright Mabie, Bliss Perry "A Study of Prose", Arthur Quiller-Couch
  "On the Art of Writing" (his lectures: excellent archetype), Sherwin
  Cody, Stephen Leacock "How to Write".
- Prefaces: use gutendex subject/search on "preface", "introduction to",
  "essay on style", "art of fiction", "letters to a young", "on writing".

Verify every pg_id via gutendex before listing it (title+author+PD). If you
cannot verify, do not list it.

## What NOT to do

- Do not download the corpus, tokenize, shard, or touch manifests — that is
  the mechanical pipeline and runs after the owner's veto.
- Do not edit AGENTS.md, PLAN.md, or anything under evals/, data/, tests/.
- Do not run GPU/Modal anything. This task is laptop-safe and CPU-only.
- Do not commit. Write `craft_candidates.md` and stop.

## Definition of done

`craft_candidates.md` exists at repo root with the exact columns above, 60+
rows, every pg_id verified, conflicts marked per the constraints, and a
short method note (2–3 paragraphs) at the top: what you searched, what you
excluded and why, and where you think the list is weakest (the owner reads
that note to calibrate trust in the rest).
