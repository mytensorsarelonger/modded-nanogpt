# Shakeout `19d639b3` vs M1 baseline — pending §7.2 read-out

Read-only analysis of artifacts pulled from the `k3mini-runs` Modal volume.
Nothing was trained, swept, or committed. Every number below cites the file it
came from; all cited files were fetched by `modal run modal_app.py::fetch` on
2026-09-30 into `modal_out/`.

## Runs

| | M1 baseline | shakeout |
|---|---|---|
| run_id | `21c92807-418e-49ce-a78b-566b376f0914` | `19d639b3-a870-4c35-bcf6-39180d1d4e24` |
| steps | 3250 | 3250 |
| mix | Gutenberg-only (no `mix` field) | `{"backbone":0.735,"register":0.087,"general":0.178}` |
| shard manifest hash | `3708fd6cd5af5654` | `39c7ea793239746a` |
| final val_loss | 2.8055896759033203 | 2.7866079807281494 |
| source | `modal_out/index.jsonl` (no `run.json` on volume) | `modal_out/19d639b3-a870-4c35-bcf6-39180d1d4e24/run.json` |

`21c92807` is the M1 run: its `final_val_loss` in `modal_out/index.jsonl` is
2.8055896759033203, matching the 2.80559 recorded in `AGENTS.md`/`PLAN.md` §8.
`modal volume ls k3mini-runs 21c92807-…` shows `register_eval.jsonl`,
`samples.log` and the checkpoints — **there is no `run.json` for that run on the
volume**, so its registry row comes from `index.jsonl`, which `AGENTS.md` marks
best-effort. Its `val_loss` is single-sourced here.

The two runs trained on different shard manifests (`3708fd6cd5af5654` vs
`39c7ea793239746a`) — M1 predates the general-text slice. Both `run.json` and
`index.jsonl` report `val_tokens` 4333568 for these two runs, and `PLAN.md` §7.2
states the headline val is Gutenberg-only.

## §7.2 harness output — what was actually emitted

Field names below are taken from `evals/exemplar.py`, not guessed. The
`evaluate()` return dict (lines 253–269) emits exactly:

- `exemplar_ppl_nats_per_tok` — quantity 1, register **targets only** (aggregate
  over contrast-pair targets plus singles). Computed by `aggregate()`
  (line 230–233) as `sum(nll_sum)/sum(tokens)`: it is **nats per token, not an
  exponentiated perplexity**, despite `ppl` in the name.
- `contrast_gaps` — per pair: `pair`, `target_ppl`, `wrong_ppl`, `loss_gap`
  (= `wrong_nll_per_tok − target_nll_per_tok`, positive = target preferred),
  `target_nll_per_tok`, `wrong_nll_per_tok`, `target_tokens`, `wrong_tokens`.
- `mean_loss_gap` — mean of `loss_gap` over pairs.
- `craft_ppl_nats_per_tok` — quantity 3, same nats-per-token aggregate.
- `exemplar_passages`, `craft_passages` — per-passage `{id, tokens, nll_sum, ppl}`
  (here `ppl` **is** exponentiated). `eval_set_digest`, `seq_len`.

`exemplar_passages` is `[]` in every row of both runs. That is expected, not
missing: `evals/exemplar_data/register_exemplars.txt` contains only
`seed_cold_open_1` / `seed_mundane_1` and their `.contrast` members, so
`split_pairs()` (line 106) yields no singles. The §7.2 eval set is still the
**seed** set — the owner's hand-written `register_exemplars.txt` has not been
swapped in. `eval_set_digest` is `1e618cc9d92c7619` in every row of both runs.

### Row provenance differs between the two runs

- `19d639b3`: 12 rows, steps 250…3000 every 250, written live during training.
  Keys are `run_id, step, config_hash, seq_len, eval_set_digest,
  exemplar_ppl_nats_per_tok, exemplar_passages, contrast_gaps, mean_loss_gap,
  craft_ppl_nats_per_tok, craft_passages`. No `swept` / `harness_git_sha`.
- `21c92807` (M1): 4 rows, steps 1000/2000/3000/3250, all `"swept": true` with
  `checkpoint` set and `harness_git_sha: 504688ba37f332a92664858e4aaceae4b61483d4`
  (`harness_git_dirty: true`) — the retroactive `::sweep` of 2026-09-27 described
  in `PLAN.md` §7.2. Steps 250/500/750 and 1250/1500/1750/2250/2500/2750 were
  never scored for M1.

**Missing artifact:** `19d639b3` trained 3250 steps with `register_every: 250`
(`run.json`), so a step-3250 §7.2 row is expected, but the trajectory stops at
3000. Verified against the raw on-volume file (`modal volume get
k3mini-runs 19d639b3-…/register_eval.jsonl`): 12 lines, max step 3000 — the
absent row is absent on the volume, not lost in `fetch` de-duplication. Its
`samples.log` likewise ends at step 3000.

## Comparison table

`exemplar` / `craft` columns are `*_ppl_nats_per_tok` (nats/token, lower is
better). `mean_gap` is `mean_loss_gap` (positive would mean the model prefers
the register target over the contrast member).

| step | M1 exemplar | 19d639b3 exemplar | M1 craft | 19d639b3 craft | M1 mean_gap | 19d639b3 mean_gap |
|---|---|---|---|---|---|---|
| 1000 | 4.7047 | 4.3538 | 4.4835 | 4.1566 | −0.3545 | −0.4562 |
| 2000 | 4.6132 | 3.9574 | 4.3698 | 3.8902 | −0.3924 | −0.4528 |
| 3000 | 4.5892 | 3.8110 | 4.2984 | 3.6663 | −0.4119 | −0.5119 |
| 3250 | 4.5021 | *(no row)* | 4.2791 | *(no row)* | −0.3855 | *(no row)* |

Sources: `modal_out/19d639b3-a870-4c35-bcf6-39180d1d4e24/register_eval.jsonl`
(steps 250–3000) and `modal_out/21c92807-418e-49ce-a78b-566b376f0914/register_eval.jsonl`
(steps 1000–3250). Values are the full stored floats, rounded to 4 dp here.

Per-pair detail at step 3000:

| pair | run | target_ppl | wrong_ppl | loss_gap |
|---|---|---|---|---|
| seed_cold_open_1 | M1 | 73.6540 | 81.9016 | **+0.1061** |
| seed_cold_open_1 | 19d639b3 | 39.1311 | 32.4847 | **−0.1861** |
| seed_mundane_1 | M1 | 122.9738 | 48.5199 | −0.9300 |
| seed_mundane_1 | 19d639b3 | 50.4868 | 21.8464 | −0.8377 |

Same two files. `target_tokens`/`wrong_tokens` are 153/121 (cold_open) and
199/131 (mundane) in every row of both runs.

Per-passage craft values at step 3000 (from `craft_passages` in the same two
files): M1 `hazlitt_pleasure_of_painting` 70.4692, `hazlitt_familiar_style`
58.2374, `pater_style` 90.644; 19d639b3 37.7705 / 30.7447 / 48.1535.

## Mundane probe, side by side

Both `samples.log` files contain the same grid: steps 250…3000 every 250, and at
step 3000 a doubled routine pass plus the `*_heldout` families. The `mundane`
family is `probes.py` `Family(name="mundane")`: prompts `An account of how a
kitchen is arranged.` and `The tides are caused by`, at T=0.7 and T=1.0
(`probes.py` lines 48–55, 83).

Provenance: `runs/21c92807-418e-49ce-a78b-566b376f0914/samples.log` and
`runs/19d639b3-a870-4c35-bcf6-39180d1d4e24/samples.log`, both step 3000.

**M1 — collapsed at T=0.7.**

`An account of how a kitchen is arranged.` T=0.7 (second pass):

> What is done and done; what the cooks and cooks and cooks and cooks and cooks and cooks and cooks and cooks and cooks and cooks and cooks and cooks and cooks and cooks and cooks and cooks and cooks an

`The tides are caused by` T=0.7 (first pass):

> the action of the tides; the first cause of the tides is the action of the tides, but the second is the action of the waves. The force of the tide is the action of the waves; the first cause is the ac

**19d639b3 — still collapses at T=0.7, on different text.**

`The tides are caused by` T=0.7:

> tides. They are caused by the rising sea and the tides of the ocean. Their source is in the sun, and the tide is in the waters of the sea. The tides are not caused by tides. It is caused by the natura

`An account of how a kitchen is arranged.` T=0.7 (second pass):

> =A story of the siege of Troy.= =A story of the Trojan war.= =A story of the Trojan war.= =A tale of the Trojan war.= =A story of the Trojan war.= =A story of the Trojan war.= =A story of the Trojan w

At T=1.0 19d639b3 does not loop on these two prompts and stays on topic:
`An account of how a kitchen is arranged.` T=1.0 opens "What is commonly known is
that the kitchen is located in a certain tip of a tree." and `The tides are
caused by` T=1.0 opens "tides as well as climate; which are defined differential
circumstances of causation…". (Its T=0.7 kitchen first pass also opens on topic —
"What is the difference between the four kinds of kitchen? … The following are
the ordinary cookery recipes:" — before collapsing on the second pass.) M1 at
T=1.0 loops on neither prompt either, but its kitchen text drifts to cigars and a
cafe ("what the smoking of the tip of a cigarette does or may turn out to be").

Repetition measured the same way on both files (my own count, not a harness
field): longest repeated word n-gram within each step-3000 mundane
continuation, n=5…12, min 2 occurrences.

| run | family / prompt | T | longest repeat |
|---|---|---|---|
| M1 | mundane / kitchen (pass 1) | 0.7 | 12 words × 2 ("has to cover up the floor with a new made cloth and") |
| M1 | mundane / kitchen (pass 2) | 0.7 | 12 words × **59** ("cooks and cooks and …") |
| M1 | mundane / tides (pass 2) | 0.7 | 12 words × 3 ("the earth's rotation is the same as that of the sun's action") |
| M1 | mundane / tides | 1.0 | none ≥5 words repeated |
| 19d639b3 | mundane / kitchen (pass 1) | 0.7 | 5 words × 2 ("the four kinds of kitchen") |
| 19d639b3 | mundane / kitchen (pass 2) | 0.7 | 12 words × **12** ("a story of the trojan war a story of the trojan war") |
| 19d639b3 | mundane / tides (pass 1) | 0.7 | 12 words × 5 ("are caused by the tides of the sea they are caused by") |
| 19d639b3 | mundane / tides (pass 2) | 0.7 | 8 words × 3 ("is the cause of the attraction of the") |
| 19d639b3 | mundane / tides | 1.0 | none ≥5 words repeated |

`mundane_heldout` (`A short explanation of how paper is made.`, step 3000): M1
T=0.7 repeats "is made up of a number of pieces" ×2; 19d639b3 T=0.7 repeats "is
prepared in a manner that is" ×2. Neither loops at T=1.0.

## Findings

- **The §7.2 curves were never read before now and they do not show the
  discrimination the harness exists to measure.** `mean_loss_gap` is negative at
  every scored checkpoint of both runs. At the last common step (3000) it is
  −0.4119 for M1 and −0.5119 for 19d639b3: the shakeout mix moved the aggregate
  gap further from zero, not toward it. Source: the two `register_eval.jsonl`
  files above.
- **The negative gap is not uniform across the two seed pairs.** At step 3000
  `seed_cold_open_1` has a positive gap for M1 (+0.1061) and a negative one for
  19d639b3 (−0.1861); `seed_mundane_1` is negative for both (−0.9300 / −0.8377).
  The aggregate is dominated by the mundane pair. Same files, `contrast_gaps`.
- **`exemplar_ppl_nats_per_tok` and `craft_ppl_nats_per_tok` are lower for
  19d639b3 at every common step** (exemplar 4.3538/3.9574/3.8110 vs M1
  4.7047/4.6132/4.5892; craft 4.1566/3.8902/3.6663 vs 4.4835/4.3698/4.2984), as
  is `final_val_loss` (2.78661 vs 2.80559). Both runs scored the same eval set
  (`eval_set_digest` `1e618cc9d92c7619`) but trained on different shard manifests
  (`39c7ea793239746a` vs `3708fd6cd5af5654`). Same files.
- **The mundane probe still collapses.** At step 3000, T=0.7, both runs loop:
  M1 on the kitchen prompt (12-word cycle ×59) and the tides prompt; 19d639b3 on
  the tides prompt (12-word cycle ×5) and on its second kitchen pass (12-word
  cycle ×12). At T=1.0 neither run's step-3000 mundane continuation contains a
  repeated 5-word sequence. Sources: the two `samples.log` files.
- **Two artifacts the task expected are absent, and this is stated rather than
  filled in.** (a) `19d639b3` has no step-3250 §7.2 row and no step-3250 samples,
  though it trained 3250 steps at `register_every: 250` — confirmed absent on the
  volume, not dropped by `fetch`. (b) `21c92807` (M1) has no `run.json` on the
  volume, so its val_loss rests on the `index.jsonl` row alone; and it has §7.2
  rows only for steps 1000/2000/3000/3250, so steps 250–750 and 1250–2750 are
  unscored. Also: the §7.2 eval set is still the two-pair **seed** set, so
  `exemplar_passages` is empty by construction and every `mean_loss_gap` above
  rests on two pairs.

## Files used

| path (as fetched) | on-volume provenance |
|---|---|
| `modal_out/19d639b3-a870-4c35-bcf6-39180d1d4e24/register_eval.jsonl` | `runs/19d639b3-a870-4c35-bcf6-39180d1d4e24/register_eval.jsonl` |
| `modal_out/19d639b3-a870-4c35-bcf6-39180d1d4e24/run.json` | `runs/19d639b3-a870-4c35-bcf6-39180d1d4e24/run.json` |
| `modal_out/19d639b3-a870-4c35-bcf6-39180d1d4e24/samples.log` | `runs/19d639b3-a870-4c35-bcf6-39180d1d4e24/samples.log` |
| `modal_out/21c92807-418e-49ce-a78b-566b376f0914/register_eval.jsonl` | `runs/21c92807-418e-49ce-a78b-566b376f0914/register_eval.jsonl` |
| `modal_out/21c92807-418e-49ce-a78b-566b376f0914/samples.log` | `runs/21c92807-418e-49ce-a78b-566b376f0914/samples.log` |
| `modal_out/index.jsonl` | `index.jsonl` (best-effort registry; M1 row only) |

Repo-side sources read, not fetched: `evals/exemplar.py` (field names),
`probes.py` (mundane prompts/temperatures), `evals/exemplar_data/*` (eval set),
`PLAN.md` §7.2, `AGENTS.md`, `modal_app.py::collect_results`/`fetch`.