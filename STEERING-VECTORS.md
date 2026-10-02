# STEERING-VECTORS.md — a steering-vector lens on this project's training

Written 2026-10-01 from a tweet thread on steering-vector intuitions, reproduced
verbatim in §1. **Unlike MIX-LITERATURE.md, the source is not peer-reviewed and
none of its claims are verified against anything** — it is a practitioner's
working intuitions plus one anecdote. The author has published in this area (§1
lists the related work), which raises the odds the intuitions are good; it does
not make the claims evidence. §2 separates what in it is testable from what is a
lens, and §6 states what would falsify the recommendations. Purpose: name the
mechanism-level readout the register/mix work currently lacks, and the controls
any such measurement owes. Referenced from PLAN.md §7.2.

The one-sentence synthesis: **the mix ratio is a feature-prevalence knob, so a
"register vector" is the same kind of intervention as the register slice at a
different strength — and it can be measured in forward passes, using checkpoint
pairs this project already has on the volume.**

---

## 1. The source (verbatim)

> a couple lenses on / intuitions about steering vectors i find useful
>
> persona vectors, concept vectors, belief vectors, task vectors are all the same
> thing, which can be constructed and applied in different ways (mean differences,
> PCA, NLA reconstructor, SAE, etc., activation addition, patching, etc.)
>
> the LLM residual stream is a big, noisy bag of representations. the goal in
> extracting a steering vector is to somehow cut down that noisy bag into one or
> more clean, reusable, and interpretable representations.
>
> so for example in mean differences, the idea is to cancel the distractors out on
> either side of the subtraction. imagine you have some distribution of "user age"
> features popping up in the residual stream that you want to ignore while
> extracting, say, model distress. if you just averaged a bunch of texts where the
> model is distressed, the resulting "distress vector" will be entangled with the
> average of those user age features! but with mean differences, assuming the
> distribution of user age in the distress and non-distress sides is the same, then
> the user age feature cancels:
> `(distress + age) - (no_distress + age) = distress`
>
> ...but wait, what's this "distress" thing? it popped up in the residual stream
> "on texts where the model is distressed," but everyone knows when e.g. doing
> regular evals that model behavior is more complicated than that. the problem
> doesn't go away just because you invoked a SteeringVector trainer and are now
> "doing interpretability research" with impressive-looking tensors. you're still
> dealing with LLMs and their confusing nature and multiple realizability of
> behavior.
>
> so another way you can view steering vectors is as bags of SAE features - pass a
> steering vector straight through an SAE and you'll get out labeled features.
> (usual SAE caveats applying.)
>
> in this view, when you're extracting a steering vector, what you're doing is
> collecting a clean bag of features, say with hypothetical labels like:
> `( "Robot in fiction acts in distress", "Distressed person", ..., "25-year-old software user", "User (Egyptian official)", ..., ) - ( "25-year-old software user", "User (Egyptian official)", ..., ) = ( "Robot in fiction acts in distress", "Distressed person", ..., )`
>
> but again, just because that bag of features was elicited from the texts you
> used, doesn't mean it's for the reasons you think!
>
> if you pass "I am feeling very distressed..." to different models:
> one could pull in features related to roleplaying, acting, etc: "Actor pretends
> to be distressed in movie"
> another could pull in features related to a fictional character: "Character in
> novel is distressed"
> and a third could pull in features related to true belief in distress.
>
> how do you know which one you have? it's evaluations again, except on hard mode
> because steering vectors are tricky to work with correctly (e.g. due to
> off-target effects.)
>
> what do i mean by true belief? we don't really know, but there is a difference -
> e.g. you can finetune a model to "believe" a conspiracy theory, but the
> activations will betray it's still thinking in terms of the real world and then
> just translating to the theory. but for example RL seems to more "truly" embed
> beliefs into models. something something RL is for epistemics something something
> off v.s. on-policy something something inferential distance.
>
> another lens is that steering vectors are kind of like super-low-rank LoRA
> finetuning. (even weaker than a rank-1 LoRA - it's like a bias-only finetune.)
>
> in fact, you can construct a steering vector from two checkpoints on the same
> context instead of two contexts on one checkpoint. just take the average
> activations from both checkpoints and do the same as you normally would.
>
> again, the intuition here being finetuning is "just" pushing around the
> prevalence of features / representations in the finetune's residual stream.

**Source:** Thebes Vogel (@voooooogel; also theia vogel / vgel — vgel.me),
<https://x.com/voooooogel/status/2105793927035093490>, captured 2026-10-01.

**Related work by the same author, where the thread's methods actually live:**
the `repeng` control-vector library (mean-difference control vectors trained from
synthetic data); the cross-model emergent-misalignment steering vector (positive
examples: Qwen2.5-Coder activations, negative: the misaligned
Qwen-Coder-Insecure) — this is the "anti-emergent-misalignment vector" and the
extraction method the thread alludes to; and *Latent Introspection: Models Can
Detect Prior Concept Injections* (Pearson-Vogel, Vaněk, Douglas, Kulveit —
arXiv:2602.20031, submitted ICML 2026), which is the "Latent Introspection
submissions" the thread mentions.

Worth noting what that last paper measures, because it is the same shape as this
project's problem: a Qwen 32B model **denies** the injection in its sampled
output while the logit lens shows clear detection signals in the residual stream.
A model's stated behavior and what its residual stream carries can come apart —
which is precisely the gap this document is about, and a reason to read the
`mundane` probe and the §7.2 gap as behavioral instruments only.

## 2. Testable claims vs. lenses

Read this section before using §5. Half the source's value is method and does not
depend on its anecdote being reproducible.

**Testable, and cheap to test here:**

- **Cancellation is a property of the design, not of the subtraction.** The
  identity `(distress + age) − (no_distress + age) = distress` holds only if the
  distractor distribution is matched across the two sides. This is the balanced-
  experiment assumption, and it is *checkable before extracting anything*: fit a
  probe for the suspected distractor on both sides and report the balance. An
  unbalanced pair set is fixed with a better pair set, not a cleverer vector.
- **The checkpoint-diff construction.** "Take the average activations from both
  checkpoints and do the same as you normally would." Same contexts, two
  checkpoints, mean difference. Same operation as task vectors in model
  arithmetic. **This project has the checkpoint pairs already** (§3).
- **Capacity: a steering vector is weaker than a rank-1 LoRA.** Formally, a
  rank-1 LoRA is `Δh = b(aᵀx)` — a fixed direction with an input-dependent gate.
  A steering vector is the same object with the gate replaced by a constant:
  `Δh = αd`. So the limiting factor is not rank but **input-conditioning**: it
  cannot route on the input. A construct whose expression requires conditional
  application ("distressed about this, not that") will smear, and no care in
  extraction fixes it. That is a falsifiable prediction about any construct,
  runnable before committing to a direction.
- **Multiple realizability.** The same eliciting text can pull "actor pretending",
  "character in a novel", or "true belief" in different models. Consequence: a
  high within-model effect with low cross-model transfer means you have a
  model-specific idiosyncrasy wearing a concept's name. Cross-model transfer *is*
  the instrument.
- **Belief vs. translation.** "You can finetune a model to believe a conspiracy
  theory, but the activations will betray it's still thinking in terms of the real
  world and then just translating." That is probeable: if the model asserts X
  while a probe still recovers the real-world fact (or its base rate), you have
  translation; if the real-world encoding is gone, you have belief. Interventional
  evidence, not correlational.
- **The controls list (§4).** Prompt, sampler, finetune, norm-matched random
  vector. These are cheap and they are the difference between a measurement and a
  measurement-shaped object.

**Lenses and intuitions — useful for framing, not evidence:**

- "Big, noisy bag of representations"; extraction as cutting it down. A way to
  think about the mean-difference recipe, not a claim.
- "Bags of SAE features." Operationally useful (§3) but carries the usual SAE
  caveats the source itself flags.
- The **"Certainly!" result is n=1**: one vector, one model, one failure mode.
  Its value here is as a *warning about off-target effects and overshoot* — a
  negative direction is not "the opposite behavior", it is the nearest available
  mode past the training manifold, and degenerate repetition is the generic way
  that looks. Do not cite it as a result.
- "RL is for epistemics, something something on-policy." An interesting
  hypothesis, unoperationalized in the source. §5 R5 proposes the test.
- The closing line — "finetuning is 'just' pushing around the prevalence of
  features / representations" — is the most consequential sentence for this
  project. §3.

## 3. Why this lands on this project specifically

**1. The measured failure is exactly the multiple-realizability failure.** AGENTS.md
records it: Milestone 1's prose is "competent Victorian pastiche and not eerie", and
the `mundane` probe collapsed into repetition. The source's diagnosis of the same
shape — "actor pretends to be distressed in movie" vs "character in novel is
distressed" vs true belief — is a description of *Victorian pastiche vs the
register*. The register slice may be moving the costume features rather than the
dread features, and the current instruments cannot tell those apart. The §7.2 gap
says how much worse the model finds the target than the decoy; it does not say
*which features moved*.

**2. The mix ratio is a prevalence knob, and this reframes §4.2.** If a finetune is
"pushing around the prevalence of features in the residual stream", then the
register slice weight is that same intervention at full strength, and a register
vector is that intervention at the minimum strength that is still measurable. The
register weight (8.7%, target 10–25%, PLAN.md §5.1.2) is currently chosen from
loss and the §7.2 gap. A vector gives the mechanism: *did the slice actually move
the representation toward the register, or only change the output distribution?*

**3. The checkpoint pairs already exist, and one of them is a natural experiment.**
The tweet's between-checkpoint method needs same-context, two-checkpoint pairs.
This project has:

- **Within a run:** M1 baseline `21c92807`, dense checkpoints every 250 steps
  (1000/2000/3000/3250). Baseline-vs-final is "what did training do to the
  representation", directly comparable to the §7.2 gap curve that is *negative
  across the whole curve* (−0.35 → −0.41 nats/tok, deepening).
- **Between runs:** M1 baseline vs the general-slice shakeout `19d639b3`
  (backbone .735 / register .087 / general .178, val 2.78661 vs 2.80559). The
  repo has already recorded that the general slice moved the §7.2 gap **the wrong
  way** (commit `ceb2ad3`). A checkpoint-diff vector between these two on
  identical contexts says *whether the general slice moved the representation away
  from the register direction*. That converts a number into a mechanism, and it is
  forward passes only.

**4. §7.2's contrast pairs are already a mean-difference design.** The harness
(`evals/exemplar.py`) holds target writing vs. superficially-similar-but-wrong
writing, per-passage, with a contamination guard. That is the pair structure the
source's recipe wants — "two contexts on one checkpoint". The extractor is a
small addition to an existing artifact, not a new instrument. Per AGENTS.md rule 3,
it should be **one file that does the thing**, not an interpretability framework.

**5. The `mundane` probe and the sampler observation are the same idea.** §7.1
makes `mundane` load-bearing precisely because register-as-costume survives a
haunted-house prompt and dies on a kitchen. The source's sampler example (a
detector that injects "stay focused" when the trace wavers, then calling the result
goal-orientation) is the general form of that: **an output-space controller is not
a representation edit.** Worth keeping straight when reading `eerie_rl` results —
a reward shaping a rollout distribution is not evidence about what the model
"believes".

## 4. The control battery — runnable checklist

Any claim of the form "steering on this direction does X" owes all five. Four are
cheap; the fifth is the one people skip.

1. **Prompt control.** Does a prompt produce the effect? If yes, the direction
   exists but is not privileged — the interesting cases are the *dissociations*
   (behavior steerable but not promptable, or the reverse). Agreement tells you
   the direction exists; disagreement tells you it is doing work the prompt
   pathway does not.
2. **Sampler control.** Does the effect survive holding the decoding distribution
   fixed? If reweighting or temperature reproduces it, you changed the
   distribution, not the representation.
3. **Finetune control.** Fair only if trained on the **same contrast** the vector
   was extracted from. Vectors are licensed by a contrast (a handful of paired
   examples); finetunes are licensed by a distribution. Comparing a contrast-fit
   vector against a distribution-fit LoRA proves nothing.
4. **Negated vector.** The opposite sign should produce the opposite behavior.
   When it does not, you have learned something real about the axis — the source's
   "Certainly!" case is the canonical failure, and it is informative, not fatal.
5. **Norm-matched random vector.** The most-skipped and the most informative.
   Random directions in high dimension are near-orthogonal to everything, so if a
   random push of matched norm reproduces the effect, you have demonstrated a
   magnitude effect, not a directional one. (Better: match the norm *and* sample
   the direction from the layer's own activation distribution.)

**And the balance check from §2 belongs in the extractor, not in a notebook:**
report the distractor balance of the pair set before reporting the vector.

**A vector claim is a Case with a verifier at the juncture** (the `synthetic`
repo's shape): hypothesis, controls, held-out causal test, report. A direction
that fails the held-out causal test is a correlation wearing a mechanism's clothes.

## 5. Recommendations

**R1 — Extract the register vector from the two checkpoints that already exist.**
M1 baseline vs general-slice shakeout, identical contexts, mean activation
difference at the residual stream across layers. Question it answers: did the
general slice move the representation away from the register direction? This is
the mechanism behind a result already recorded behaviorally, and it needs no
training — forward passes on Modal, `modal run`-shaped per AGENTS.md rule 1.

**R2 — Extract the within-run curve vector and check it against the §7.2 gap.**
Baseline ckpt 1000 vs 3250, same contexts. Prediction to falsify: if the
contrast-pair gap deepens (−0.35 → −0.41), the between-checkpoint vector should
point *away* from the exemplar direction. If the gap and the representation
disagree, that is a finding about the harness, and worth more than the vector.

**R3 — Name the features, don't just locate them (higher cost).** Pass the
register vector through an SAE and read the labels — the source's bag-of-features
view. This is the only one of these that can distinguish "Victorian novel style"
features from "dread/uncanny" features, i.e. costume from register. **Cost
warning:** there is no pretrained SAE for a from-scratch model at this scale, so
this means training one. Do it after R1/R2 show there is a direction worth naming.

**R4 — Build the extractor inside §7.2, not beside it.** Same contrast pairs, same
JSONL contract, one balance check in the loop. One file. If the pair set turns out
to be distractor-unbalanced, that is a finding about the §7.2 eval set that
matters independently of any vector.

**R5 — `eerie_rl`: test SFT vs RL on intervention strength, not behavior.** The
source's belief-vs-translation distinction is the hypothesis that RL embeds a
register "more truly" than SFT. Operationally: train the same register by SFT and
by GRPO, then compare **not the outputs (both can match) but how much a probe on
the register direction — or an ablation of it — changes the output.** An
intervention that moves a genuinely-embedded feature should be stronger than one
that moves a translated one. `RL_STRATEGY.md` §3's pass@k-of-base discipline
applies unchanged: record base pass@k before claiming any RL gain.

## 6. What would falsify this, and the honest caveats

- **If R1/R2 find no direction that separates the checkpoints on the exemplar
  pairs, the lens does not apply here** and the register question stays behavioral.
  That is a real possible outcome: a 120M-parameter model at 2B tokens may not have
  a linearly-accessible register direction at all. Report it as such.
- **The source is a tweet.** No peer review, no replication, and its one
  quantitative anecdote is n=1. Nothing here should be cited as literature the way
  MIX-LITERATURE.md's papers can be. (The author's *related papers* listed in §1
  are a different matter — those are citable, and `repeng` is the tool that
  implements the extraction recipe.) The *method* (mean differences, the control
  battery, the balance check) is standard and stands on its own; the *claims about
  what steering shows* do not.
- **The transfer risk is real:** the source's examples are instruction-tuned
  chat models on persona/emotion axes. This project's model is a from-scratch
  literary base model on a register axis. Persona axes may be unusually
  linearly-accessible; a literary register may not be.
- **Off-target effects are structural, not a tuning problem** (§2, capacity
  bullet). Any steering result on this model should be reported with the
  off-target mass, not only the target effect.
- **A steering vector cannot tell you the model is experiencing anything.** The
  source says this directly and it matches this repo's discipline: mechanical
  findings with the evidence that produced them, no judge, no phenomenology
  smuggled in as a measurement.