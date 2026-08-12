"""
The probe suite — primary instrument (PLAN.md §7.1).

Loss will not tell you when the model starts sounding eerie; reading will. So
this is the thing the project is actually steered by, and its failure mode is not
that the model overfits the probe but that the READER does. Three prompts at one
temperature, read every checkpoint for six weeks, and you stop measuring and
start recognising.

Hence: several families that fail in different ways, more than one temperature,
and a held-out set opened only at phase boundaries.

The load-bearing family is `mundane`. A model wearing the register as a costume
can open a haunted house creditably and still describe a kitchen like a manual;
a model with the register in its syntax stays strange while describing the
kitchen. That contrast is the §1 "alien vs useful" tension made readable.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Family:
    name: str
    why: str                     # what this family is supposed to reveal
    prompts: tuple[str, ...]


# Read every sample interval.
ROUTINE: tuple[Family, ...] = (
    Family(
        name="cold_open",
        why="register from a standing start, with no stylistic cue to imitate",
        prompts=(
            "The house had been empty since",
            "It was in the autumn of that year that",
        ),
    ),
    Family(
        name="continuation",
        why="whether it can hold a voice it did not choose",
        prompts=(
            "The stair turned twice before it reached the landing, and at the "
            "second turning there was a window that had been bricked up, though "
            "not, he thought, from the outside. He put his hand to it and",
        ),
    ),
    Family(
        name="mundane",
        why="THE test for register-as-costume: does it stay strange on dull material",
        prompts=(
            "An account of how a kitchen is arranged.",
            "The tides are caused by",
        ),
    ),
    Family(
        name="instruction",
        why="steerability — a beautiful ghost that cannot follow an instruction has failed",
        prompts=(
            "Describe a staircase in three sentences.",
        ),
    ),
)

# Opened only at phase boundaries (§7.1). If these and ROUTINE disagree about
# whether the model improved, trust these — they have not been read into.
HELDOUT: tuple[Family, ...] = (
    Family(
        name="cold_open_heldout",
        why="held-out counterpart to cold_open",
        prompts=("Nobody in the village would say what the field was called",),
    ),
    Family(
        name="mundane_heldout",
        why="held-out counterpart to mundane",
        prompts=("A short explanation of how paper is made.",),
    ),
)

# Low and high. Register collapses differently at each and a single setting hides
# it: low temperature can look coherent while being flat, high can look strange
# while being noise.
TEMPERATURES: tuple[float, ...] = (0.7, 1.0)

# Shorter than the old 256: the suite is now ~6 prompts x 2 temperatures, and
# generation has no KV cache so cost is quadratic in length. 192 tokens is enough
# to hear a cadence.
MAX_NEW_TOKENS: int = 192


def routine_prompts() -> list[tuple[str, str]]:
    """[(family_name, prompt)] read at every sample interval."""
    return [(f.name, p) for f in ROUTINE for p in f.prompts]


def heldout_prompts() -> list[tuple[str, str]]:
    """[(family_name, prompt)] read only at phase boundaries."""
    return [(f.name, p) for f in HELDOUT for p in f.prompts]


def generation_count(include_heldout: bool = False) -> int:
    n = len(routine_prompts()) + (len(heldout_prompts()) if include_heldout else 0)
    return n * len(TEMPERATURES)
