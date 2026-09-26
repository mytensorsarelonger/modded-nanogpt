"""Eerie Tinker RL environment — one Case as one Tinker RL episode.

Generalizes ``foundermath/tinker_env.py`` from the synthetic repo:

* One ``Case`` → one ``Env``, exactly as in the Founder Math env.
* ``initial_observation()`` renders the case prompt with the model's
  renderer, plus a system line asking for literary continuation /
  restoration / infilling.
* ``step(action)`` decodes tokens → text → ``score_text(case, text)``
  → reward.  Single step, ``episode_done=True``.
* ``CaseGroupBuilder`` yields ``group_size`` copies (GRPO needs groups).
* ``CaseDataset`` / ``CaseDatasetBuilder`` iterate over the case list.

The reward is ``score_text`` — the same function any offline scorer
calls, so the smoke test and the RL loop see identical numbers.  The
reward is a float in [0, 1], not binary, because literary restoration
is graded; but it is still mechanically checkable and judge-free.

Built against **tinker-cookbook 0.5.x / tinker 0.29.x** — ``Action``,
``StepResult``, ``EnvGroupBuilder``, ``RLDataset`` are that version's
shapes.  Read ``tinker_cookbook/rl/types.py`` and one worked recipe
before trusting this; do not invent the interface.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .case import Case, Task, score_text


# --- System lines -----------------------------------------------------------

_BASE_SYSTEM = (
    "You are a literary model fluent in the weird and eerie register — "
    "prose in the tradition of Poe, Machen, Blackwood, Dunsany, and the "
    "early weird tale. Continue, restore, or fill the text in that voice. "
    "Do not explain or comment. Produce only the passage."
)

_TASK_INSTRUCTION: dict[Task, str] = {
    Task.continuation: (
        "Continue the passage that follows. The text given is the opening; "
        "write what comes next, in the same register and voice."
    ),
    Task.restoration: (
        "The text given has been flattened and modernised. Restore it to "
        "its original weird and eerie voice. The restored passage should "
        "carry the same content but the original cadence, diction, and "
        "atmosphere."
    ),
    Task.infilling: (
        "The text contains a [FILL] marker where a passage has been "
        "removed. Write the missing passage that belongs there, in the "
        "same register and voice as the surrounding text."
    ),
}


def system_line_for(case: Case) -> str:
    """The base instruction plus the task-specific direction."""
    return f"{_BASE_SYSTEM}\n\n{_TASK_INSTRUCTION.get(case.task, '')}"


# --- The environment ---------------------------------------------------------


class EerieEnv(Env):
    """Single-turn env over one ``Case``."""

    def __init__(self, case: Case, renderer: renderers.Renderer) -> None:
        self.case = case
        self.renderer = renderer
        self.system_line = system_line_for(case)

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _messages(self) -> list[renderers.Message]:
        return [
            {"role": "system", "content": self.system_line},
            {"role": "user", "content": self.case.prompt},
        ]

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return (
            self.renderer.build_generation_prompt(self._messages()),
            self.stop_condition,
        )

    async def step(
        self, action: Action, *, extra: ActionExtra | None = None
    ) -> StepResult:
        message, termination = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        reward, detail = score_text(self.case, content)

        metrics: Metrics = {
            "reward": reward,
            "precision": float(detail["precision"]),
            "recall": float(detail["recall"]),
            "f1": float(detail["f1"]),
            "word_overlap": float(detail["word_overlap"]),
            "gen_len": float(detail["gen_len"]),
            "tgt_len": float(detail["tgt_len"]),
            "clean_stop": float(termination.is_clean),
            f"task/{self.case.task.value}": 1.0,
        }
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
            logs={
                "case_id": self.case.id,
                "task": self.case.task.value,
                "source": f"{self.case.source_title} ({self.case.source_author})",
                "precision": detail["precision"],
                "recall": detail["recall"],
                "f1": detail["f1"],
                "word_overlap": detail["word_overlap"],
                "response": content,
            },
        )


# --- Group builder (GRPO) ----------------------------------------------------


@dataclass(frozen=True)
class EerieGroupBuilder(EnvGroupBuilder):
    """``group_size`` copies of one case — GRPO centers rewards across them."""

    case: Case
    renderer: renderers.Renderer
    group_size: int
    dataset_name: str = "eerie-rl"

    async def make_envs(self) -> Sequence[Env]:
        return [
            EerieEnv(self.case, self.renderer)
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name, f"task:{self.case.task.value}"]


# --- Dataset -----------------------------------------------------------------


class EerieDataset(RLDataset):
    """Batches of ``EerieGroupBuilder``, one builder per case.

    ``batch_size`` is capped at the number of cases so a batch never
    repeats a case, exactly as in ``CaseDataset``.
    """

    def __init__(
        self,
        cases: list[Case],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
    ) -> None:
        if not cases:
            raise ValueError("no cases")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.cases = cases
        self.batch_size = min(batch_size, len(cases))
        self.group_size = group_size
        self.renderer = renderer

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = (index * self.batch_size) % len(self.cases)
        rows = [
            self.cases[(start + i) % len(self.cases)]
            for i in range(self.batch_size)
        ]
        return [
            EerieGroupBuilder(case, self.renderer, self.group_size)
            for case in rows
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.cases) / self.batch_size)


# --- Dataset builder (chz) --------------------------------------------------


@chz.chz
class EerieDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    cases_json: str  # serialised list of case dicts — chz can't hold Case objects

    async def __call__(self) -> tuple[EerieDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(
            self.renderer_name, tokenizer=tokenizer
        )
        cases = _cases_from_json(self.cases_json)
        return (
            EerieDataset(
                cases=cases,
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
            ),
            None,
        )


def _cases_from_json(s: str) -> list[Case]:
    """Deserialise cases from JSON (chz can't hold Case objects)."""
    import json

    from .case import Task

    raw = json.loads(s)
    return [
        Case(
            id=c["id"],
            task=Task(c["task"]),
            prompt=c["prompt"],
            target=c["target"],
            source_title=c["source_title"],
            source_author=c["source_author"],
            source_id=c["source_id"],
        )
        for c in raw
    ]


def cases_to_json(cases: list[Case]) -> str:
    """Serialise cases to JSON for the chz builder."""
    import json

    return json.dumps(
        [
            {
                "id": c.id,
                "task": c.task.value,
                "prompt": c.prompt,
                "target": c.target,
                "source_title": c.source_title,
                "source_author": c.source_author,
                "source_id": c.source_id,
            }
            for c in cases
        ]
    )
