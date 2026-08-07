"""Invariants of the two-finger DAgger distillation tasks.

The distillation env feeds a frozen v11_align checkpoint through its "actor"
observation group. Those weights were fitted to that vector in that order, so
any change to the group's contents silently feeds the teacher something it was
never trained on -- which shows up as a mediocre student rather than an error.
These tests pin the group down.
"""

from __future__ import annotations

import pytest

from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import (
  _PRIVILEGED_TERMS,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TEACHER_TASK = "Mjlab-Grasp-TwoFinger-Flexiv-LiftGoal-TouchCalmAlign"
DISTILL_TASKS = [
  "Mjlab-Grasp-TwoFinger-Flexiv-Distill-Depth",
  "Mjlab-Grasp-TwoFinger-Flexiv-Distill-Rgb",
  "Mjlab-Grasp-TwoFinger-Flexiv-Distill-Rgbd",
]


@pytest.mark.parametrize("task_id", DISTILL_TASKS)
def test_teacher_group_matches_state_task(task_id: str) -> None:
  """The distillation env's teacher observation is the state task's, verbatim."""
  teacher = load_env_cfg(TEACHER_TASK).observations["actor"]
  distilled = load_env_cfg(task_id).observations["actor"]

  assert list(distilled.terms) == list(teacher.terms)
  assert distilled.enable_corruption == teacher.enable_corruption
  for name, term in teacher.terms.items():
    assert distilled.terms[name].func is term.func
    assert distilled.terms[name].params == term.params


@pytest.mark.parametrize("task_id", DISTILL_TASKS)
def test_student_group_has_no_privileged_terms(task_id: str) -> None:
  """The student sees proprioception and the goal height, never the cube."""
  student = load_env_cfg(task_id).observations["student"]

  assert not set(student.terms) & set(_PRIVILEGED_TERMS)
  assert "goal_height" in student.terms, (
    "the commanded lift height varies per episode, so a student without it is "
    "guessing how high to lift"
  )


@pytest.mark.parametrize("task_id", DISTILL_TASKS)
def test_obs_groups_resolve_against_env(task_id: str) -> None:
  """Every group the runner routes into a model actually exists in the env."""
  groups = load_env_cfg(task_id).observations
  obs_groups = load_rl_cfg(task_id).obs_groups

  assert set(obs_groups) == {"student", "teacher"}
  for model, names in obs_groups.items():
    for name in names:
      assert name in groups, f"{task_id}: {model} reads missing group '{name}'"
  assert "camera" not in obs_groups["teacher"], (
    "the teacher is a state policy; handing it an image changes its input dim"
  )
