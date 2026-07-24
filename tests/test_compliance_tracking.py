"""Unit tests for the compliance-tracking teacher's structural invariants.

These pin the parts of the teacher that define the target behaviour and would
otherwise regress silently: a tracking reward reports "success" against whatever
target it is given, so a broken path or a broken freeze law looks like a
well-trained policy right up until you watch a rollout.

All of it is pure tensor math, so no simulator is needed.
"""

import torch

from mjlab.tasks.compliance_tracking.mdp.path import (
  PHASE_APPROACH,
  PHASE_CLOSE,
  PHASE_DESCEND,
  PHASE_LIFT,
  PathSchedule,
  ReferencePath,
)
from mjlab.tasks.compliance_tracking.mdp.perturbation import (
  HumanPerturbation,
  PerturbationCfg,
)
from mjlab.tasks.compliance_tracking.mdp.teacher import freeze_gate

DEVICE = "cpu"


def _path(num_envs: int = 4) -> ReferencePath:
  path = ReferencePath(num_envs, DEVICE, PathSchedule())
  ids = torch.arange(num_envs)
  start = torch.tensor([[0.38, 0.0, 0.24]]).repeat(num_envs, 1)
  grasp = torch.tensor([[0.50, 0.0, 0.02]]).repeat(num_envs, 1)
  path.set_waypoints(ids, start, grasp, standoff=0.12, lift_height=0.10)
  return path


# --- reference path ----------------------------------------------------------


def test_path_hits_its_waypoints() -> None:
  path = _path()
  s_app, s_pre, s_close = path.schedule.breakpoints
  cases = [
    (0.0, path.x_start),
    (s_app, path.x_above),
    (s_pre, path.x_grasp),
    (s_close, path.x_grasp),  # the closure dwell holds the grasp pose
    (1.0, path.x_lift),
  ]
  for s_value, expected in cases:
    s = torch.full((4,), s_value)
    torch.testing.assert_close(path.position(s), expected, atol=1e-5, rtol=0)


def test_path_is_continuous_in_s() -> None:
  """No teleports: a piecewise path with a mis-ordered branch would jump."""
  n = 501
  path = _path(num_envs=n)  # one "env" per sample point, all sharing waypoints
  pts = path.position(torch.linspace(0.0, 1.0, n))
  steps = torch.norm(pts[1:] - pts[:-1], dim=-1)
  # Longest segment is ~0.25 m over ~0.35 of s, i.e. ~1.4 mm per 1/500 step.
  assert float(steps.max()) < 0.005


def test_path_phases_are_ordered() -> None:
  path = _path()
  s_app, s_pre, s_close = path.schedule.breakpoints
  probe = torch.tensor([0.0, s_app - 1e-4, s_app, s_pre, s_close, 1.0])
  phases = path.phase(probe).tolist()
  assert phases == [
    PHASE_APPROACH,
    PHASE_APPROACH,
    PHASE_DESCEND,
    PHASE_CLOSE,
    PHASE_LIFT,
    PHASE_LIFT,
  ]


def test_path_velocity_matches_finite_difference() -> None:
  """``velocity`` must be the true d/dt of ``position``; the admittance law
  integrates against it, so an inconsistent pair injects a phantom force."""
  path = _path()
  s = torch.full((4,), 0.3)
  s_dot = torch.full((4,), path.s_rate)
  dt = 1e-4
  fd = (path.position(s + s_dot * dt) - path.position(s)) / dt
  torch.testing.assert_close(path.velocity(s, s_dot), fd, atol=1e-4, rtol=1e-3)


# --- path-parameter freezing (§1.2) -----------------------------------------


def test_freeze_gate_is_full_rate_inside_the_tube_and_zero_outside() -> None:
  lo, hi = 0.015, 0.04
  gate = freeze_gate(torch.tensor([0.0, lo, hi, 0.5]), lo, hi)
  assert float(gate[0]) == 1.0
  assert float(gate[1]) == 1.0  # still nominal at the onset
  assert float(gate[2]) == 0.0  # fully frozen at d_freeze
  assert float(gate[3]) == 0.0  # and stays frozen beyond it


def test_freeze_gate_is_monotone_and_smooth() -> None:
  """A hard switch would put a step in the target velocity the student tracks."""
  lo, hi = 0.015, 0.04
  d = torch.linspace(0.0, 0.06, 200)
  gate = freeze_gate(d, lo, hi)
  diffs = gate[1:] - gate[:-1]
  assert bool((diffs <= 1e-6).all()), "gate must be non-increasing in deviation"
  # Smoothstep has zero slope at both ends, so the largest step is interior.
  assert float(diffs.abs().max()) < 0.05


# --- perturbation model (§1.4) ----------------------------------------------


def _run_perturbation(cfg: PerturbationCfg, steps: int = 400, num_envs: int = 64):
  torch.manual_seed(0)
  pert = HumanPerturbation(cfg, num_envs, DEVICE)
  pert.reset(torch.arange(num_envs))
  x_att = torch.zeros(num_envs, 3)
  v_att = torch.zeros(num_envs, 3)
  dt = 0.01
  forces, active = [], []
  for i in range(steps):
    s = torch.full((num_envs,), min(1.0, i * dt / 3.4))
    forces.append(pert.update(dt, s, x_att, v_att).clone())
    active.append(pert.active.clone())
  return torch.stack(forces), torch.stack(active)


def test_perturbation_force_is_saturated() -> None:
  cfg = PerturbationCfg(p_no_perturbation=0.0, stiffness_range=(900.0, 900.0))
  forces, _ = _run_perturbation(cfg)
  assert float(torch.norm(forces, dim=-1).max()) <= cfg.force_limit + 1e-4


def test_perturbation_onset_ramps_when_unsaturated() -> None:
  """With a soft hand (saturation never binds) the force must follow the
  min-jerk anchor ramp, i.e. rise gradually rather than switching on.

  Release is deliberately excluded — letting go *is* abrupt, and the smooth
  return transient comes from integrating the reference impedance (§1.3), not
  from ramping the force down.
  """
  cfg = PerturbationCfg(
    p_no_perturbation=0.0,
    stiffness_range=(80.0, 80.0),
    displacement_range=(0.05, 0.05),
  )
  forces, active = _run_perturbation(cfg)
  jumps = torch.norm(forces[1:] - forces[:-1], dim=-1)
  still_engaged = active[1:] & active[:-1]
  assert float(jumps[still_engaged].max()) < 1.0


def test_perturbation_has_no_impact_spike_at_the_hard_corner() -> None:
  """At the aggressive corner (stiff hand, large displacement) the hand *wants*
  ~180 N, so the saturation does most of the limiting. The invariant is that it
  still cannot arrive as a step: a teleported anchor would jump straight to the
  40 N limit within one control period.
  """
  cfg = PerturbationCfg(
    p_no_perturbation=0.0,
    stiffness_range=(900.0, 900.0),
    displacement_range=(0.20, 0.20),
  )
  forces, active = _run_perturbation(cfg)
  jumps = torch.norm(forces[1:] - forces[:-1], dim=-1)
  still_engaged = active[1:] & active[:-1]
  assert float(jumps[still_engaged].max()) < 0.5 * cfg.force_limit


def test_perturbation_release_is_clean() -> None:
  """Force goes to exactly zero on release, with no residual dribble."""
  cfg = PerturbationCfg(p_no_perturbation=0.0)
  forces, active = _run_perturbation(cfg)
  assert float(torch.norm(forces, dim=-1)[~active].max()) == 0.0


def test_some_episodes_have_no_perturbation() -> None:
  cfg = PerturbationCfg(p_no_perturbation=0.25)
  _, active = _run_perturbation(cfg, num_envs=512)
  never = (~active.any(dim=0)).float().mean()
  assert 0.15 < float(never) < 0.40


def test_no_perturbation_setting_is_absolute() -> None:
  cfg = PerturbationCfg(p_no_perturbation=1.0)
  forces, active = _run_perturbation(cfg)
  assert not bool(active.any())
  assert float(forces.abs().max()) == 0.0


def test_perturbation_onsets_are_keyed_to_the_path_parameter() -> None:
  """Events must not fire before their scheduled ``s``; that is what keeps
  "push after the grasp" well-defined even when ``s`` freezes."""
  torch.manual_seed(0)
  cfg = PerturbationCfg(p_no_perturbation=0.0, onset_s_range=(0.8, 0.9))
  num_envs = 64
  pert = HumanPerturbation(cfg, num_envs, DEVICE)
  pert.reset(torch.arange(num_envs))
  x_att, v_att = torch.zeros(num_envs, 3), torch.zeros(num_envs, 3)
  # s pinned below every scheduled onset: nothing may fire, ever.
  for _ in range(500):
    pert.update(0.01, torch.full((num_envs,), 0.5), x_att, v_att)
    assert not bool(pert.active.any())
