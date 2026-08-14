"""Tests for the rate-limited / low-passed joint position action.

The property worth testing is the one the deployment relies on: the published
command's rate is bounded in rad/s, at whatever rate ``apply_actions`` happens
to be called. The bug this guards against is the substep one -- ``apply_actions``
runs once per physics substep, four times per control step at decimation=4, so a
limiter written in per-step units is silently 4x too fast and nothing raises.
"""

from pathlib import Path
from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import BuiltinMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.manipulation.mdp.actions import SmoothedJointPositionActionCfg

PHYSICS_DT = 0.005
# `apply_actions` runs once per physics substep, `process_actions` once per
# control step, and that ratio IS the bug this file guards against -- so the
# fake env has to carry both clocks, consistently. `step_dt` is read by
# `step_allowance` and by the default `incremental_scale`.
DECIMATION = 4
STEP_DT = PHYSICS_DT * DECIMATION
NUM_ENVS = 4


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def entity(device):
  path = Path(__file__).parent / "fixtures" / "fixed_base_articulated.xml"
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(path)),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=("joint.*",),
          transmission_type=TransmissionType.JOINT,
          effort_limit=10.0,
        ),
      )
    ),
  )
  ent = Entity(cfg)
  model = ent.compile()
  sim = Simulation(num_envs=NUM_ENVS, cfg=SimulationCfg(), model=model, device=device)
  ent.initialize(model, sim.model, sim.data, device)
  return ent


def _build(entity, device, **kwargs):
  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = NUM_ENVS
  env.device = device
  env.physics_dt = PHYSICS_DT
  env.step_dt = STEP_DT
  env.scene = {"robot": entity}
  cfg = SmoothedJointPositionActionCfg(
    entity_name="robot", actuator_names=("joint.*",), **kwargs
  )
  term = cfg.build(env)
  term.reset()
  return term


def test_slew_limit_bounds_the_per_substep_command_change(entity, device):
  """No substep may move the command more than v_max * physics_dt."""
  term = _build(entity, device, rate_limit=2.0)
  # Far larger than the limit can cover in the substeps below, so every one of
  # them is clamped rather than merely close.
  term.process_actions(torch.full((NUM_ENVS, term.action_dim), 50.0, device=device))

  prev = term._cmd.clone()
  for _ in range(20):
    term.apply_actions()
    delta = (term._cmd - prev).abs().max().item()
    assert delta <= 2.0 * PHYSICS_DT + 1e-6
    prev = term._cmd.clone()


def test_per_joint_limits_resolve_by_name(entity, device):
  """A dict of name patterns gives each joint its own cap."""
  names = list(_build(entity, device, rate_limit=1.0).target_names)
  slow, fast = names[0], names[-1]
  term = _build(entity, device, rate_limit={slow: 0.5, fast: 4.0})
  term.process_actions(torch.full((NUM_ENVS, term.action_dim), 50.0, device=device))

  start = term._cmd.clone()
  term.apply_actions()
  moved = (term._cmd - start).abs()
  assert moved[:, 0].max().item() == pytest.approx(0.5 * PHYSICS_DT, rel=1e-4)
  assert moved[:, -1].max().item() == pytest.approx(4.0 * PHYSICS_DT, rel=1e-4)


def test_unlimited_joints_pass_straight_through(entity, device):
  """Joints the dict does not name keep the stock memoryless behaviour."""
  term = _build(entity, device, rate_limit={"joint1": 0.5})
  target = torch.full((NUM_ENVS, term.action_dim), 3.0, device=device)
  term.process_actions(target)
  term.apply_actions()
  assert torch.allclose(term._cmd[:, 1:], term._processed_actions[:, 1:])


def test_ema_moves_a_fixed_fraction_of_the_gap(entity, device):
  """alpha = exp(-dt/tau), so one substep closes 1 - alpha of the distance."""
  tau = 0.05
  term = _build(entity, device, ema_tau=tau)
  target = torch.full((NUM_ENVS, term.action_dim), 1.0, device=device)
  term.process_actions(target)

  start = term._cmd.clone()
  term.apply_actions()
  expected = 1.0 - torch.exp(torch.tensor(-PHYSICS_DT / tau))
  gap = term._processed_actions - start
  assert torch.allclose(term._cmd - start, gap * expected, atol=1e-6)


def test_reset_seeds_the_command_from_measured_joints(entity, device):
  """Not from the default pose: the reset events jitter the arm off it."""
  term = _build(entity, device, rate_limit=1.0)
  term._cmd.fill_(99.0)
  term.reset(env_ids=torch.tensor([0, 2], device=device))

  measured = entity.data.joint_pos[:, term.target_ids]
  assert torch.allclose(term._cmd[[0, 2]], measured[[0, 2]])
  assert torch.all(term._cmd[[1, 3]] == 99.0)


def test_rate_scale_multiplies_the_limit(entity, device):
  """What the curriculum drives, so the schedule is one number."""
  term = _build(entity, device, rate_limit=1.0)
  term.process_actions(torch.full((NUM_ENVS, term.action_dim), 50.0, device=device))
  term.rate_scale = 3.0

  start = term._cmd.clone()
  term.apply_actions()
  moved = (term._cmd - start).abs().max().item()
  assert moved == pytest.approx(3.0 * PHYSICS_DT, rel=1e-4)


def test_saturation_counts_overdrive_in_units_of_the_cap(entity, device):
  """N means "asking for N+1 times what the cap can deliver"."""
  term = _build(entity, device, rate_limit=1.0)
  step = 1.0 * PHYSICS_DT
  # Place the request exactly 4 caps beyond the command, so the expected
  # overdrive is a number rather than an inequality.
  term.process_actions(torch.zeros((NUM_ENVS, term.action_dim), device=device))
  term._processed_actions = term._cmd + 4.0 * step
  term.apply_actions()

  assert term.saturation.max().item() == pytest.approx(3.0, rel=1e-4)


def test_saturation_is_zero_while_the_limiter_keeps_up(entity, device):
  """A request the cap can follow is not overdrive, however small the cap."""
  term = _build(entity, device, rate_limit=1.0)
  term.process_actions(torch.zeros((NUM_ENVS, term.action_dim), device=device))
  term._processed_actions = term._cmd + 0.5 * (1.0 * PHYSICS_DT)
  term.apply_actions()

  assert term.saturation.max().item() == 0.0


def test_saturation_averages_over_substeps_and_resets_each_control_step(entity, device):
  """The reward reads once per control step; `apply_actions` runs once per substep.

  Without the reset in `process_actions` the penalty would grow without bound
  over an episode instead of reporting the current step.
  """
  term = _build(entity, device, rate_limit=1.0)
  target = torch.full((NUM_ENVS, term.action_dim), 50.0, device=device)
  term.process_actions(target)
  for _ in range(4):
    term.apply_actions()
  first = term.saturation.max().item()
  assert first > 0.0

  term.process_actions(target)
  assert term.saturation.max().item() == 0.0
  term.apply_actions()
  # One substep of a still-saturated request, so the mean is a mean of one and
  # is on the same scale as the four-substep average above, not four times it.
  assert term.saturation.max().item() == pytest.approx(first, rel=0.05)


def test_uncapped_joints_never_report_saturation(entity, device):
  """`step` is infinite there, so the ratio has to come out 0, not NaN."""
  term = _build(entity, device, rate_limit={"joint1": 0.5})
  term.process_actions(torch.full((NUM_ENVS, term.action_dim), 50.0, device=device))
  term.apply_actions()

  assert torch.all(term.saturation[:, 1:] == 0.0)
  assert term.saturation[:, 0].min().item() > 0.0
