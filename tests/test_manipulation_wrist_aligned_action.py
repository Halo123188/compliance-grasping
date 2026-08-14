"""The wrist-aligned action solves the jaw's yaw instead of leaving it to PPO.

The property that matters is the FOLD. A square cube under a jaw that grips the
same way at 0 and at 180 degrees means the yaw error is only defined modulo
``symmetry``, so the solver must take the short way round. Without the fold the
wrist would chase up to a half-turn it never needed -- which is the whole reason
a closed-form solver replaced letting PPO discover the alignment.

The tests drive the real ``_yaw_error`` against a real scene: two sites stand in
for the finger pads, and the cube's root quaternion is written to sim so the
error is a genuine jaw-minus-object angle rather than a reimplementation of the
formula under test.

They also pin that the cfg is read at construction. The class unpacks
``object_name``/``symmetry``/``gain`` into attributes rather than reading
``self.cfg`` per step -- which is also what keeps the attribute from narrowing
``BaseAction.cfg`` -- so something has to confirm the values arrive.
"""

import math
from unittest.mock import Mock

import mujoco
import pytest
from conftest import get_test_device

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.manipulation.mdp.actions import (
  WristAlignedJointPositionAction,
  WristAlignedJointPositionActionCfg,
)

NUM_ENVS = 2

# A wrist that yaws, and two pad sites straddling it on the x axis so the jaw's
# closing axis is +x at q=0 and the jaw yaw is exactly the wrist angle.
ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="wrist" type="hinge" axis="0 0 1"/>
      <joint name="lift" type="slide" axis="0 0 1"/>
      <geom name="base_geom" type="box" size="0.05 0.05 0.02" mass="1.0"/>
      <site name="pad_left" pos="-0.03 0 0"/>
      <site name="pad_right" pos="0.03 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""

CUBE_XML = """
<mujoco>
  <worldbody>
    <body name="cube" pos="0 0 0.1">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def scene(device):
  robot = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=("wrist", "lift"),
          transmission_type=TransmissionType.JOINT,
          effort_limit=10.0,
        ),
      )
    ),
  )
  cube = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(CUBE_XML))
  scene_cfg = SceneCfg(
    num_envs=NUM_ENVS, env_spacing=5.0, entities={"robot": robot, "cube": cube}
  )
  sc = Scene(scene_cfg, device)
  model = sc.compile()
  sim = Simulation(
    num_envs=NUM_ENVS, cfg=SimulationCfg(njmax=40), model=model, device=device
  )
  sc.initialize(sim.mj_model, sim.model, sim.data)
  sim.forward()
  return sc, sim


def _build(scene_sim, device, **kwargs) -> WristAlignedJointPositionAction:
  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = NUM_ENVS
  env.device = device
  env.physics_dt = 0.005
  env.step_dt = 0.02
  scene, _ = scene_sim
  env.scene = {"robot": scene["robot"], "cube": scene["cube"]}
  params: dict = dict(
    entity_name="robot",
    actuator_names=("wrist", "lift"),
    object_name="cube",
    pad_sites=("pad_left", "pad_right"),
    wrist_joint="wrist",
  )
  params.update(kwargs)
  return WristAlignedJointPositionActionCfg(**params).build(env)


def _set_cube_yaw(scene_sim, device, yaw: float) -> None:
  """Rotate the cube about z, leaving it where it is.

  `sim.forward()` after the write is not optional: `root_link_quat_w` is derived
  state, so without it every test reads the pose from before the rotation and
  the yaw error comes out 0 no matter what was asked for.
  """
  scene, sim = scene_sim
  cube = scene["cube"]
  pose = cube.data.root_link_pose_w.clone()
  half = yaw / 2.0
  pose[:, 3] = math.cos(half)
  pose[:, 4] = 0.0
  pose[:, 5] = 0.0
  pose[:, 6] = math.sin(half)
  cube.write_root_link_pose_to_sim(pose)
  sim.forward()


def test_cfg_values_are_unpacked_at_construction(scene, device):
  """The class reads these once; a typo would surface only as a bad grasp."""
  term = _build(scene, device, symmetry=math.pi / 3, gain=0.7)
  assert term._object_name == "cube"
  assert term._symmetry == pytest.approx(math.pi / 3)
  assert term._gain == pytest.approx(0.7)


def test_no_cfg_attribute_is_narrowed():
  """Narrowing `cfg` is what made pyright flag the BaseAction override."""
  assert "cfg" not in WristAlignedJointPositionAction.__annotations__


@pytest.mark.parametrize(
  "cube_yaw, expected",
  [
    (0.0, 0.0),
    (0.3, -0.3),
    (math.pi / 2, 0.0),  # exactly one symmetry step: the jaw already fits
    (math.pi / 2 + 0.2, -0.2),
    (math.pi, 0.0),  # two steps, back where it started
    (-math.pi / 2 - 0.2, 0.2),
  ],
)
def test_yaw_error_takes_the_short_way_round(scene, device, cube_yaw, expected):
  sym = math.pi / 2
  term = _build(scene, device, symmetry=sym)
  _set_cube_yaw(scene, device, cube_yaw)
  err = term._yaw_error()
  assert float(err[0]) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("cube_yaw", [0.0, 0.4, 1.1, 2.0, -1.7, 3.0])
def test_yaw_error_never_leaves_the_symmetry_window(scene, device, cube_yaw):
  """The bound is the point: a bigger error means the wrist chases a turn."""
  sym = math.pi / 2
  term = _build(scene, device, symmetry=sym)
  _set_cube_yaw(scene, device, cube_yaw)
  assert float(term._yaw_error().abs().max()) <= sym / 2 + 1e-5


def test_a_wider_symmetry_widens_the_window(scene, device):
  """A jaw with no 90-degree symmetry has to be allowed the bigger correction."""
  _set_cube_yaw(scene, device, 1.2)
  narrow = _build(scene, device, symmetry=math.pi / 2)._yaw_error()
  wide = _build(scene, device, symmetry=2 * math.pi)._yaw_error()
  assert float(narrow.abs().max()) < float(wide.abs().max())
  assert float(wide[0]) == pytest.approx(-1.2, abs=1e-5)


def test_the_wrist_column_is_the_named_joint(scene, device):
  term = _build(scene, device)
  assert term.target_names[term._wrist_col] == "wrist"


def test_an_unknown_wrist_joint_is_rejected(scene, device):
  with pytest.raises(AssertionError, match="not in action dims"):
    _build(scene, device, wrist_joint="no_such_joint")
