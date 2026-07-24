"""Task registrations for the Flexiv teacher-student compliant-grasping stages."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import flexiv_tracking_env_cfg
from .rl_cfg import flexiv_tracking_ppo_runner_cfg

# Stage A: pre-grasp only. Reach + perturbations + s-freezing + admittance
# target; no fingers, no object, no weld.
register_mjlab_task(
  task_id="Mjlab-ComplianceTracking-StageA-Flexiv",
  env_cfg=flexiv_tracking_env_cfg(stage="A"),
  play_env_cfg=flexiv_tracking_env_cfg(stage="A", play=True),
  rl_cfg=flexiv_tracking_ppo_runner_cfg("compliance_tracking_flexiv_a"),
  runner_cls=ManipulationOnPolicyRunner,
)

# Stage B: + scripted grasp, object, weld, post-grasp perturbations.
register_mjlab_task(
  task_id="Mjlab-ComplianceTracking-StageB-Flexiv",
  env_cfg=flexiv_tracking_env_cfg(stage="B"),
  play_env_cfg=flexiv_tracking_env_cfg(stage="B", play=True),
  rl_cfg=flexiv_tracking_ppo_runner_cfg("compliance_tracking_flexiv_b"),
  runner_cls=ManipulationOnPolicyRunner,
)

# Stage C: + domain randomization for transfer.
register_mjlab_task(
  task_id="Mjlab-ComplianceTracking-StageC-Flexiv",
  env_cfg=flexiv_tracking_env_cfg(stage="C"),
  play_env_cfg=flexiv_tracking_env_cfg(stage="C", play=True),
  rl_cfg=flexiv_tracking_ppo_runner_cfg("compliance_tracking_flexiv_c"),
  runner_cls=ManipulationOnPolicyRunner,
)

# Stiffness-supervised variants (§reward departure): the teacher additionally
# emits a target impedance and the reward tracks it, plus a widened freeze band.
# These break pure-tracking on purpose to buy directional compliance (K∥ < K⊥)
# and to unstick the lift; see mdp.stiffness_tracking and the task README.
register_mjlab_task(
  task_id="Mjlab-ComplianceTracking-StageA-Sup-Flexiv",
  env_cfg=flexiv_tracking_env_cfg(stage="A", supervise_stiffness=True),
  play_env_cfg=flexiv_tracking_env_cfg(stage="A", play=True, supervise_stiffness=True),
  rl_cfg=flexiv_tracking_ppo_runner_cfg("compliance_tracking_flexiv_a_sup"),
  runner_cls=ManipulationOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-ComplianceTracking-StageB-Sup-Flexiv",
  env_cfg=flexiv_tracking_env_cfg(stage="B", supervise_stiffness=True),
  play_env_cfg=flexiv_tracking_env_cfg(stage="B", play=True, supervise_stiffness=True),
  rl_cfg=flexiv_tracking_ppo_runner_cfg("compliance_tracking_flexiv_b_sup"),
  runner_cls=ManipulationOnPolicyRunner,
)

# DIAGNOSTIC: anisotropic supervision + the privileged pull direction fed to the
# actor. Isolates whether observability (not the diagonal-K action) blocks
# directional compliance. K∥/K⊥ < 1 here => observability is the wall.
register_mjlab_task(
  task_id="Mjlab-ComplianceTracking-StageA-SupDiag-Flexiv",
  env_cfg=flexiv_tracking_env_cfg(
    stage="A", supervise_stiffness=True, actor_sees_pull_dir=True
  ),
  play_env_cfg=flexiv_tracking_env_cfg(
    stage="A", play=True, supervise_stiffness=True, actor_sees_pull_dir=True
  ),
  rl_cfg=flexiv_tracking_ppo_runner_cfg("compliance_tracking_flexiv_a_supdiag"),
  runner_cls=ManipulationOnPolicyRunner,
)

# EXP-2: direct joint-torque action. The policy emits torque and tracks the
# teacher's compliant torque target (full rotatable anisotropic impedance about
# x_ref), removing the diagonal-K representation wall. Two variants: proprio-only,
# and +privileged pull direction (removes the observability wall too, so the two
# effects can be separated).
register_mjlab_task(
  task_id="Mjlab-ComplianceTracking-StageA-Torque-Flexiv",
  env_cfg=flexiv_tracking_env_cfg(stage="A", torque_action=True),
  play_env_cfg=flexiv_tracking_env_cfg(stage="A", play=True, torque_action=True),
  rl_cfg=flexiv_tracking_ppo_runner_cfg("compliance_tracking_flexiv_a_torque"),
  runner_cls=ManipulationOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-ComplianceTracking-StageA-TorqueDiag-Flexiv",
  env_cfg=flexiv_tracking_env_cfg(
    stage="A", torque_action=True, actor_sees_pull_dir=True
  ),
  play_env_cfg=flexiv_tracking_env_cfg(
    stage="A", play=True, torque_action=True, actor_sees_pull_dir=True
  ),
  rl_cfg=flexiv_tracking_ppo_runner_cfg("compliance_tracking_flexiv_a_torquediag"),
  runner_cls=ManipulationOnPolicyRunner,
)
