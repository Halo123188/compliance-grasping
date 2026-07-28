from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import flexiv_reach_env_cfg
from .rl_cfg import flexiv_reach_ppo_runner_cfg

# E2 main task: GRU, anisotropic K, displacement curriculum.
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv",
  env_cfg=flexiv_reach_env_cfg(),
  play_env_cfg=flexiv_reach_env_cfg(play=True),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# E4 A1: isotropic (scalar) stiffness — is anisotropy the source of the benefit?
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-ScalarK",
  env_cfg=flexiv_reach_env_cfg(ablation="scalar_k"),
  play_env_cfg=flexiv_reach_env_cfg(play=True, ablation="scalar_k"),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# E4 A2: MLP (single-frame) instead of GRU — is the τ history necessary?
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-MLP",
  env_cfg=flexiv_reach_env_cfg(),
  play_env_cfg=flexiv_reach_env_cfg(play=True),
  rl_cfg=flexiv_reach_ppo_runner_cfg(mlp=True),
  runner_cls=ManipulationOnPolicyRunner,
)

# E4 A3: no curriculum, train directly on the level-3 distribution.
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-NoCurriculum",
  env_cfg=flexiv_reach_env_cfg(ablation="no_curriculum"),
  play_env_cfg=flexiv_reach_env_cfg(play=True, ablation="no_curriculum"),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Item 6: disturbance shape variety (hold / shake / drag).
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-Shapes",
  env_cfg=flexiv_reach_env_cfg(ablation="shapes"),
  play_env_cfg=flexiv_reach_env_cfg(play=True, ablation="shapes"),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Item 5: grip moment (r x F) + policy-controlled rotational stiffness.
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-Torque",
  env_cfg=flexiv_reach_env_cfg(ablation="torque"),
  play_env_cfg=flexiv_reach_env_cfg(play=True, ablation="torque"),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# e14: items 6 + 5 combined on the corrected (harder-only) randomisation.
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-Full",
  env_cfg=flexiv_reach_env_cfg(ablation="full"),
  play_env_cfg=flexiv_reach_env_cfg(play=True, ablation="full"),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Bare Rizon 4S (no UMI gripper) on the e15 "full" setup: same disturbance
# shapes, grip moment and corrected randomisation, arm only.
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-Bare-Full",
  env_cfg=flexiv_reach_env_cfg(ablation="full", bare=True),
  play_env_cfg=flexiv_reach_env_cfg(play=True, ablation="full", bare=True),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Direct joint-torque control (vs modulating an impedance law).  The policy
# outputs one torque per arm joint; compliance is emergent (driven by the force
# penalty) rather than a commanded stiffness.  Two variants differ only in
# whether gravity is compensated for the policy.
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorquePure",
  env_cfg=flexiv_reach_env_cfg(torque="pure"),
  play_env_cfg=flexiv_reach_env_cfg(play=True, torque="pure"),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueResidual",
  env_cfg=flexiv_reach_env_cfg(torque="residual"),
  play_env_cfg=flexiv_reach_env_cfg(play=True, torque="residual"),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Curriculum phase A for the residual-torque variant: disturbance off, learn the
# base reach-and-hold first, then resume with the push on (see train sbatch).
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueResidual-Warmup",
  env_cfg=flexiv_reach_env_cfg(torque="residual", warmup=True),
  play_env_cfg=flexiv_reach_env_cfg(play=True, torque="residual", warmup=True),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Incremental (rate) residual torque: the policy emits a bounded torque increment
# per step (<=15% of the budget) instead of an absolute torque, so torque cannot
# spike and cannot flail at reset.  Same two-phase curriculum as above.
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueRate-Warmup",
  env_cfg=flexiv_reach_env_cfg(torque="residual", warmup=True, torque_delta_frac=0.15),
  play_env_cfg=flexiv_reach_env_cfg(
    play=True, torque="residual", warmup=True, torque_delta_frac=0.15
  ),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueRate",
  env_cfg=flexiv_reach_env_cfg(torque="residual", torque_delta_frac=0.15),
  play_env_cfg=flexiv_reach_env_cfg(
    play=True, torque="residual", torque_delta_frac=0.15
  ),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Smooth residual torque: absolute torque (fast response kept) but the reset-time
# whip is suppressed by a joint-velocity penalty and over-torquing by a stronger
# effort penalty -- targeting the measured gaps to e15 (100x early joint speed,
# 50% vs 12% torque budget, 18% vs 0% saturation).  Two-phase curriculum.
# vel_penalty is deliberately small: a squared penalty of -0.003 froze even the
# base reach (progress went negative), because it taxes the reach motion as much
# as the reset whip.  -0.0005 bites the whip (v~12, v^2 huge) while leaving the
# ~1 rad/s reach motion almost untouched.
_SMOOTH = dict(
  torque="residual", torque_vel_penalty=-0.0005, torque_effort_weight=-0.005
)
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueSmooth-Warmup",
  env_cfg=flexiv_reach_env_cfg(warmup=True, **_SMOOTH),
  play_env_cfg=flexiv_reach_env_cfg(play=True, warmup=True, **_SMOOTH),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueSmooth",
  env_cfg=flexiv_reach_env_cfg(**_SMOOTH),
  play_env_cfg=flexiv_reach_env_cfg(play=True, **_SMOOTH),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)


# Acceleration-penalty smooth torque on the BARE arm (no gripper).  The reset
# whip is a burst of joint acceleration; a clamped joint-acc penalty targets it
# surgically without taxing the steady yield/return motion (unlike the velocity
# penalty).  Two-phase curriculum, bare Rizon 4S.
# -2e-5 froze the phase-A reach (pos plateaued at 0.25, acc penalty -3.86/ep vs
# progress +0.09 -- ~40x the task signal, so standing still beat reaching).  A
# no-penalty bare probe reached pos=0.002, confirming the geometry is fine and the
# penalty was the culprit.  -5e-6 brings the acc penalty to ~-1/ep, comparable to
# progress, so it bites the reset whip without taxing the reach.
_ACCBARE = dict(bare=True, torque="residual", torque_acc_penalty=-5e-6)
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueAccBare-Warmup",
  env_cfg=flexiv_reach_env_cfg(warmup=True, **_ACCBARE),
  play_env_cfg=flexiv_reach_env_cfg(play=True, warmup=True, **_ACCBARE),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueAccBare",
  env_cfg=flexiv_reach_env_cfg(**_ACCBARE),
  play_env_cfg=flexiv_reach_env_cfg(play=True, **_ACCBARE),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Velocity-penalty smooth torque on the BARE arm.  The acceleration penalty
# (accbare) failed to suppress the reset whip (earlyQV went UP to 25.95, worst of
# all variants) because its clamp removes the gradient exactly at the largest
# whips.  The velocity penalty directly penalises the whipped quantity (v^2,
# unbounded) and DID cut earlyQV 12.3 -> 3.76 on the gripper arm; this run ports
# that proven lever to the bare arm for a clean bare-arm smooth baseline.
_SMOOTHBARE = dict(
  bare=True, torque="residual", torque_vel_penalty=-0.0005, torque_effort_weight=-0.005
)
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueSmoothBare-Warmup",
  env_cfg=flexiv_reach_env_cfg(warmup=True, **_SMOOTHBARE),
  play_env_cfg=flexiv_reach_env_cfg(play=True, warmup=True, **_SMOOTHBARE),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueSmoothBare",
  env_cfg=flexiv_reach_env_cfg(**_SMOOTHBARE),
  play_env_cfg=flexiv_reach_env_cfg(play=True, **_SMOOTHBARE),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# Isolation probe: bare arm + residual torque with NO smoothness penalty.  Used to
# decide whether the accbare phase-A reach failure (pos plateaued at 0.25) is
# caused by the acc penalty or by the bare EE/grasp geometry itself.
_BAREPROBE = dict(bare=True, torque="residual")
register_mjlab_task(
  task_id="Mjlab-Compliance-Reach-Flexiv-TorqueBareProbe-Warmup",
  env_cfg=flexiv_reach_env_cfg(warmup=True, **_BAREPROBE),
  play_env_cfg=flexiv_reach_env_cfg(play=True, warmup=True, **_BAREPROBE),
  rl_cfg=flexiv_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
