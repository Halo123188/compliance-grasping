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
