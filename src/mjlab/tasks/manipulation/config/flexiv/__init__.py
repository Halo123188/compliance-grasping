from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import flexiv_pick_place_env_cfg
from .rl_cfg import flexiv_pick_place_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Pick-Place-Flexiv",
  env_cfg=flexiv_pick_place_env_cfg(),
  play_env_cfg=flexiv_pick_place_env_cfg(play=True),
  rl_cfg=flexiv_pick_place_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
