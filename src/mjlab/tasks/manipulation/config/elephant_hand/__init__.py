from mjlab.tasks.manipulation.config.elephant_hand.env_cfgs import (
  elephant_hand_pick_place_env_cfg,
)
from mjlab.tasks.manipulation.config.elephant_hand.rl_cfg import (
  elephant_hand_ppo_runner_cfg,
)
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Pick-Place-ElephantHand",
  env_cfg=elephant_hand_pick_place_env_cfg(),
  play_env_cfg=elephant_hand_pick_place_env_cfg(play=True),
  rl_cfg=elephant_hand_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
