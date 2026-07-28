"""MDP terms for the Stage-1 compliance (reach-under-disturbance) task."""

from mjlab.tasks.compliance.mdp.curriculum import (
  displacement_curriculum as displacement_curriculum,
)
from mjlab.tasks.compliance.mdp.human import HumanDisturbance as HumanDisturbance
from mjlab.tasks.compliance.mdp.human import HumanDisturbanceCfg as HumanDisturbanceCfg
from mjlab.tasks.compliance.mdp.observations import ee_lin_vel as ee_lin_vel
from mjlab.tasks.compliance.mdp.observations import ee_pos as ee_pos
from mjlab.tasks.compliance.mdp.observations import goal_error as goal_error
from mjlab.tasks.compliance.mdp.observations import (
  joint_total_torque as joint_total_torque,
)
from mjlab.tasks.compliance.mdp.observations import (
  privileged_human as privileged_human,
)
from mjlab.tasks.compliance.mdp.reach_command import ReachCommand as ReachCommand
from mjlab.tasks.compliance.mdp.reach_command import ReachCommandCfg as ReachCommandCfg
from mjlab.tasks.compliance.mdp.rewards import force_penalty as force_penalty
from mjlab.tasks.compliance.mdp.rewards import (
  joint_acc_penalty as joint_acc_penalty,
)
from mjlab.tasks.compliance.mdp.rewards import progress as progress
from mjlab.tasks.compliance.mdp.rewards import push_anisotropy as push_anisotropy
from mjlab.tasks.compliance.mdp.rewards import push_softness as push_softness
from mjlab.tasks.compliance.mdp.rewards import stiffness_penalty as stiffness_penalty
from mjlab.tasks.compliance.mdp.rewards import success_bonus as success_bonus
from mjlab.tasks.compliance.mdp.rewards import time_penalty as time_penalty
from mjlab.tasks.compliance.mdp.rewards import (
  torque_effort_penalty as torque_effort_penalty,
)
from mjlab.tasks.compliance.mdp.terminations import reached_goal as reached_goal
