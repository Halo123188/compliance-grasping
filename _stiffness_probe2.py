"""Two-direction steady-state stiffness probe: k_par AND k_perp.

The one-direction probe (_stiffness_probe.py) only pushes along u and reads
k_par = |F| / yield_along_u. That cannot tell "directional" (soft along the push,
stiff across) from "isotropically soft" -- with a pure-along-u force the yield is
along u in either case. To measure the cross stiffness you must apply a force
*perpendicular* to the soft axis.

So on top of the main saturating push along u (which defines the axis the policy
senses and softens along), we inject a smaller constant test force F_test along a
fixed per-env direction w perpendicular to u, and read
    k_perp = F_test / yield_along_w.
The test force is small relative to the main push, so the sensed direction stays
~u (the resultant tilts by atan(F_test/|F|) ~= 14 deg at 10/40 N); the measured w
is then ~76 deg off the soft axis, close enough to read the cross stiffness.

k_par / k_perp < 1 is the directional-compliance signature: genuinely soft along
the push and stiff across it. k_par ~= k_perp means isotropically soft (or stiff).

  MUJOCO_GL=egl uv run python _stiffness_probe2.py <task> <ckpt_dir>
"""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks.compliance_tracking.config.flexiv  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.compliance_tracking.mdp.teacher import TeacherCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

task, ckpt_dir = sys.argv[1], sys.argv[2]
DEV = "cuda:0"
N = 256
F_TEST = 10.0  # perpendicular test force (N); small vs the 40 N main push

p = Path(ckpt_dir)
ckpt = sorted(p.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))[-1]

torch.manual_seed(0)
cfg = load_env_cfg(task)
cfg.scene.num_envs = N
cfg.episode_length_s = 8.0
cfg.observations["actor"].enable_corruption = True
pert = cfg.commands["teacher"].perturbation
pert.p_no_perturbation = 0.0
pert.num_events_range = (1, 1)
pert.onset_s_range = (0.08, 0.08)
pert.displacement_range = (0.20, 0.20)
pert.hold_time_range = (6.0, 6.0)
pert.ramp_time_range = (0.5, 0.5)
pert.stiffness_range = (800.0, 800.0)

env = ManagerBasedRlEnv(cfg, device=DEV)
teacher = env.command_manager.get_term("teacher")
assert isinstance(teacher, TeacherCommand)


def _wperp(u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
  """Unit vector perpendicular to u, in the plane spanned by u and the fixed r."""
  w = r - (r * u).sum(-1, keepdim=True) * u
  return w / w.norm(dim=-1, keepdim=True).clamp(min=1e-6)


# Fixed per-env random reference so w is stable across the episode.
_r = torch.randn(N, 3, device=DEV)
_orig_update = teacher.perturbation.update


def _patched_update(dt, s, x_att, v_att):
  # Main saturating push along u (unchanged; also sets perturbation._force, so the
  # k_par read below still uses the pure along-u force).
  f = _orig_update(dt, s, x_att, v_att)
  u = teacher.perturbation.direction
  active = teacher.perturbation.active.float().unsqueeze(-1)
  return f + F_TEST * _wperp(u, _r) * active


teacher.perturbation.update = _patched_update  # shadow the bound method

agent_cfg = load_rl_cfg(task)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(task)(wrapped, asdict(agent_cfg), device=DEV)
runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=DEV)
policy = runner.get_inference_policy(device=DEV)

obs = wrapped.get_observations()
kpar_a, kperp_a, ypar_a, yperp_a, steady_a = [], [], [], [], []
for t in range(700):
  with torch.no_grad():
    action = policy(obs)
  obs, _, dones, _ = wrapped.step(action)
  if hasattr(policy, "reset"):
    policy.reset(dones)
  f = teacher.perturbation.force  # along-u only (set inside _orig_update)
  fmag = torch.norm(f, dim=-1)
  u = teacher.perturbation.direction
  w = _wperp(u, _r)
  dev = teacher.ee_pos_w() - teacher.x_ref()
  y_par = (dev * u).sum(dim=-1)
  y_perp = (dev * w).sum(dim=-1)  # yield along the test force
  steady = teacher.perturbation.active & (fmag > 35.0) & (y_par > 0.002)
  kpar_a.append(fmag / y_par.clamp(min=0.002))
  kperp_a.append(F_TEST / y_perp.clamp(min=0.001))
  ypar_a.append(y_par)
  yperp_a.append(y_perp)
  steady_a.append(steady.float())

S = torch.stack(steady_a).bool()
n = S.sum().clamp(min=1)
mean = lambda x: float((torch.stack(x) * S).sum() / n)  # noqa: E731
# Ratio of means (not mean of ratios, which is Jensen-biased upward).
kpar = mean(kpar_a)
# k_perp from the mean perpendicular yield, differencing out per-step noise.
yperp_mean = mean(yperp_a)
kperp = F_TEST / max(yperp_mean, 1e-4)
print(f"\n=== {task} :: {ckpt.name} ===")
print(f"  k_par  (along push)   {kpar:8.1f} N/m   yield {mean(ypar_a) * 100:5.2f} cm")
print(f"  k_perp (across push)  {kperp:8.1f} N/m   yield {yperp_mean * 100:5.2f} cm")
print(f"  k_par / k_perp        {kpar / max(kperp, 1e-6):8.2f}     (<1 = directional)")
print(f"  steady samples/env    {float(S.sum()) / N:8.1f}")
env.close()
