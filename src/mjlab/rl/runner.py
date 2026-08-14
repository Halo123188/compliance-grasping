import inspect
import os
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl.utils import resolve_callable

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

# Fields declared on the shared `RslRlPpoAlgorithmCfg` that only a PPO SUBCLASS
# reads; see their docstrings in mjlab.rl.config. Stock `PPO` raises on them.
#
# They live on the shared cfg because every PPO-family run here is configured
# through one dataclass, and they are filtered HERE -- in the shared runner --
# rather than in the one task that introduced them, because every task inherits
# the cfg and so every task inherits the crash. Filtering in
# `ManipulationOnPolicyRunner` alone left the velocity and tracking runners
# dying on construction with
# ``unexpected keyword argument 'critic_warmup_iters'``.
_SUBCLASS_ONLY_ALGORITHM_FIELDS = ("critic_warmup_iters", "std_max")


def drop_unsupported_algorithm_fields(train_cfg: dict) -> dict:
  """Drop algorithm fields the algorithm cannot take, IN PLACE.

  The filter is against the algorithm class's OWN signature, so a subclass that
  does accept these fields still receives them -- dropping them from the cfg
  instead would silently disable the feature for the runs that want it.

  In place, and returning the same object, on purpose. rsl_rl mutates
  ``train_cfg["algorithm"]`` itself (``resolve_symmetry_config`` injects
  non-serializable objects into it, which is why ``train.py`` dumps the config
  files BEFORE constructing the runner -- see mjlab issue #764). Handing it a
  filtered COPY would quietly break that contract: the caller's dict would stop
  being the one the runner works on. Nothing is lost from the record, because
  the dump already happened by the time this runs.
  """
  alg_cfg = train_cfg.get("algorithm", {})
  # `class_name` is a STRING in the cfg; resolve it the way rsl_rl's own runner
  # does rather than guessing, or the filter silently never fires and the
  # original TypeError comes back.
  name = alg_cfg.get("class_name")
  if not isinstance(name, str):
    return train_cfg
  try:
    cls = resolve_callable(name)
  except Exception:
    return train_cfg
  if not isinstance(cls, type):
    return train_cfg

  accepted = inspect.signature(cls.__init__).parameters
  if any(p.kind is p.VAR_KEYWORD for p in accepted.values()):
    return train_cfg

  # ONLY the subclass-only fields. A blanket "drop anything __init__ does not
  # take" also removes `share_cnn_encoders`, which rsl_rl's
  # `construct_algorithm` pops for ITSELF before calling __init__ -- so
  # filtering that here would silently stop the actor and critic sharing a CNN
  # encoder, with nothing to show for it but a worse run.
  dropped = [
    k for k in _SUBCLASS_ONLY_ALGORITHM_FIELDS if k in alg_cfg and k not in accepted
  ]
  if not dropped:
    return train_cfg
  print(
    f"[runner] {cls.__name__} does not accept {dropped}; ignoring."
    " These are PPO-subclass-only settings on the shared PPO cfg."
  )
  for key in dropped:
    del alg_cfg[key]
  return train_cfg


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  MODEL_KEYS: tuple[str, ...] = ("actor", "critic")
  """Keys in ``train_cfg`` that hold a per-model config dict."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in self.MODEL_KEYS:
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    train_cfg = drop_unsupported_algorithm_fields(train_cfg)
    super().__init__(env, train_cfg, log_dir, device)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore common_step_counter to preserve curricula state.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos


class MjlabDistillationRunner(MjlabOnPolicyRunner, DistillationRunner):
  """DAgger runner: the student acts, a frozen privileged teacher labels.

  Shares MjlabOnPolicyRunner's checkpoint handling (environment state,
  legacy-format migration, gated W&B upload) and only swaps which config keys
  hold model definitions. ``learn()`` comes from RSL-RL's DistillationRunner,
  which refuses to start until teacher weights have actually been loaded.

  Load the teacher with an explicit ``load_cfg``::

    runner.load(teacher_ckpt, load_cfg={"teacher": True, "iteration": False})

  RSL-RL's ``Distillation.load`` reads the teacher's weights out of a PPO
  checkpoint's ``actor_state_dict`` when no ``teacher_state_dict`` is present,
  so a plain PPO run is a valid teacher.
  """

  MODEL_KEYS = ("student", "teacher")
