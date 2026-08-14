import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabDistillationRunner, MjlabOnPolicyRunner


class _OnnxExportOnSaveMixin:
  """Export the policy to ONNX (with scene metadata) on every checkpoint save."""

  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)  # type: ignore[misc]
    policy_dir, filename, onnx_path = self._get_export_paths(path)  # type: ignore[attr-defined]
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)  # type: ignore[attr-defined]
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type == "wandb" and wandb.run  # type: ignore[attr-defined]
        else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:  # type: ignore[attr-defined]
        wandb.save(
          str(onnx_path),
          base_path=str(policy_dir),
        )
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")


class ManipulationOnPolicyRunner(_OnnxExportOnSaveMixin, MjlabOnPolicyRunner):
  """PPO runner that exports ONNX on every checkpoint save.

  The filtering of ``FinetunePPO``-only algorithm fields used to live here; it
  is in ``MjlabOnPolicyRunner`` now, because the fields sit on the shared
  ``RslRlPpoAlgorithmCfg`` and so every task inherited the crash, not just this
  one.
  """

  env: RslRlVecEnvWrapper


class ManipulationDistillationRunner(_OnnxExportOnSaveMixin, MjlabDistillationRunner):
  """DAgger runner for the manipulation tasks.

  ``export_policy_to_onnx`` exports ``alg.get_policy()``, which for distillation
  is the student -- i.e. the checkpoint's ONNX is the deployable vision policy,
  not the teacher.
  """

  env: RslRlVecEnvWrapper
