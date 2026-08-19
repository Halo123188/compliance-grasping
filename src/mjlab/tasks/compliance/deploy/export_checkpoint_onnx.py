"""Export any ``model_*.pt`` in a run directory to a deployable ONNX.

Training only writes an ONNX alongside the *latest* checkpoint (see
``ManipulationOnPolicyRunner.save``), and it overwrites the same
``<run_name>.onnx`` every time.  So by the end of a run only the final policy is
deployable, and comparing against an earlier iteration -- to check whether late
training changed the behaviour -- is impossible without re-exporting.

This rebuilds the actor straight from ``params/agent.yaml`` plus the checkpoint's
``actor_state_dict``, with no environment, no GPU and no mjlab task registry, then
runs the same ``as_onnx()`` + ``torch.onnx.export`` path the runner uses.

    uv run python .../export_checkpoint_onnx.py \
        --checkpoint logs/rsl_rl/<exp>/<run>/model_4500.pt

writes ``<run>/model_4500.onnx``, which you point the deploy script at:

    ... deploy_rizon_torque.py --policy tracking --onnx <that file>

Verify an export reproduces the shipped one with ``--verify``: re-export the
newest checkpoint and diff it against the ONNX training wrote.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from rsl_rl.models.rnn_model import RNNModel
from tensordict import TensorDict


def build_actor(agent_cfg: dict, obs_dim: int, act_dim: int) -> RNNModel:
  """Reconstruct the training-time actor from the saved agent config."""
  actor_cfg = dict(agent_cfg["actor"])
  actor_cfg.pop("class_name", None)
  # Same normalization MjlabOnPolicyRunner.__init__ does: the saved config keeps
  # optional sections as explicit nulls, but the model's __init__ has no such
  # parameters, so they must be dropped rather than passed as None.
  for opt in ("cnn_cfg", "distribution_cfg"):
    if actor_cfg.get(opt) is None:
      actor_cfg.pop(opt, None)
  if actor_cfg.get("rnn_type") is None:
    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      actor_cfg.pop(opt, None)
  actor_cfg.pop("cnn_cfg", None)  # never a RNNModel parameter
  obs_groups = {k: list(v) for k, v in agent_cfg["obs_groups"].items()}
  group = obs_groups["actor"][0]
  dummy = TensorDict({group: torch.zeros(1, obs_dim)}, batch_size=[1])
  return RNNModel(
    obs=dummy,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=act_dim,
    **actor_cfg,
  )


def infer_dims(state_dict: dict) -> tuple[int, int]:
  """Read (obs_dim, act_dim) off the weights rather than trusting a task cfg."""
  obs_dim = act_dim = None
  for k, v in state_dict.items():
    if not hasattr(v, "shape"):
      continue
    if k.endswith("_mean") and v.ndim == 2:
      obs_dim = int(v.shape[-1])
    if k.startswith("rnn.weight_ih") and obs_dim is None:
      obs_dim = int(v.shape[-1])
  # The last MLP layer's bias is the action dimension.
  mlp_biases = [
    (int(k.split(".")[1]), v)
    for k, v in state_dict.items()
    if k.startswith("mlp.") and k.endswith(".bias") and hasattr(v, "shape")
  ]
  if mlp_biases:
    act_dim = int(max(mlp_biases, key=lambda kv: kv[0])[1].shape[0])
  if obs_dim is None or act_dim is None:
    raise SystemExit(f"could not infer dims from keys: {sorted(state_dict)[:12]}")
  return obs_dim, act_dim


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", type=Path, required=True, help="path to model_N.pt")
  ap.add_argument(
    "--out", type=Path, default=None, help="output .onnx (default: alongside)"
  )
  ap.add_argument(
    "--verify",
    action="store_true",
    help="also diff against the run's shipped <run_name>.onnx (only meaningful "
    "for the newest checkpoint, which is the one training exported)",
  )
  args = ap.parse_args()

  ckpt_path: Path = args.checkpoint
  if not ckpt_path.exists():
    raise SystemExit(f"no such checkpoint: {ckpt_path}")
  run_dir = ckpt_path.parent
  agent_yaml = run_dir / "params" / "agent.yaml"
  if not agent_yaml.exists():
    raise SystemExit(f"missing {agent_yaml}; cannot rebuild the actor")

  agent_cfg = yaml.unsafe_load(agent_yaml.read_text())
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  sd = ckpt["actor_state_dict"]
  obs_dim, act_dim = infer_dims(sd)
  print(f"checkpoint : {ckpt_path.name}  (iter={ckpt.get('iter')})")
  print(
    f"actor      : obs={obs_dim} act={act_dim} {agent_cfg['actor']['rnn_type']} "
    f"hidden={agent_cfg['actor']['rnn_hidden_dim']}"
  )

  actor = build_actor(agent_cfg, obs_dim, act_dim)
  missing, unexpected = actor.load_state_dict(sd, strict=False)
  if missing:
    print(f"  [warn] missing keys: {list(missing)[:6]}")
  if unexpected:
    print(f"  [warn] unexpected keys: {list(unexpected)[:6]}")

  onnx_model = actor.as_onnx()
  onnx_model.to("cpu").eval()
  out = args.out or run_dir / f"{ckpt_path.stem}.onnx"
  torch.onnx.export(
    onnx_model,
    onnx_model.get_dummy_inputs(),  # type: ignore[operator]
    str(out),
    export_params=True,
    opset_version=18,
    input_names=onnx_model.input_names,  # type: ignore[arg-type]
    output_names=onnx_model.output_names,  # type: ignore[arg-type]
    dynamic_axes={},
    dynamo=False,
  )
  print(f"wrote      : {out}")

  if args.verify:
    import numpy as np
    import onnx

    shipped = run_dir / f"{run_dir.name}.onnx"
    if not shipped.exists():
      raise SystemExit(f"no shipped ONNX at {shipped}")
    a = {
      i.name: onnx.numpy_helper.to_array(i)
      for i in onnx.load(str(out)).graph.initializer
    }
    b = {
      i.name: onnx.numpy_helper.to_array(i)
      for i in onnx.load(str(shipped)).graph.initializer
    }
    if set(a) != set(b):
      print(f"  [FAIL] initializer names differ: {set(a) ^ set(b)}")
      return
    worst = max((float(np.abs(a[k] - b[k]).max()), k) for k in a)
    print(f"  verify vs {shipped.name}: worst |diff| = {worst[0]:.3e} on {worst[1]}")
    print("  -> identical" if worst[0] == 0.0 else "  -> DIFFERENT")


if __name__ == "__main__":
  main()
