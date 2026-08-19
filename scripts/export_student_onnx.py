"""Export a bare distillation `model_N.pt` to a deployable ONNX.

`tasks/compliance/deploy/export_checkpoint_onnx.py` rebuilds an actor from the
run's own `params/agent.yaml`. This one exists for the case where that file is
gone: a checkpoint copied off the cluster on its own carries `student_state_dict`
and an iteration count, and nothing else. The architecture is then recoverable
from the WEIGHTS, and this script recovers it rather than asking for a config
that no longer exists:

    hidden dims        the MLP's own layer widths
    CNN channels       each conv's out_channels
    CNN kernels        each conv's kernel size
    observation width  the baked normalizer's width
    action width       the last MLP bias
    feature grid       the spatial-softmax coordinate buffers

WHAT THE WEIGHTS CANNOT SAY is the training env: the action term (scale, finger
travel, slew limit) and the CNN's STRIDE. Both come from `deploy/profiles.py`,
which is why `--profile` is required.

The stride is the subtle one and the reason this script is not simply
`load_state_dict`. Same-padding makes each layer `ceil(dim / stride)`, so a
30x40 feature grid is a 120x160 frame at stride 1/2/2 and a 240x320 frame at
2/2/2 -- and BOTH rebuild, load with `strict=True`, and export. Getting it wrong
does not raise; it silently ships a network that reads the world at the wrong
scale. So the profile declares the stride and the frame size, and this script
checks the PAIR against the grid the checkpoint actually carries. That is a real
check only because neither half is derived from the other.

The metadata written here is the profile's claim rather than independent
evidence -- a training-time export reads it off the live env, this one copies it
out of `calib.py` -- so what `deploy/policy.py`'s check catches is drift between
this export and calib.py, not a wrong profile.

    uv run python scripts/export_student_onnx.py \
        --checkpoint ~/yiboc/cube_model_2999.pt --profile cube

writes `cube_model_2999.onnx` beside the checkpoint, named so that
`profiles.select` finds the same profile again from the filename alone.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel

# `deploy/` is a top-level directory rather than an installed package, because
# the robot host runs it without mjlab on the machine at all. Same import dance
# as `wide_onnx_parity.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))
from deploy import calib, profiles  # noqa: E402

# The distillation student's architecture, minus everything derived below. These
# are the settings `params/agent.yaml` carried on every wide-claw run; they are
# not recoverable from the weights and have not changed since the claw was
# fitted, so they are constants here rather than flags.
ACTIVATION = "elu"
PADDING = "zeros"  # i.e. same-padding; see rsl_rl.modules.cnn._compute_padding
SPATIAL_SOFTMAX_TEMPERATURE = 1.0
DISTRIBUTION_CFG = {
  "class_name": "GaussianDistribution",
  "init_std": 0.02,
  "std_type": "scalar",
}


def _conv_indices(state: dict) -> list[int]:
  return sorted(
    int(k.split(".")[3])
    for k in state
    if k.startswith("cnns.camera.cnn.") and k.endswith(".weight")
  )


def _mlp_indices(state: dict) -> list[int]:
  return sorted(
    int(k.split(".")[1])
    for k in state
    if k.startswith("mlp.") and k.endswith(".weight")
  )


def feature_grid(state: dict) -> tuple[int, int]:
  """The CNN's output (H, W), read off the spatial-softmax coordinate buffers.

  `SpatialSoftmax` builds them with `meshgrid(linspace(-1,1,H), linspace(-1,1,W))`
  flattened, so `pos_x` holds H distinct values and `pos_y` holds W of them.
  """
  px = state["cnns.camera.spatial_softmax.pos_x"].numpy().ravel()
  py = state["cnns.camera.spatial_softmax.pos_y"].numpy().ravel()
  h, w = len(np.unique(px)), len(np.unique(py))
  if h * w != px.size:
    raise SystemExit(f"spatial-softmax grid {h}x{w} does not fill {px.size} points")
  return h, w


def same_padded_grid(
  depth_hw: tuple[int, int], strides: tuple[int, ...]
) -> tuple[int, int]:
  """The feature map `depth_hw` becomes under same-padded `strides`.

  `rsl_rl.modules.cnn` pads to keep `ceil(dim / stride)` at every layer, so this
  is the whole of it. Compared against the checkpoint's own grid below; the
  comparison is the point, since the grid alone does not determine either the
  stride or the frame size.
  """
  h, w = depth_hw
  for stride in strides:
    h, w = math.ceil(h / stride), math.ceil(w / stride)
  return h, w


def build_student(
  state: dict, strides: tuple[int, ...], depth_hw: tuple[int, int]
) -> SpatialSoftmaxCNNModel:
  """Reconstruct the student from its own weights, ready for `load_state_dict`."""
  obs_dim = int(state["obs_normalizer._mean"].shape[-1])
  mlp = _mlp_indices(state)
  hidden_dims = [int(state[f"mlp.{i}.weight"].shape[0]) for i in mlp[:-1]]
  act_dim = int(state[f"mlp.{mlp[-1]}.weight"].shape[0])
  conv = _conv_indices(state)
  cnn_cfg = {
    "output_channels": [
      int(state[f"cnns.camera.cnn.{i}.weight"].shape[0]) for i in conv
    ],
    "kernel_size": [int(state[f"cnns.camera.cnn.{i}.weight"].shape[-1]) for i in conv],
    "stride": list(strides),
    "padding": PADDING,
    "activation": ACTIVATION,
    "max_pool": False,
    "spatial_softmax_temperature": SPATIAL_SOFTMAX_TEMPERATURE,
  }
  channels = int(state[f"cnns.camera.cnn.{conv[0]}.weight"].shape[1])
  if len(strides) != len(conv):
    raise SystemExit(
      f"--cnn-stride has {len(strides)} entries for a {len(conv)}-layer CNN"
    )
  dummy = TensorDict(
    {
      "student": torch.zeros(1, obs_dim),
      "camera": torch.zeros(1, channels, *depth_hw),
    },
    batch_size=[1],
  )
  return SpatialSoftmaxCNNModel(
    obs=dummy,
    obs_groups={"student": ["student", "camera"]},
    obs_set="student",
    output_dim=act_dim,
    cnn_cfg=cnn_cfg,
    hidden_dims=hidden_dims,
    activation=ACTIVATION,
    obs_normalization="obs_normalizer._mean" in state,
    distribution_cfg=DISTRIBUTION_CFG,
  )


def metadata(profile: profiles.Profile, checkpoint: Path, history: int) -> dict:
  """The scene constants `deploy/policy.py` checks, from calib.py and the profile.

  NOT independent evidence. A training-time export reads these off the live env;
  here they are asserted copies, so what the load-time check catches is a
  calib.py that has moved since the export, or a file that has been edited.
  """
  return {
    "run_path": f"{checkpoint.name} (offline export)",
    "joint_names": list(calib.JOINT_NAMES),
    "joint_stiffness": list(calib.JOINT_STIFFNESS),
    "joint_damping": list(calib.JOINT_DAMPING),
    "default_joint_pos": list(calib.DEFAULT_JOINT_POS),
    "command_names": ["lift_height"],
    # The training exporter writes the TEACHER's terms here, which is why
    # policy.py ignores this key and takes the layout from the profile. Written
    # for the record, and labelled.
    "observation_names": list(profile.term_names),
    "action_scale": list(profile.action_scale),
    "deploy_profile": profile.name,
    "student_history_length": history,
    "depth_hw": list(profile.depth_hw),
    "cnn_stride": list(profile.cnn_stride),
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", type=Path, required=True, help="a model_N.pt")
  ap.add_argument(
    "--profile",
    required=True,
    help=(
      "which deploy/profiles.py entry this checkpoint is. Required because the "
      "action scale, the finger travel and the slew limit are not in the "
      f"weights. One of: {', '.join(sorted(profiles.PROFILES))}"
    ),
  )
  ap.add_argument("--out", type=Path, default=None, help="default: alongside")
  ap.add_argument(
    "--cnn-stride",
    default=None,
    help="override the profile's per-layer CNN stride, as e.g. 1,2,2",
  )
  ap.add_argument(
    "--depth-hw",
    default=None,
    help="override the profile's depth frame, as H,W",
  )
  ap.add_argument(
    "--weights",
    default="student_state_dict",
    help="which state dict to export (the student is the deployable half)",
  )
  ap.add_argument(
    "--check",
    action="store_true",
    help="run the export against the torch model on random inputs (needs onnxruntime)",
  )
  args = ap.parse_args()

  if not args.checkpoint.exists():
    raise SystemExit(f"no such checkpoint: {args.checkpoint}")
  profile = profiles.by_name(args.profile)
  if args.cnn_stride is None:
    strides = tuple(profile.cnn_stride)
  else:
    strides = tuple(int(x) for x in args.cnn_stride.split(","))
  if args.depth_hw is None:
    depth_hw = profile.depth_hw
  else:
    h, w = (int(x) for x in args.depth_hw.split(","))
    depth_hw = (h, w)

  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  if args.weights not in ckpt:
    raise SystemExit(f"{args.checkpoint.name} has no {args.weights}: {sorted(ckpt)}")
  state = ckpt[args.weights]

  grid = feature_grid(state)
  obs_dim = int(state["obs_normalizer._mean"].shape[-1])
  history, remainder = divmod(obs_dim, profile.frame_dim)
  if remainder:
    raise SystemExit(
      f"profile {profile.name} assembles a {profile.frame_dim}-d frame but this "
      f"checkpoint reads {obs_dim}-d, which is not a whole number of them."
    )

  print(f"checkpoint : {args.checkpoint.name} (iter={ckpt.get('iter')})")
  print(f"profile    : {profile.name}")
  print(
    f"observation: {obs_dim}-d = {history} x {profile.frame_dim}-d frame "
    f"({'+'.join(f'{n}:{w}' for n, w in profile.obs_terms)})"
  )
  stride_str = "/".join(str(s) for s in strides)
  print(
    f"depth      : {depth_hw[0]}x{depth_hw[1]} at stride {stride_str} "
    f"-> {grid[0]}x{grid[1]} grid"
  )
  want = same_padded_grid(depth_hw, strides)
  if want != grid:
    raise SystemExit(
      f"profile {profile.name} declares a {depth_hw[0]}x{depth_hw[1]} frame at "
      f"stride {stride_str}, which makes a {want[0]}x{want[1]} feature map -- "
      f"but this checkpoint's spatial softmax is {grid[0]}x{grid[1]}. One of "
      "the two is wrong, and the checkpoint is not the one that can be edited. "
      f"A {grid[0]}x{grid[1]} grid off a {depth_hw[0]}x{depth_hw[1]} frame "
      f"needs a total stride of {depth_hw[0] // grid[0]}."
    )

  student = build_student(state, strides, depth_hw)
  # STRICT: the grid is a buffer, so this catches a wrong stride PRODUCT. It
  # does not catch a wrong stride at a wrong resolution -- the check above is
  # what does that, and it is the reason the profile declares both.
  student.load_state_dict(state, strict=True)
  student.eval()

  onnx_model = student.as_onnx()
  onnx_model.to("cpu").eval()
  out = args.out or args.checkpoint.with_suffix(".onnx")
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
  attach_metadata_to_onnx(str(out), metadata(profile, args.checkpoint, history))
  print(f"wrote      : {out}")

  if args.check:
    import onnxruntime as ort  # noqa: PLC0415 -- only needed for the check

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(1, obs_dim)).astype(np.float32)
    cam = rng.uniform(size=(1, 1, *depth_hw)).astype(np.float32)
    with torch.no_grad():
      want = onnx_model(torch.from_numpy(obs), torch.from_numpy(cam)).numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
      ["actions"], {"obs": obs, "camera": cam}
    )[0]
    print(f"check      : worst |torch - onnx| = {np.abs(want - got).max():.3e}")


if __name__ == "__main__":
  main()
