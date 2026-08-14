"""WHERE on the cube does the hand actually land?

  uv run python scripts/wide_grasp_pose_audit.py NAME=TASK:CKPT [NAME=TASK:CKPT ...]
  uv run python scripts/wide_grasp_pose_audit.py envs=128 NAME=TASK:CKPT:teacher

WHY THIS EXISTS. Success on this task is 100% for the teacher and 99.2% for the
best student, so every metric currently in use -- `episode_success`,
`Train/mean_reward`, `eval_deploy.py` -- discriminates nothing between a
textbook antipodal grasp and a policy that hooks the cube by its top edge and
gets away with it because the sim's friction is generous. Watching the clips
says the grasps are bad; nothing in the pipeline says WHICH WAY they are bad, so
no reward change can be scored.

This measures the pose itself, off the actual contact points rather than any
proxy. The point of proxies has been made repeatedly on this task: pad
separation, pad midpoint, jaw yaw and pad tilt are each satisfiable with the
fingertips in mid-air. So every column here is read from a contact sensor that
only fires when a fingertip collider is genuinely touching the cube, and the
contact positions are expressed in the CUBE's own frame, which is the frame the
words "the middle of a face", "the top" and "an edge" are true in.

THE COLUMNS, in the order a bad grasp goes wrong:

  grip height mm  How high up the cube the fingers are holding it, measured from
                  the cube's CENTRE, in mm. 0 = gripping across the middle,
                  which is what is wanted. +25 = right at the top edge, the
                  failure the clips show. Negative = below centre, which on this
                  bench means the tips are pressed into the foam. Signed on
                  purpose -- the tail that matters is the high one.
  off-centre mm   How far the grip is from the middle of the face SIDEWAYS,
                  in mm -- the other axis of "not in the middle". 0 = the pads
                  straddle the face centre; 25 = the pads are at the vertical
                  edge of the face, i.e. pinching a corner.
  pad z spread mm The height DIFFERENCE between the two pads, in mm. `grip
                  height` is their midpoint, so it reads the same +10 mm for
                  "both pads 10 mm above centre" and for "one pad on the top
                  edge, one across the middle" -- two different faults wanting
                  two different fixes. This is what tells them apart: 0 means
                  the pair is level, 20+ means the hand is holding the cube
                  crooked.
  vert margin mm  The rows above say where on the face the grip is; these two
  horiz margin mm say how much face is LEFT around it, for whichever of the two
                  pads is worse off. ~25 mm is the dead centre of a face and 0
                  is exactly on an edge. `vert` is the room up to the TOP edge
                  and `horiz` the room out to the nearest VERTICAL edge, and
                  they are kept apart because a single combined margin cannot
                  say which edge the hand is hanging off -- hooking the top and
                  pinching a corner score identically and want opposite fixes.
                  Unlike a face LABEL these do not flicker when the contact sits
                  right on a boundary, which is what makes them the honest
                  reading of "is it grabbing the thing by an edge".
  pad site err mm How far the force actually lands from the PAD SITE, in mm,
  pad site lat mm for whichever pad is worse. The pad site is the point this
                  whole claw is calibrated around and the point the grasp is
                  meant to happen at, so this is the direct answer to "is it
                  gripping where it is supposed to".

                  `err` is the straight-line distance, and it does NOT bottom
                  out at zero: the site sits behind the finger's gripping
                  surface, about 8.7 mm outboard of the cube face at a real grip
                  (PAD_SEP_AT_GRASP 67.4 mm against the cube's 50 mm), so a
                  flawless grasp still reads ~8.7. That part is fixed geometry.
                  `lat` projects the closing direction out and keeps the rest,
                  so it starts at 0 for a perfect grasp and measures only the
                  part that means something: how far the contact has slid along
                  and across the finger, away from the tip and up into its
                  crook, where the cube is scooped rather than pinched.
  site off-face mm  How far the pad SITE is from the centre of the face it is
                  up against, measured in that face's plane, for the worse pad.
                  The rows above describe where the FORCE ended up; this one
                  describes where the KINEMATICS put the finger, which is the
                  only one of the two a reward can steer directly. 0 = the
                  finger is squared up on the middle of a face. This is the
                  quantity `pad_face_centring` scores, so the audit and the
                  reward cannot drift apart.
  antipodal deg   Are the two fingers pressing straight at each other? It is the
                  angle between the left contact normal and the reversed right
                  one. 0 = the textbook antipodal grasp, both pads flat on
                  opposite faces pushing along one line. Large = the pads are on
                  faces that are not opposite (a corner or an edge grasp), where
                  the two pushes do not cancel and the cube squirts out under
                  load.
  friction deg    How much of the grip is being carried by FRICTION rather than
                  by squeezing. The angle between the contact force and the
                  surface normal: 0 = pushing straight into the face, 90 =
                  pure sliding shear. The contact can only hold while this stays
                  inside the friction cone, atan(mu); at the DR band's worst
                  case (mu = 0.5) that is 26.6 deg, so anything near or past
                  that is a grasp being held by luck.
  jaw tilt deg    How far the CLOSING AXIS is out of horizontal. `jaw yaw` is
                  rotation about vertical (squaring up to a face) and cannot see
                  the hand being rolled; this can. A rolled hand tips the whole
                  pinch line and the cube hangs off it at the same angle, so
                  this is the column to read next to `cube tilt`.
  jaw yaw deg     The old alignment number, kept so this table can be read next
                  to `eval_deploy.py`: the angle between the jaw axis and the
                  nearest cube face, folded into 0-45 deg. 0 = square to a face.
  cube tilt deg   How far the cube has been knocked off flat, in deg, measured
                  as the angle between its nearest principal axis and vertical.
                  A centred grasp barely disturbs it; a top-edge grasp levers it
                  over, and this is what shows that happening.
  grip N          The WEAKER of the two contact forces. Context for everything
                  above -- a beautiful pose held at 0.3 N is not a grasp.

For most rows a bigger number is the worse one, so the last two columns are the
90th percentile and the maximum across envs. For the two margins and `grip N` it
is the other way round, and those rows report the 10th percentile and the
minimum instead -- "worst" always means worst.

  where the pads land   Which FACE each pad ends up on, as a fraction rather
                  than an average: how often a pad is on a SIDE face (what an
                  antipodal grasp uses) rather than the TOP or bottom, and how
                  often the two pads are on genuinely OPPOSITE faces. This is
                  the single number that answers "is it grabbing the thing by
                  its top", and the worst-tenth column is there because the mean
                  hides the envs that fail.

Split into two phases, because they answer different questions:

  closing   the fingers are on the cube but it has not left the surface. This is
            the pose the policy CHOSE, before the lift can rearrange it.
  carry     the cube is in the air. This is the pose that has to survive, and
            the difference between the two rows is how much the lift itself
            dragged the cube through the fingers.

Every number is reduced per env first (averaged over that env's steps in the
phase) and then across envs, so one env stuck in a weird pose for 200 steps
cannot outvote 127 normal ones.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api import math as math_utils

sys.path.insert(0, str(Path(__file__).parent))
from tools.task_geometry import TaskGeometry, geometry_for  # noqa: E402
from wide_speed_audit import LIFTED, _load_policy  # noqa: E402

STEPS, DEV = 300, "cuda:0"
# Below this a "contact" is a numerical graze, not a grip, and its position and
# normal are meaningless. Every column here is conditioned on BOTH sides being
# over it, which is also what makes "closing" a real phase rather than "the
# fingers happened to be near the cube".
GRIP_N = 0.05


def _pose_sensors(geo: TaskGeometry) -> tuple[ContactSensorCfg, ...]:
  """Per-side fingertip-vs-cube sensors that also report WHERE and WHICH WAY.

  The task's own `left_cube_contact` / `right_cube_contact` cannot answer this:
  they request only ("found", "force") and reduce with "netforce", which sums
  every contact on the finger into one wrench and throws the contact POSITIONS
  away. These are the same matches with the geometry kept -- "maxforce" so the
  strongest contact per collider survives reduction with its own position and
  normal, and `global_frame` so the force is comparable with that normal.
  """
  return tuple(
    ContactSensorCfg(
      name=f"{side}_pose_contact",
      primary=ContactMatch(mode="geom", pattern=pattern, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
      fields=("found", "force", "pos", "normal", "tangent"),
      reduce="maxforce",
      num_slots=1,
      global_frame=True,
    )
    for side, pattern in (
      ("left", geo.left_tip_geoms),
      ("right", geo.right_tip_geoms),
    )
  )


def _strongest(env: ManagerBasedRlEnv, name: str) -> tuple[torch.Tensor, ...]:
  """The single hardest-loaded contact on one finger: |f|, position, normal, f.

  The distal link is four COACD pieces, so the sensor comes back with one slot
  per piece and at most a couple of them loaded. Picking the strongest is what
  makes "the contact" well defined; averaging the four would average in three
  zeros and pull the reported position toward the finger's centroid.
  """
  data = env.scene[name].data
  assert data.force is not None and data.pos is not None
  assert data.normal is not None
  mag = torch.norm(data.force, dim=-1)  # [B, N]
  idx = mag.argmax(dim=1)  # [B]
  pick = idx.view(-1, 1, 1).expand(-1, 1, 3)
  return (
    mag.gather(1, idx.view(-1, 1)).squeeze(1),
    data.pos.gather(1, pick).squeeze(1),
    data.normal.gather(1, pick).squeeze(1),
    data.force.gather(1, pick).squeeze(1),
  )


def _angle_deg(a: torch.Tensor, b: torch.Tensor, folded: bool = False) -> torch.Tensor:
  """Angle between two batches of vectors, in degrees; NaN-safe on zero length."""
  na = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)
  nb = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-9)
  cos = (na * nb).sum(-1)
  if folded:  # direction-agnostic: a force and its reaction are the same thing
    cos = cos.abs()
  return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


def rollout(task: str, ckpt: str, role: str, num_envs: int) -> dict:
  geo = geometry_for(task)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  # The same two corrections the sibling audits make, for the same reasons: one
  # continuous window per env rather than a mixture of episodes, and no goal
  # resample, because `LiftingCommand` RESPAWNS the cube on resample -- which
  # here would teleport the cube out of the hand mid-carry and be recorded as a
  # grasp that fell apart.
  cfg.terminations = {}
  cfg.commands["lift_height"].resampling_time_range = (1e9, 1e9)
  assert cfg.scene.sensors is not None
  cfg.scene.sensors = (*cfg.scene.sensors, *_pose_sensors(geo))

  env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode=None)
  a = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=a.clip_actions)
  runner = load_runner_cls(task)(wrapped, asdict(a), device=DEV)
  policy = _load_policy(runner, ckpt, role)

  robot, cube = env.scene["robot"], env.scene["cube"]
  pad_ids = [list(robot.site_names).index(s) for s in geo.pad_sites]

  torch.manual_seed(0)
  obs = wrapped.reset()[0]
  # AFTER the reset, because `dr_cube_scale` draws it there: with size DR on,
  # the half-edge is a per-env random variable in 22.5-27.5 mm and the constant
  # in scene.py is only its mean. Reporting "25 mm from centre = the top edge"
  # off the constant would be wrong by up to 2.5 mm in either direction.
  cube_geom = int(cube.indexing.geom_ids[0])
  half = env.sim.model.geom_size[:, cube_geom, 0].clone()

  rec: dict[str, list[torch.Tensor]] = {k: [] for k in _KEYS}
  for _ in range(STEPS):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, _, _ = wrapped.step(act)

    fl, pl, nl, vl = _strongest(env, "left_pose_contact")
    fr, pr, nr, vr = _strongest(env, "right_pose_contact")
    cp = cube.data.root_link_pos_w
    cq = cube.data.root_link_quat_w
    pads = robot.data.site_pos_w[:, pad_ids]  # [B, 2, 3]

    # Contact positions in the CUBE's frame. Everything about "where on the
    # cube" is a statement in this frame and false in any other -- a tipped
    # cube's top face is not the world's +z.
    rec["loc_l"].append(math_utils.quat_apply_inverse(cq, pl - cp).clone())
    rec["loc_r"].append(math_utils.quat_apply_inverse(cq, pr - cp).clone())
    # The PAD SITES in the same frame, which is a different question from where
    # the contact is: the contact is where the force ended up, the site is where
    # the kinematics put the finger. A reward can only steer the second, so this
    # is the row that sizes one -- and it is the quantity `pad_face_centring`
    # scores, so the audit and the reward cannot drift apart.
    rec["sloc_l"].append(math_utils.quat_apply_inverse(cq, pads[:, 0] - cp).clone())
    rec["sloc_r"].append(math_utils.quat_apply_inverse(cq, pads[:, 1] - cp).clone())
    rec["f_l"].append(fl.clone())
    rec["f_r"].append(fr.clone())
    # Both normals point finger -> cube, so a perfect antipodal pair is exactly
    # anti-parallel and this reads 0.
    rec["anti"].append(_angle_deg(nl, -nr).clone())
    # Worse of the two sides: the grasp slips at whichever contact leaves the
    # friction cone first, so the mean of the pair would hide the failure.
    rec["fric"].append(
      torch.maximum(_angle_deg(vl, nl, folded=True), _angle_deg(vr, nr, folded=True))
    )
    # How far the force lands from the PAD SITE, which is the point the whole
    # claw is calibrated around and the point the grasp is supposed to happen
    # at. Straight-line distance from the site to the contact, per side.
    #
    # It has a floor, and the floor is not zero: the site sits behind the
    # finger's gripping surface, ~8.7 mm outboard of the cube face at a real
    # grip (PAD_SEP_AT_GRASP 67.4 mm against the cube's 50 mm), so even a
    # perfect grasp reads ~8.7. That component is fixed geometry and carries no
    # information, which is why the LATERAL part is recorded alongside it: the
    # residual after projecting out the closing direction is the part that
    # actually says "the force is not where the pad site is".
    axis = pads[:, 1] - pads[:, 0]
    jaw_hat = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    for key, contact, site in (("l", pl, pads[:, 0]), ("r", pr, pads[:, 1])):
      d = contact - site
      rec[f"tip_{key}"].append(torch.norm(d, dim=-1).clone())
      along = (d * jaw_hat).sum(-1, keepdim=True) * jaw_hat
      rec[f"lat_{key}"].append(torch.norm(d - along, dim=-1).clone())

    # Jaw-vs-face yaw, the same fold `jaw_alignment_gate` uses, so the number is
    # comparable with the gate's own and with eval_deploy's `jaw err`.
    jaw = torch.atan2(axis[:, 1], axis[:, 0])
    w, x, y, z = cq[:, 0], cq[:, 1], cq[:, 2], cq[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    d = (jaw - yaw) % (torch.pi / 2)
    rec["jaw"].append(torch.rad2deg(torch.minimum(d, torch.pi / 2 - d)).clone())

    # How far the closing axis itself is out of HORIZONTAL. `jaw yaw` measures
    # rotation about vertical -- squaring up to a face -- and is blind to the
    # hand being rolled, which tips the whole pinch line and takes the cube with
    # it. Added because round 2 centred the grip on every axis it scored and
    # cube tilt got WORSE, which no column then present could explain.
    rec["jawtilt"].append(
      torch.rad2deg(
        torch.asin((axis[:, 2] / axis.norm(dim=-1).clamp(min=1e-9)).clamp(-1, 1))
      )
      .abs()
      .clone()
    )

    # Cube tilt: a cube resting on ANY face is flat, so the honest measure is
    # the angle from vertical of its nearest principal axis, not of its local z.
    up = torch.tensor([0.0, 0.0, 1.0], device=cq.device).expand(cq.shape[0], 3)
    local_up = math_utils.quat_apply_inverse(cq, up)
    best = local_up.abs().amax(dim=-1).clamp(max=1.0)
    rec["tilt"].append(torch.rad2deg(torch.acos(best)).clone())

    rec["h"].append((cp[:, 2] - env.scene.env_origins[:, 2] - geo.surface_z).clone())

  out = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
  out["half"] = half.cpu().numpy()
  env.close()
  return out


_KEYS = (
  "loc_l",
  "loc_r",
  "sloc_l",
  "sloc_r",
  "f_l",
  "f_r",
  "anti",
  "fric",
  "tip_l",
  "tip_r",
  "lat_l",
  "lat_r",
  "jaw",
  "jawtilt",
  "tilt",
  "h",
)


def _derive(r: dict) -> dict:
  """Per-step columns plus the two phase masks. Shapes are (T, B)."""
  loc_l, loc_r = r["loc_l"], r["loc_r"]
  mid = 0.5 * (loc_l + loc_r)  # (T, B, 3), in the cube's frame

  # Gripped = BOTH pads loaded. Every geometric column is meaningless otherwise
  # (an unloaded slot reports a stale or zero position), so this mask gates all
  # of them rather than being reported as its own caveat.
  gripped = (r["f_l"] > GRIP_N) & (r["f_r"] > GRIP_N)

  rise = r["h"] - r["h"][0]
  lifted = np.maximum.accumulate(rise > LIFTED, axis=0)

  # Which face each pad is on: the cube-frame axis of largest magnitude. On a
  # box the contact point lies on the surface, so that axis IS the face and its
  # sign is which of the pair.
  ax_l, ax_r = np.abs(loc_l).argmax(-1), np.abs(loc_r).argmax(-1)
  sg_l = np.sign(np.take_along_axis(loc_l, ax_l[..., None], -1)[..., 0])
  sg_r = np.sign(np.take_along_axis(loc_r, ax_r[..., None], -1)[..., 0])

  return dict(
    cols={
      "grip height mm": mid[..., 2] * 1e3,
      "off-centre mm": np.linalg.norm(mid[..., :2], axis=-1) * 1e3,
      # `grip height` is the MIDPOINT of the two contacts, so on its own it
      # cannot tell "both pads 10 mm high" from "one pad on the top edge and
      # one mid-face" -- opposite failures needing opposite fixes. This is the
      # difference between the two pads' heights, which separates them.
      "pad z spread mm": np.abs(loc_l[..., 2] - loc_r[..., 2]) * 1e3,
      # How much face is left around the contact, split into its two
      # directions and taken at the WORSE of the two pads -- the grasp levers
      # apart at whichever contact runs off its face first, so averaging the
      # pair would hide it. Combined into one number these two are ambiguous:
      # a grip 2 mm under the top edge and a grip 2 mm from a vertical edge
      # score the same and want completely different fixes.
      "vert margin mm": np.minimum(
        _margins(loc_l, r["half"])[0], _margins(loc_r, r["half"])[0]
      )
      * 1e3,
      "horiz margin mm": np.minimum(
        _margins(loc_l, r["half"])[1], _margins(loc_r, r["half"])[1]
      )
      * 1e3,
      # Worse of the two pads, like the margins and the friction angle: a mean
      # pays half credit for one pad landing correctly while the other misses,
      # which is the one-sided grasp this task produces repeatedly.
      "pad site err mm": np.maximum(r["tip_l"], r["tip_r"]) * 1e3,
      "pad site lat mm": np.maximum(r["lat_l"], r["lat_r"]) * 1e3,
      # How far the SITE is from the centre of the face it is up against,
      # measured in the face's own plane. 0 means the finger is squared up on
      # the middle of a face, which is the pose every other row here is a
      # symptom of missing. This is exactly what `pad_face_centring` scores.
      "site off-face mm": np.maximum(_in_face(r["sloc_l"]), _in_face(r["sloc_r"]))
      * 1e3,
      "antipodal deg": r["anti"],
      "friction deg": r["fric"],
      "jaw yaw deg": r["jaw"],
      "jaw tilt deg": r["jawtilt"],
      "cube tilt deg": r["tilt"],
      "grip N": np.minimum(r["f_l"], r["f_r"]),
    },
    # SELF-CHECK, not a result. Every geometric column here assumes the sensor's
    # contact position lands on the cube's surface, i.e. that the largest
    # cube-frame coordinate equals the half-edge. If that assumption is wrong --
    # a stale slot, the wrong frame, the wrong geom size -- nothing else in the
    # table means anything, and it would fail as plausible numbers rather than
    # as an error. Printed in mm; anything past a millimetre or two of
    # penetration invalidates the run.
    surf=(np.abs(loc_l).max(-1) - r["half"][None, :]),
    side_face=0.5 * ((ax_l != 2).astype(float) + (ax_r != 2).astype(float)),
    opposite=((ax_l == ax_r) & (sg_l != sg_r) & (ax_l != 2)).astype(float),
    gripped=gripped,
    closing=gripped & ~lifted,
    carry=gripped & lifted,
    lifted=lifted,
    half=r["half"],
  )


def _in_face(loc: np.ndarray) -> np.ndarray:
  """Distance from a point to the centre of the face it faces, in that plane.

  The face is the cube-frame axis of largest magnitude; dropping that axis
  leaves the two in-face coordinates, and their norm is how far off the face's
  centre the point sits. Independent of the cube's size, unlike the margins, so
  it stays meaningful under `dr_cube_scale`.
  """
  a = np.abs(loc)
  ax = a.argmax(-1)[..., None]
  in_face = a.copy()
  np.put_along_axis(in_face, ax, 0.0, -1)
  return np.linalg.norm(in_face, axis=-1)


def _margins(loc: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Room left around a contact on its own face: (vertical, horizontal), in m.

  The face is the cube-frame axis of largest magnitude, and the margin is how
  much of the face is left along each of the other two directions. A contact in
  the dead centre of a side face scores the half-edge (~25 mm) both ways; one on
  the top edge scores 0 vertically, and one on a vertical edge scores 0
  horizontally. Keeping them apart is the point -- a single combined margin
  cannot say WHICH edge the hand is hanging off, and the two answers call for
  different rewards.

  A contact whose dominant axis IS z is already on the top or bottom face, so
  its vertical margin is 0 by construction.

  This is also why the FACE LABEL alone cannot be trusted near an edge: a
  contact a millimetre from the top edge has |z| and |y| both within a hair of
  the half-edge, so which one wins the argmax is a coin flip. That is what makes
  `OPPOSITE faces` read 30-50% on grasps whose contact normals are anti-parallel
  to within a degree -- both numbers are reporting the same edge contact, one by
  flickering and one by reading ~0.
  """
  a = np.abs(loc)
  ax = a.argmax(-1)
  on_top = ax == 2
  vert = np.where(on_top, 0.0, half[None, :] - a[..., 2])
  # The horizontal direction that is NOT the face normal. On the top face both
  # x and y are in-face, so the binding one is the larger.
  horiz_xy = np.where(
    on_top,
    a[..., :2].max(-1),
    np.where(ax == 0, a[..., 1], a[..., 0]),
  )
  return vert, half[None, :] - horiz_xy


def _per_env(col: np.ndarray, mask: np.ndarray) -> np.ndarray:
  """Average a (T, B) column over each env's masked steps; NaN where none."""
  n = mask.sum(0)
  tot = np.where(mask, col, 0.0).sum(0)
  return np.where(n > 0, tot / np.maximum(n, 1), np.nan)


# Rows where a SMALL number is the bad one, so their tail columns are read from
# the other end of the distribution. Everything else is "bigger is worse".
_LOW_IS_BAD = ("vert margin mm", "horiz margin mm", "grip N")


def _row(label: str, v: np.ndarray) -> str:
  if np.all(np.isnan(v)):
    return f" {label:<15}      --      --      --      --"
  lo = label in _LOW_IS_BAD
  tail = np.nanpercentile(v, 10 if lo else 90)
  worst = np.nanmin(v) if lo else np.nanmax(v)
  return (
    f" {label:<15}{np.nanmedian(v):8.1f}{np.nanmean(v):8.1f}{tail:8.1f}{worst:8.1f}"
  )


def report(name: str, r: dict) -> None:
  d = _derive(r)
  print(f"\n=== {name}")
  half = d["half"] * 1e3
  print(
    f"  self-check: contact sits {np.median(d['surf'][d['gripped']]) * 1e3:+.2f} mm"
    " off the cube surface (must be ~0)\n"
    f"  cube half-edge {half.min():.1f}-{half.max():.1f} mm, so"
    f" |grip height| = {half.mean():.1f} mm IS the top/bottom edge"
  )
  for phase in ("closing", "carry"):
    mask = d[phase]
    envs = int((mask.sum(0) > 0).sum())
    n_env = mask.shape[1]
    print(f"\n  [{phase}]  {envs}/{n_env} envs ever get a two-finger grip here")
    if envs == 0:
      continue
    print(f" {'':<15}{'median':>8}{'mean':>8}{'bad p90':>8}{'worst':>8}")
    for label, col in d["cols"].items():
      print(_row(label, _per_env(col, mask)))
    # Per env first, like every row above -- pooling these over all steps
    # instead lets one env that holds a bad pose for 200 steps outvote a
    # hundred envs that hold a good one, and the two halves of the report then
    # disagree with each other.
    side = _per_env(d["side_face"], mask)
    opp = _per_env(d["opposite"], mask)
    print(
      f"  on a SIDE face: mean {np.nanmean(side):5.1%},"
      f" worst tenth {np.nanpercentile(side, 10):5.1%}"
      f"   |   OPPOSITE faces: mean {np.nanmean(opp):5.1%},"
      f" worst tenth {np.nanpercentile(opp, 10):5.1%}"
    )
  never = int((~d["lifted"][-1]).sum())
  print(f"\n  never lifted: {never}/{d['lifted'].shape[1]} envs")


def main() -> None:
  specs, num_envs = [], 128
  for arg in sys.argv[1:]:
    if arg.startswith("envs="):
      num_envs = int(arg.split("=", 1)[1])
      continue
    name, rest = arg.split("=", 1)
    task, ckpt = rest.split(":", 1)
    role = "policy"
    if ckpt.endswith((":teacher", ":student")):
      ckpt, tail = ckpt.rsplit(":", 1)
      role = "teacher" if tail == "teacher" else "policy"
    specs.append((name, task, ckpt, role))

  if not specs:
    print(__doc__)
    raise SystemExit(2)

  for name, task, ckpt, role in specs:
    report(name, rollout(task, ckpt, role, num_envs))

  print(
    f"\n{num_envs} envs x {STEPS} steps, deterministic policy, terminations off."
    "\nEvery row is conditioned on BOTH fingertips loaded above"
    f" {GRIP_N} N -- steps where"
    "\nthe hand is not actually holding the cube contribute nothing, so a policy"
    "\nthat grips for 10 steps and one that grips for 200 are scored on the pose"
    "\nrather than on the duration."
  )


if __name__ == "__main__":
  main()
