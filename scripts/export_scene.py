"""Export the two-finger grasp scene as a self-contained MuJoCo model.

Produces ``scene.xml`` plus an ``assets/`` directory of the meshes it references,
which loads in vanilla MuJoCo (``mujoco.MjModel.from_xml_path``) with no mjlab
dependency. Use this to hand the *model* to someone; the RL task (rewards,
observations, actions) lives in the mjlab config and is not part of the export.

  uv run python scripts/export_scene.py [OUT_DIR] [--zip]
"""

import re
import shutil
import sys
from pathlib import Path

import mujoco

from mjlab.scene import Scene
from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import (
  CUBE_HALF,
  CUBE_REGION,
  TABLE_H,
)
from mjlab.tasks.registry import load_env_cfg

args = [a for a in sys.argv[1:] if not a.startswith("-")]
out = Path(args[0] if args else "exports/two_finger_grasp_scene").resolve()
as_zip = "--zip" in sys.argv

REPO = Path(__file__).resolve().parents[1]
# Roots the exported ``file=`` attributes are resolved against. Scene.write()
# only emits assets that live in ``spec.assets`` (in-memory), which is empty for
# specs loaded from disk -- so we copy the referenced meshes ourselves.
MESH_ROOTS = (
  REPO / "assets" / "flexiv_rizon4s" / "assets",
  REPO / "assets" / "two_finger_hand" / "meshes",
)

cfg = load_env_cfg("Mjlab-Grasp-TwoFinger-Flexiv", play=True)
cfg.scene.num_envs = 1
scene = Scene(cfg.scene, device="cpu")
scene.write(out, zip=False)

xml_path = out / "scene.xml"
xml = xml_path.read_text()

# MjSpec.attach() namespaces asset *declarations* but not the material reference
# inside a <default> block, so the exported XML carries `material="robot"` while
# the material is declared as `robot/robot`. Repoint any dangling reference at
# the declared name that ends in "/<name>".
declared = set(re.findall(r'<material name="([^"]+)"', xml))
for ref in set(re.findall(r'material="([^"]+)"', xml)):
  if ref in declared:
    continue
  match = [d for d in declared if d.endswith("/" + ref)]
  if len(match) == 1:
    xml = xml.replace(f'material="{ref}"', f'material="{match[0]}"')
    print(f"patched dangling material reference {ref!r} -> {match[0]!r}")

xml_path.write_text(xml)

# Copy every mesh the XML references.
copied = 0
for ref in sorted(set(re.findall(r'file="([^"]+)"', xml))):
  src = next((r / ref for r in MESH_ROOTS if (r / ref).exists()), None)
  if src is None:
    raise FileNotFoundError(f"mesh {ref!r} not found under {MESH_ROOTS}")
  dst = out / "assets" / ref
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)
  # OBJ meshes reference a .mtl sitting alongside them.
  mtl = src.parent / "material.mtl"
  if src.suffix == ".obj" and mtl.exists():
    shutil.copy2(mtl, dst.parent / "material.mtl")
  copied += 1

# Collapse the keyframes into one usable initial state. The export carries two:
# the arm asset's stale folded-up HOME pose (key 0, what a viewer opens on) and
# the task's hover pose. Neither places the cube, whose free joint therefore
# defaults to the world origin -- i.e. buried inside the table. Keep the hover
# pose, drop the stale one, and seat the cube at the centre of its spawn region.
model = mujoco.MjModel.from_xml_path(str(xml_path))
hover = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "robot/init_state")
assert hover >= 0, "expected a 'robot/init_state' keyframe in the export"
qpos = model.key_qpos[hover].copy()
ctrl = model.key_ctrl[hover].copy()

cube_qadr = model.jnt_qposadr[
  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube/cube_joint")
]
cx = sum(CUBE_REGION["x"]) / 2
qpos[cube_qadr : cube_qadr + 7] = [cx, 0.0, TABLE_H + CUBE_HALF, 1.0, 0.0, 0.0, 0.0]

fmt = " ".join(f"{v:g}" for v in qpos)
key = f'    <key name="init_state" qpos="{fmt}" ctrl="{" ".join(f"{v:g}" for v in ctrl)}" />'
xml = re.sub(
  r"  <keyframe>.*?</keyframe>", f"  <keyframe>\n{key}\n  </keyframe>", xml, flags=re.S
)
# <size nkey="2"> preallocates slots, so dropping a key would otherwise leave a
# nameless empty keyframe behind.
xml = re.sub(r'nkey="\d+"', 'nkey="1"', xml)
xml_path.write_text(xml)

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
print(
  f"verified in vanilla MuJoCo: {model.nbody} bodies, {model.nq} qpos, "
  f"{model.ngeom} geoms, {model.nu} actuators, {model.ncam} cameras, "
  f"{copied} meshes"
)

if as_zip:
  shutil.make_archive(str(out), "zip", root_dir=out)
  shutil.rmtree(out)
  print(f"wrote {out}.zip")
else:
  print(f"wrote {out}/")
