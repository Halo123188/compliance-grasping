## Quickstart

Dependencies: Python 3.10+, `mujoco`, `numpy` (model + headless checks); `mjviser` +
`viser` (interactive viewer only).

**Standalone hand — no arm, no Menagerie:**
```bash
python mygripper_arm_viewer.py --check --hand-only         # headless sanity (15 DOF, 6 act)
python mygripper_arm_viewer.py --hand-only                 # interactive, weld closure
python mygripper_arm_viewer.py --hand-only --coupled       # interactive, mimic closure
```

**Mounted on the Rizon 4S — needs MuJoCo Menagerie:**
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
export MJ_MENAGERIE=$PWD/mujoco_menagerie     # default: ~/mujoco_menagerie
python mygripper_arm_viewer.py --check --weld
python mygripper_arm_viewer.py --weld         # arm + hand, rigid loop closure
python mygripper_arm_viewer.py --weld --port 8083
```

The viewer is a web app (viser): it prints a port — tunnel it
(`ssh -L PORT:localhost:PORT user@host`) and open `http://localhost:PORT`. The
**Actuation tab → Mount Adjustment** sliders position the gripper on the flange; the
`--weld` build also shows a **live weld-alignment readout** (should stay ~0 mm).

## Layout

```
mygripper_arm_viewer.py            model + viewer: build_model / _coupled / _welded / build_hand_only
assets/robots/mygripper_H100_R/    URDF + decimated STL meshes
scripts/                           load check + loop-closure / joint-limit diagnostics
scripts/prototyping/               exploratory scripts behind the loop-closure work
docs/                              derivation notes (loop closure, weld stabilization, joint limits)
```

## Provenance
URDF + meshes originally exported in `jonathan0713/Dex-Retargeting-personal`.
