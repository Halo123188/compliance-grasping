"""Which scene constants a task id belongs to.

Two grasp packages now register tasks against two different robots on two
different benches, and the eval/render scripts need three numbers from whichever
one they were pointed at:

  lift_height   the bar success is scored against
  surface_z     the height cube height is measured FROM. This is the one that
                bites: on the old bench it is the table top, on the new one it is
                the top of 50 mm of foam, 50 mm above the arm's own mount. Using
                the wrong one shifts every reported peak by 50 mm, silently and
                in the direction that makes a working policy look like a failure.
  pad_sites     the fingertip site names, which gained a ``_tf`` suffix.
  tip_geoms     the per-side fingertip COLLIDER patterns. Same trap as the
                sites: this claw's distal link is a 4-part COACD decomposition,
                so the old ``left_2_col`` matches nothing here -- and a contact
                sensor whose pattern matches nothing reports no contact rather
                than raising.

Keyed off the task id rather than passed in, so a script cannot be run against
one task with another's geometry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskGeometry:
  lift_height: float
  surface_z: float
  pad_sites: tuple[str, str]
  grasp_site: str
  left_tip_geoms: str
  right_tip_geoms: str


def geometry_for(task: str) -> TaskGeometry:
  if "TwoFingerWide" in task:
    from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.env_cfgs import (
      GRASP_SITE,
      LEFT_FINGERTIP_GEOMS,
      LIFT_HEIGHT,
      PAD_SITES,
      RIGHT_FINGERTIP_GEOMS,
    )
    from mjlab.tasks.manipulation.config.flexiv_two_finger_wide.scene import (
      WORK_SURFACE_Z,
    )

    return TaskGeometry(
      LIFT_HEIGHT,
      WORK_SURFACE_Z,
      PAD_SITES,
      GRASP_SITE,
      LEFT_FINGERTIP_GEOMS,
      RIGHT_FINGERTIP_GEOMS,
    )

  from mjlab.tasks.manipulation.config.flexiv_two_finger.env_cfgs import (
    LEFT_FINGERTIP_GEOMS,
    LIFT_HEIGHT,
    PAD_SITES,
    RIGHT_FINGERTIP_GEOMS,
    TABLE_H,
  )

  return TaskGeometry(
    LIFT_HEIGHT,
    TABLE_H,
    PAD_SITES,
    "grasp_site",
    LEFT_FINGERTIP_GEOMS,
    RIGHT_FINGERTIP_GEOMS,
  )
