=========
Changelog
=========

Upcoming version (not yet released)
-----------------------------------

Added
^^^^^

- ``deploy/run.py --record`` now carries the four finger motors' current, and
  the run prints a ``[grip]`` summary of it against ``--grip-cap`` whether or
  not the recording was on. The gripper runs current-based position control, so
  the cap IS the grip force, and without this column a finger that stopped short
  of its goal was undiagnosable: being held by the object and being stuck look
  the same in position and velocity. Eleven bench recordings of the round-3
  arms all end with the object pushed sideways and still on the surface, and
  which of the two that is decides whether the fix is ``--grip-cap`` or the
  linkage. The current comes from the same OBS packet as the pose and velocity
  beside it, and is reported per motor rather than aggregated, because the
  asymmetric case -- one pad leaning on the object while the other never
  arrives -- is how an object leaves the jaw instead of being lifted.
- Added ``scripts/wide_eval_sweep.py`` and ``scripts/wide_plot_sweep.py``, which
  split a policy's single deployment success number by object SIZE and by WHERE
  on the bench the object spawned. ``eval_deploy`` averages over the whole
  training distribution at once, so three of the four round-1/round-2 students
  score 96-100% there and none of that says where the remaining failures live.
  The sweep pins the cube to one edge length at a time (15-60 mm, deliberately
  outside every policy's trained range at both ends) and then to one cell of the
  spawn box at a time, driving both by mutating the live cfg between rollouts so
  the scene compile and the checkpoint load happen once per policy rather than
  once per point. It records per-episode rows including the whole height trace,
  because the task's success bar is an ABSOLUTE height of the cube centre and so
  asks a 15 mm cube for 92.5 mm of climb against a 60 mm cube's 70 -- the plot
  scores both that bar and a size-fair one, and they agree, which is what rules
  the metric out as the explanation for the small-cube cliff.
  ``box=x0,x1,y0,y1`` sweeps the position grid over the WHOLE bench rather than
  the trained spawn box, where the limit stops being the policy: past the D435's
  cone the student is looking at an empty table, and past ~720 mm the arm needs
  more than 1.5 sigma of action to get there. The figure draws both boundaries
  from ``wide_spawn_gate.feasibility_map`` so a dead cell can be read as blind or
  out of reach rather than as a policy that cannot grasp there.
- Added ``size_privileged`` to ``flexiv_two_finger_distill_env_cfg`` and the two
  arms it registers,
  ``Mjlab-Grasp-TwoFingerWide-Flexiv-Distill-Depth-R2-{Reach,Small}-SizeBlind``.
  ``observe_object_size`` was added for the TEACHER, which reads privileged state
  by definition, and the student inherited it only because ``_PRIVILEGED_TERMS``
  predates the term -- so the round-2 students are handed the exact half-edge of
  the cube they are meant to be looking at, at zero noise under ``play``, and
  their small-object result answers "can it grasp a 15 mm cube it has been TOLD
  is 15 mm" rather than "can it read the size off the image". The new arms
  withhold it; everything else, the teacher's ``actor`` group included, is
  unchanged, so both distil from the same R2 checkpoints and the pair prices what
  reading size off a 160x120 depth image is worth. It is a task parameter rather
  than an edit to ``_PRIVILEGED_TERMS`` because it changes the student's
  observation WIDTH (35 dimensions to 34): editing the constant would orphan
  every student already on disk instead of putting the difference in the task id.
  The new width is the one ``deploy/policy.py`` already assembles.
- Added ``wide_spawn_gate.feasibility_map``, the per-pose form of the gate's
  camera and reach checks (the gate itself reduces it to a worst case), so the
  evaluation figures draw the same limits the gate enforces rather than a second
  implementation of them.
- Added PPO fine-tuning of an already-distilled camera student:
  ``ManipulationFinetuneRunner`` warm-starts a PPO actor from a distillation
  checkpoint's ``student_state_dict``, and ``FinetunePPO`` adds
  ``critic_warmup_iters`` (hold the actor still while the fresh critic fits the
  returns of a policy that is not moving) and ``std_max`` (clamp the executed
  std, which the policy gradient inflates even at ``entropy_coef`` 0, and whose
  cliff on this task is measured: 0.02 → 97.8% lifted, 0.05 → 42.5%). The point
  is that behaviour cloning is per-timestep and so cannot price temporal
  roughness at all -- a student sitting a steady 1.6° off and one alternating
  ±1.6° score identically -- while the smoothness rewards that made the teacher
  smooth are already computed on every distillation step and thrown away.
  Registered as
  ``Mjlab-Grasp-TwoFingerWide-Flexiv-Finetune-Depth-Success-Dr-FaceLevel``,
  whose env drops the ``cube_lifted`` termination so that holding the cube keeps
  paying; without that the policy is better off hovering than grasping.
- Added ``SmoothedJointPositionActionCfg``, a joint position action that bounds
  how fast the *command* may move: a hard slew cap in rad/s (``rate_limit``), a
  first-order low-pass time constant in seconds (``ema_tau``), or both. Both are
  in physical units and integrated at the physics timestep, so they mean the same
  thing at any ``decimation`` and can be mirrored on the robot host at any publish
  rate. ``action_rate_limit_curriculum`` scales the cap over training.
- Added ``scripts/wide_speed_audit.py``, which reports a policy's joint
  velocities, published and requested command rates, and tool-point speed, split
  by approach-versus-carry phase, with a table of what each candidate limit would
  clip.
- Added the wide-claw speed arms ``Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-{Slew,SlewTight,SlewCurr,Slow,Ema}``
  and their matching ``-Distill-Depth-`` students. ``Slew``/``SlewTight``/``SlewCurr``
  cap the commanded rate, ``Ema`` low-passes it, and ``Slow`` leaves the action
  space alone and tightens the velocity penalty instead.
- Added ``action_saturation_penalty`` and ``smoothed_action_command``, the two
  halves of keeping a policy off its own rate limiter. A slew cap bounds speed
  but not direction changes, and saturating it is a degeneracy: past the cap
  only the sign of ``target - cmd`` reaches the simulation, so the request
  magnitude stops receiving gradient, drifts outward, and the joint settles into
  bang-bang chatter. The reward charges for the overdrive (dimensionless, in
  units of the cap, so a cap curriculum does not rescale it); the observation
  exposes ``q_cmd``, the limiter state the policy was previously fighting blind.
  ``SmoothedJointPositionAction`` gained ``command`` and ``saturation``
  properties to feed them.
- Added the anti-chatter arms
  ``Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr-{SatPen,CmdObs,Fix,FixRate,FixWeak}``
  and their matching ``-Distill-Depth-`` students, all on SlewCurr's cap and
  schedule. The ones that observe the command run at 54 observation dimensions
  rather than 43 and so cannot warm-start from an existing checkpoint.
- ``mjlab.tasks.compliance.deploy``: per-joint system identification on the real
  Rizon 4S, and the real-time torque bridge it runs over. The arm XML declares
  3.17/1.38/0.13 armature in its tier default classes but applies none of it to
  the ``<joint>`` elements, so the compiled model has ``dof_armature`` all zero
  and the shoulder came out 6x too light; ``frictionloss`` and ``damping`` are
  zero too. ``identify_rizon.py`` measures all three plus the gravity-comp bias,
  one joint at a time, from breakaway ramps (Coulomb friction, and the one
  measurement that does not depend on the regression converging), sines (inertia,
  because a sinusoid bounds the position excursion where a torque staircase does
  not) and velocity sweeps (damping, which sines alone cannot separate from
  Coulomb friction). The RDK's Python bindings exclude every real-time mode and
  the Scheduler, so ``rt_bridge/`` is a C++ process holding the 1 kHz loop with
  shared memory to the Python side -- ten ticks per policy step, which is the
  sim's ``decimation=10`` rather than an approximation of it.
- ``scripts/export_student_onnx.py``: turns a bare distillation ``model_N.pt``
  into a deployable ONNX. The training-time exporter needs the run's
  ``params/agent.yaml``; this one is for the checkpoints that arrive without a
  ``params/`` directory at all, and rebuilds the student from its own weights --
  layer widths, conv channels and kernels, observation width, and the CNN's
  feature grid, which is a buffer inside the checkpoint. What the weights do not
  settle is the CNN's STRIDE: same padding makes each layer ``ceil(dim/stride)``,
  so a 30x40 grid is a 120x160 frame at stride 1/2/2 or a 240x320 one at 2/2/2,
  and both load with ``strict=True``. So ``profiles.Profile`` declares both
  ``depth_hw`` and ``cnn_stride``, and the exporter checks the pair against the
  grid -- a check rather than a restatement, because neither half is derived
  from the other. ``--check`` runs the export against the torch model.
- ``deploy/`` runs the four round-3 students (``square_fixH``,
  ``square_variableH``, ``cube``, ``everyShape``). They pair round 2's action
  term -- fingers at 0.45, travel to -0.32 -- with a v11 34-d observation and no
  ``cube_size`` term, so the finger settings can no longer be read off the
  observation width; ``cube`` and ``everyShape`` also run their first
  convolution at stride 1, which doubles the spatial-softmax grid off an
  unchanged 120x160 camera. ``profiles.Profile`` gained ``depth_hw`` (asserted
  against the exported graph's ``camera`` input, so a mispaired profile raises
  at load) and ``cnn_stride``; ``scripts/render_sim_depth.py`` takes
  ``--depth-hw``, and ``deploy/live_view.py`` reads the resolution off the
  reference render and opens the camera to match, so the live and rendered
  panels are never two different pictures.
- ``deploy/profiles.py``: what changes between checkpoints -- the observation
  layout, the finger action scale, the finger travel limits and the command
  slew limit -- keyed by the run that produced the weights, and selected from
  the ONNX filename and the graph rather than edited by hand before each run.
  Every one of the four fails silently when wrong, and two of them are now
  corroborated against the file itself (the observation width off the graph, the
  action scale off the metadata) with a mismatch raising at load. ``run.py``
  prints the whole profile before anything moves and takes ``--profile`` for a
  renamed export, where the shape alone cannot say whether the limiter is on.
- ``deploy/`` runs the round-2 students, which read the object's size. Their
  observation is 35 wide, with the cube's HALF EDGE in metres between the
  actions and ``goal_height`` -- both one number, so the swap is a valid vector
  and the layout is settled against the baked normalizer (dim 33 means 0.0250
  and 0.0175, the midpoints of the two arms' trained ranges) rather than assumed.
  ``run.py --cube-mm`` takes the cube's edge in mm, defaults to
  ``calib.CUBE_EDGE_MM``, and is range-checked against the checkpoint's own
  trained sizes, because this observation goes through the same normalizer that
  turned an out-of-range ``goal_height`` into a ``|3641|`` action.
- ``deploy/`` now runs history-stacked students. The observation width is read
  off the ONNX at load, so a checkpoint whose ``student`` group sets
  ``history_length`` deploys with no flag to set, and feeding it a single frame
  raises instead of running. The stacking is TERM-MAJOR -- each term's own
  history is contiguous and the terms follow one another -- which is what
  mjlab's group-level ``history_length`` builds, since it is applied per term.
  Four whole observations laid end to end is the same number of floats in the
  wrong order and nothing downstream can detect it; on a moving sequence the
  frame-major reading takes this checkpoint's ``max|a|`` from 0.42 to 301. The
  history is backfilled from the first frame at reset rather than filled over
  the first four steps, matching ``CircularBuffer``'s first post-reset push.
  ``scripts/wide_onnx_parity.py`` checks the assembled vector against the env's
  own group at either width.
- Added ``reduce="max"`` to ``MetricsTermCfg`` for reporting episode-peak values
  (e.g. peak power, peak contact force) without needing stateful wrapper classes.
- Added ``BuiltinDcMotorActuator``, a native MuJoCo ``<dcmotor>`` wrapper.
  Supports voltage / position / velocity input modes with back-EMF,
  configurable motor constants, and optional integral, slew, inductance,
  thermal, LuGre, and cogging extensions.
- Added ``scale_with_difficulty`` to ``HfRandomUniformTerrainCfg``. When
  enabled, the noise amplitude scales with difficulty (flat at 0, full
  ``noise_range`` at 1) so the terrain progresses in a curriculum. Defaults to
  ``False``, preserving the previous difficulty-independent behavior.
- Added material domain randomization functions for MuJoCo Warp RGB rendering:
  ``dr.mat_emission``, ``dr.mat_specular``, ``dr.mat_shininess``, and
  ``dr.mat_texrepeat``.
- Added teacher-student distillation (DAgger) support: ``RslRlDistillationRunnerCfg``,
  ``RslRlDistillationAlgorithmCfg`` and ``MjlabDistillationRunner``, wired to
  RSL-RL's ``Distillation`` algorithm. ``train.py`` gained ``--teacher-checkpoint``,
  which loads teacher weights from any checkpoint (including a plain PPO run's
  ``actor_state_dict``) while leaving the student freshly initialized.
- Added ``Mjlab-Grasp-TwoFinger-Flexiv-Distill-{Depth,Rgb,Rgbd}``: the two-finger
  grasp env exposing the state teacher's observation and a camera-plus-proprio
  student observation side by side, for distilling the state policy into a
  camera-only one.
- Added the ``goal_height`` manipulation observation, the commanded lift height
  measured from the environment origin. Unlike ``object_to_goal_distance`` it
  carries no object pose, so a policy without privileged state can still be told
  how high to lift.
- ``RslRlDistillationAlgorithmCfg`` defaults to ``gradient_length=1`` and
  ``num_learning_epochs=5`` rather than RSL-RL's ``15`` and ``1``. The
  distillation generator yields one batch per rollout timestep and steps the
  optimizer only every ``gradient_length`` batches, so the stock value divides
  the learning per collected sample by 15 -- enough to leave a CNN student's
  behaviour loss flat for over a thousand iterations. ``gradient_length`` is for
  accumulating BPTT chunks and should stay at 1 unless the student is recurrent.
- ``manipulation_mdp.camera_depth`` gained ``range_noise``, ``dropout_prob``,
  ``patch_prob``, ``patch_size`` and ``scale_err``, modelling the ways a real
  depth stream differs from a rendered one. They are parameters rather than a
  cfg-level ``noise`` term because ordering matters: dropout must be applied
  after the range noise so an invalid pixel reads exactly ``0.0``, which is what
  a D435 writes. ``patch_prob`` drops correlated ``patch_size`` blocks -- the
  failure a median filter cannot remove, unlike i.i.d. dropout. All default to
  zero, i.e. the previous behaviour.
- ``manipulation_mdp.camera_depth`` gained two structured invalid-pixel models
  and dropped ``patch_prob``/``patch_size``. ``shadow_focal_px`` switches on an
  OCCLUSION SHADOW: the function finds background-to-foreground steps along
  each row of the rendered depth and invalidates the
  ``f*B*(1/z_fg - 1/z_bg)`` pixels of background a stereo pair's second imager
  cannot see, always on the same side, at 16 shifted comparisons per frame. It
  reads nothing but depth, so the claw, the cube and the table edge all shadow
  alike. ``blind_gripper_cfg`` / ``blind_object_cfg`` switch on SURFACE
  BLINDING: holes punched into named geoms by a low-resolution random field
  sampled in each object's own bounding box and held for the episode, so a
  blind patch stays on the same part of the object while the object moves.
  ``dropout_prob`` now also accepts a ``(lo, hi)`` band drawn per episode. The
  removed ``patch_prob`` put its blocks at scene-independent positions and
  redrew them every frame, which a policy can average away; both new terms are
  anchored to real geometry instead.
- The wide-claw DR gained object SIZE and LATENCY. ``manipulation_mdp.object_scale``
  scales a body's geoms by one draw on all three axes -- ``dr.geom_size``'s
  ``shared_random`` shares across geoms and still draws each axis separately,
  which turns a cube into a random cuboid -- and carries ``body_inertia`` with
  it by ``s²``, since at fixed mass ``I ~ m·a²`` and a 55 mm cube would
  otherwise keep a 50 mm one's rotational inertia. Size and mass are drawn
  INDEPENDENTLY on purpose: coupling them through a density would teach the
  policy that a bigger object is a heavier one. Latency splits the same way the
  rest of the file does -- command delay on the actuators is physics and needs a
  teacher retrain (``CG_WIDE_CMD_LATENCY_DR``), sensor delay on the student's
  camera and proprioception is perception and does not
  (``CG_WIDE_LATENCY_DR``). Both hold their lag across steps rather than
  redrawing it, because pipeline latency is correlated in time and a per-step
  redraw is zero-mean jitter a policy averages away.
- Added ``Mjlab-Grasp-TwoFingerWide-Flexiv-Success-Dr`` and
  ``-Distill-Depth-Success-Dr``, domain-randomized arms of the wide-claw grasp
  and distillation tasks. Physics randomization (object mass and friction, hand
  link inertia, arm and finger servo gains separately, joint friction and
  damping, encoder bias, bench height, start pose) is shared by both, because
  the teacher labels the student's rollouts and must have trained under the same
  dynamics. Perception randomization (camera position, orientation and field of
  view, plus depth noise and dropout) is added only to the distillation task,
  since the state teacher cannot observe any of it. See
  ``config/flexiv_two_finger_wide/dr_cfg.py`` for the provenance of every range
  and ``scripts/wide_dr_check.py`` for the gate that verifies each one actually
  moves its field.
- Added ``deploy/``, the real-robot runtime for the camera-only grasp student:
  the exported ONNX, the 34-d observation, the D435 depth recipe, and the
  two-finger XC330 gripper over its binary policy protocol. It needs numpy,
  onnxruntime and pyserial, not mjlab or mujoco, so the robot host runs the
  weights the cluster evaluated with none of the simulator on it. The
  sim-radians-to-encoder-counts map is the one thing neither repo records, so it
  is left unset and ``deploy/calibrate_hand.py`` measures it.
- Added ``deploy/run.py --no-hand``, which runs the seven arm joints alone and
  discards the finger half of every action, for checking that the arm reaches
  the grasp pose before the gripper is trusted. The fingers are reported to the
  policy as parked at their home pose, which is what the network sees at the
  start of every sim episode, so the reach is representative and the grasp is
  not. Exactly one of ``--gripper-port`` / ``--no-hand`` is now required outside
  ``--dry-run``: on Linux the port cannot be auto-detected, so a forgotten flag
  would otherwise have meant a silent hand-less run.
- Added ``deploy/run.py --speed``, a single multiplier over ``--arm-max-vel``,
  ``--arm-max-acc`` and ``--arm-max-offset``, and lowered those bring-up
  defaults to 0.25 rad/s, 0.6 rad/s² and 0.08 rad over a 2 s ramp (about a tenth
  of the |dq| p99 the checkpoint runs at in sim). A slower arm lags its target,
  so ``--max-jump`` fires more readily; restore ``--arm-max-vel 2.5
  --arm-max-acc 6.0`` before judging a success rate.
- Added ``deploy/kinematics.py`` and a Cartesian floor guard,
  ``deploy/run.py --floor-z`` / ``--floor-lookahead``. The arm was driven into
  the bench on 2026-08-07: with the speed cap removed the gripper pads dropped
  213 mm in 0.52 s (0.41 m/s) and ``--max-jump``, which watches joint tracking
  error, did not reach its threshold until two steps after the pads were
  already 1 mm under the foam. Tracking error is a lagging indicator of a
  Cartesian problem by construction — it only grows once the arm is failing to
  follow, and an arm that follows a bad target perfectly never trips it.
  ``kinematics.py`` transcribes the arm-to-pad chain from the compiled model
  and evaluates it in numpy (the robot host has no mujoco), checked against
  ``mujoco.mj_kinematics`` to 1e-6 m including on the recorded hardware
  trajectories. The guard compares the measured pad height minus
  ``descent_rate × --floor-lookahead`` against the floor, because height alone
  is also too late: at 0.4 m/s the pads cross the last 20 mm in one step.
  Replayed against the crash, it trips 88 mm above the foam, seven steps before
  the breach. The floor defaults to the foam top, which the policy never
  approaches in sim (minimum pad clearance 22 mm over 64 envs × 300 steps).
- ``deploy/run.py --arm-max-offset`` now defaults to ``tau_max / K_q`` per
  joint, read from the robot, and accepts seven values as well as one. That
  quantity is the training environment's own saturation: the wide-claw model
  sets each joint's ``actfrcrange`` from the sys-id fit to
  ``[123, 123, 64, 64, 39, 39, 39]`` N·m, which is exactly the Rizon 4S's
  ``info().tau_max``, so sim and hardware saturate identically. The previous
  scalar default of 0.08 rad was between 19% (joint1) and 49% (joint5) of that
  — a rate limit inside the policy's closed loop, which produces a limit cycle
  rather than a slower version of the trained motion. ``FlexivArm`` warns when
  the configured offset is under 95% of full authority. The claim in
  ``deploy/arm.py`` and the README that the sim arm had no torque ceiling was
  wrong; it read ``actuator_forcerange`` (0,0 on all seven) instead of the
  joint's ``actfrcrange``.
- Added ``deploy/run.py --record``, which logs every step's measured joints,
  raw action, commanded target and depth frame to an npz and writes it on the
  way out, including after a ``--max-jump`` abort — the aborting step's
  observation is in the file, since that is the one worth looking at.
- Added ``scripts/measure_depth_holes.py``, which measures the D435's missing
  returns against a rendered scene well enough to model them: how much is
  missing and where, which side of a silhouette the occlusion shadow falls on,
  whether its width matches ``f*B*(1/z_fg - 1/z_bg)``, whether a 3x3 median
  removes it, and how blotchy it is. Measured over a nine-pose sweep: at the
  working pose 7.4% of the frame is missing (14.7% of the top third); the claw
  blanks 5-23% of its own silhouette and the fraction swings 4x with viewing
  angle, which is what a specular surface does; the shadow is on the image-LEFT
  of an occluder with 10x the hole density of the right side and stays above
  85% out to 8 px; the observed width implies a 50 +- 6 mm stereo baseline,
  i.e. the D435's nominal one recovered from the image; a 3x3 median removes
  only 9% of it; and 93% of missing pixels sit in blobs of 16 px or more. The
  training env models none of this -- ``patch_prob`` drops 8x8 blocks at random
  positions, resampled every frame -- which is why the numbers are here.
- Added ``scripts/sim_depth.py``, and fixed a 1.325x horizontal magnification in
  every sim-vs-real depth comparison. ``mujoco_warp`` crops a calibrated camera
  to the render's aspect (``render_util.py`` shrinks whichever sensor dimension
  is too large), which is what makes the student's 160x120 render a 640x480
  centre-crop of the D435's 848x480 sensor and what ``deploy/perception.py``
  reproduces on the real stream. ``mujoco.Renderer`` does **not** crop — it fits
  the full 89.42° sensor field into whatever viewport it is given — so
  ``render_sim_depth.py`` rendering 640x480 and calling it "the crop" squashed
  89.42° into pixels meaning 73.53°. Measured by rendering the cube at five
  known bench positions: ``render_u = 0.768 * project_u + 18.0``, against the
  0.755 that 640/848 predicts and the 1.000 a real crop would give. The scripts
  now render at the sensor's own 848x480, where the viewport and sensor aspects
  agree and neither renderer has anything to crop, then apply the same
  centre-crop and box filter as the deploy path; ``tests/test_sim_depth.py``
  asserts the two agree pixel for pixel. Deployment was never affected — only
  the diagnostics — but the artifact was large enough to read as a camera
  extrinsic error: it made the real claw's silhouette 27% wider than the
  rendered one and gave it a spurious position-dependent lateral offset
  swinging from -9 px to +15 px across a joint1 sweep. Corrected, the widths
  agree to 1 px and the offset is a constant +6 px at every pose.
- Added ``deploy/capture_sweep.py`` and ``scripts/fit_camera_pose.py``, which
  measure the camera's pose against the arm across several poses rather than
  one. ``capture_sweep`` screens every pose before anything moves — pad
  clearance over the table, both pads inside the frame, and a cap on
  per-joint travel — refuses the whole sweep if any pose fails rather than
  stopping halfway, and returns to the starting pose on the way out including
  after a fault or Ctrl-C. ``fit_camera_pose`` fits the camera's six numbers to
  the per-pose displacements. It deliberately does **not** fit a joint1 offset:
  substituting ``p_base = R^T p_cam + cam_pos`` into ``dj*(z x p_base)`` splits
  it exactly into a camera rotation plus a camera translation, so joint1 is a
  linear combination of the six camera columns for every pose and no sweep of
  arm poses can separate them — everything joint1 moves is one rigid body, and
  a rigid body cannot say what it is rigid with respect to. The first version
  fitted it anyway and returned a condition number of 2.6e16 and a nonsense
  -8.7°. Separating them needs a feature that does not turn with joint1, which
  is what ``deploy/check_camera.py``'s cube is for. Measured over nine poses:
  the camera's base-frame y is +0.0129 rather than -0.0057, an 18.7 mm lateral
  offset, fitted with a 1.71 mm residual against 11.92 mm for no correction.
  Putting that number back into the model drops the rendered-to-real claw
  offset from 16.5 mm to 3.4 mm at every pose. It is 6.2x the +-3 mm of camera
  position jitter the policy trained under, so it is outside the trained
  distribution and neither script writes it: an extrinsic that changes between
  a diagnosis and a run is worse than a wrong one.
- Added ``deploy/live_view.py`` and ``scripts/render_sim_depth.py``: the live
  D435 frame next to the sim's render of the same scene, a signed difference
  and an overlay, served on localhost. The difference uses two colours only —
  blue where the real frame is nearer, red where it is further, with a pixel
  valid in only one of the two saturating to the same colour it is heading
  towards. The overlay superimposes the two directly, sim on the red channel
  and real on green+blue, so agreement reads as neutral grey and a
  misregistered gripper shows as a red ghost beside a cyan one; a signed
  difference cannot show that, since a shifted object makes a red band and a
  blue band with nothing tying them together. The stats line reports the whole
  image shift that best fits the two over the gripper band, in pixels and in
  millimetres.
- Added ``deploy/capture_sweep.py`` and ``scripts/fit_camera_pose.py``, which
  measure the camera's pose against the arm instead of against a plane. The
  plane fit the current extrinsic came from can only see three of the six
  numbers — pitch, roll and the camera's height over the table — because a
  plane has no yaw and no in-plane position. The sweep drives the arm through
  several poses, screening each one for pad clearance and for both pads staying
  in frame before anything moves and refusing the whole sweep rather than
  stopping halfway, and records a depth frame at each; the fit renders the sim's
  view of the same joint angles and compares. It does not solve for a joint1
  offset, and says why: a joint1 error decomposes exactly into a camera rotation
  plus a camera translation, so that column is a linear combination of the
  camera ones for every pose and the design matrix is singular by construction.
  Separating them needs something that does not turn with joint1. The fit also
  reports the claw's silhouette WIDTH in both frames and refuses to be believed
  when they disagree, because a width mismatch displaces a centroid by an amount
  that depends on where in the frame the claw sits, which reads as a camera
  error that is not there.
- Added ``deploy/check_camera.py``, which projects a cube at a measured
  base-frame position through the sim's camera model and asks the real D435
  where it actually sees it, scoring four hypotheses: as modelled, rotated
  180°, mirrored left-right, mirrored up-down. The depth image is the one
  policy input nothing else cross-checks, and a mis-mounted camera produces a
  plausible frame and a confidently wrong reach. It differences against an
  empty-bench reference rather than looking for the nearest object, since the
  claw and the camera mount are nearer than the cube at the home pose, and its
  default cube position is off-centre because a cube near the image centreline
  cannot separate a mirrored camera from a correct one (2.4 px at the scene's
  nominal spawn, against 33 px at the default) — a placement that cannot
  separate the hypotheses is reported as such instead of passing.
- Added a periodic progress line to ``deploy/run.py`` (``--print-every``)
  reporting the measured arm pose, the TCP position where RDK provides one, the
  worst ``|target - measured|`` and the achieved loop rate, plus a per-stage
  timing breakdown (camera / sensors / policy / command) at the end of every
  run, printed after an abort too.
- ``deploy/perception.py`` now streams the D435 at 90 fps (``--camera-fps``) and
  reads it on a background thread, so ``read()`` returns the latest frame
  instead of blocking. At the previous 30 fps default ``wait_for_frames`` put a
  33.3 ms floor under a 20 ms budget, measured on the bench at 44 ms per step —
  23 Hz against the 50 Hz the policy was trained at, with every step late. A
  blocking read also aliases the loop to the sensor's phase even at 90 fps.
  ``--camera-blocking`` restores the old behaviour for comparison, and the run
  summary counts frames served twice so the staleness the thread trades for is
  measured rather than assumed.
- Added ``scripts/wide_dr_ablate.py``, which re-scores one trained checkpoint
  under progressively narrower DR envs to attribute a score drop to a specific
  randomization rather than to "perception DR" as a lump. The switches it flips
  (``CG_WIDE_CAM_DR``, ``CG_WIDE_CAM_POS_DR``, ``CG_WIDE_CAM_ROT_DR``,
  ``CG_WIDE_CAM_POS_AXES``, ``CG_WIDE_CAM_YAW_DEG``, ``CG_WIDE_DEPTH_DR``) are
  read at build time, so an existing checkpoint needs no retraining.
- Added ``scripts/wide_onnx_parity.py``, which drives the env with the torch
  student while feeding the same observation through onnxruntime, so "the ONNX
  is the evaluated policy" is checked rather than assumed. It also checks that
  ``deploy/`` reassembles the student observation bit for bit.
- Added ``--surface-z`` to ``deploy/check_camera.py``, the height of whatever
  the cube is resting on. The resting height used to be hardcoded to the foam
  top, so running the check on a bench with the foam taken off put the predicted
  cube 50 mm high — 11.9 px at bench range, against the check's own 8 px
  tolerance, i.e. a confident report of a camera fault that is not there. The
  default is unchanged (foam on, the bench the policy was trained against);
  pass ``-0.015`` for the bare table.
- Added the deployment half of the training env's
  ``SmoothedJointPositionAction``. ``StudentPolicy`` now keeps the published
  command as state and bounds how fast it may move — a hard slew cap
  (``Profile.rate_limit``) and a first-order lag (``Profile.ema_tau``) — seeded
  from the measured joints at ``reset()`` exactly as the sim seeds from the
  post-reset pose. Both are per-checkpoint and live in ``deploy/profiles.py``,
  not in ``calib.py``, because they are the policy's property rather than the
  bench's. Nothing can check the pairing for you — the ONNX metadata does not
  carry the action term — so ``run.py`` prints the limit at startup and reports,
  at the end of a live run, how far past it the policy was asking, which is what
  a mismatch looks like.
- Added ``deploy/calibrate_finger_scale.py``, which measures the finger joints'
  counts-per-radian on the bench instead of assuming it. ``calib.COUNTS_PER_RAD``
  was 651.9 = 4096/2pi, i.e. the assumption that the joint turns 1:1 with the
  motor shaft; measured against the jaw's outside width it is **1028 ± 9**, a
  1.58:1 linkage. Under the old value a commanded finger angle reached only 63%
  of itself while the encoder read back 58% high, both making the claw narrower
  than the policy believed: at the home pose it held a 64.2 mm jaw where the
  policy thought it had 71.8, and a 50 mm cube presents 70.7 mm across its
  diagonal, so for 55% of the trained (full-circle) yaws the cube could not enter
  the jaw at all. Hardware grasp rate went from 2/20 to 3/4 of the runs that ran
  past the approach. ``COUNTS_AT_ZERO_RAD`` stays 0 and is now confirmed rather
  than assumed: ``outer width - inner gap`` measures exactly twice one finger's
  thickness, which only holds when the fingers are parallel.
- Added ``deploy/run.py --max-action``, aborting on a raw network output beyond
  ``calib.MAX_ABS_ACTION`` (default 15; trained actions run to about |5| and the
  observed blow-up was |600|). Under a slew limit this is the only guard that
  sees such a step in time: the limiter publishes at most 20 mrad of new arm
  motion per control step, so a |600| action and a |3| action produce the same
  first command and ``--max-jump``, which watches tracking error, only diverges
  several steps later. ``StudentPolicy`` also clamps the published target to
  ``calib.JOINT_LIMITS``, so a large action cannot command past a hard stop.

Changed
^^^^^^^

- Every rendered clip and still now resolves its output through
  ``scripts/tools/video_out.py`` and lands under ``videos/``. Renders had been
  going to ``/work/yiboc``, ``/work/yiboc/videos``, ``/work/yiboc/renders`` and
  ``~/cg-logs`` at the same time, so 130 MB of clips sat outside the repo on a
  filesystem that is not backed up with it, and "the video of the best policy"
  had no single place to be. A relative path resolves against ``videos/`` --
  ``render_student.py R2-Small-SB`` and ``render_student.py videos/R2-Small-SB``
  mean the same thing -- and an absolute path elsewhere still works but warns,
  because a one-off render to ``/tmp`` is legitimate and scattering the archive
  again by accident is not.
- Removed 32 retired diagnostic scripts, all of them written against the OLD
  two-finger claw and none of them reachable from the wide-claw line. The five
  still cited from live code are kept (``diag_joint7_use``, ``diag_student_view``
  and ``diag_success_metric`` from the wide config, ``diag_wrist_sign`` from
  ``manipulation/mdp/actions.py``, ``diag_teacher_actions`` from this file);
  where a deleted script was the only citation for a measured constant in the
  retired old-claw config, the citation is marked ``retired`` rather than left
  pointing at a file that is gone. The measurements themselves are in the
  comments and are untouched.
- ``deploy/`` now flies the round-3/4 checkpoints behind a 0.09 s command EMA
  (``square-varh``, ``cube`` and ``everyshape``), and ``deploy.run --ema-tau``
  overrides it at the bench. Unlike the slew limit this does not have to match
  training: the action term is outside the ONNX graph, so the lag can be added
  to a finished policy, and the ``-SlowEma`` task ids priced exactly that in sim
  over 2048 episodes -- +1.13, -0.68, -0.64 and +6.01 points against each arm's
  own unsmoothed baseline, i.e. free within the eval's own noise on three of
  four. Tightening the slew cap instead costs 2.2 to 10.4 points on the same
  checkpoints. What the lag buys is the PEAK: at 50 Hz a step of the command
  moves 20% of the way on the first control step, which is the half of "the
  fingers snap shut" that a rate cap does not fix. ``square-fixh`` is left
  unsmoothed because it was not in the sweep.
- ``deploy/`` now applies the same command smoothing the policy trained under,
  read off the selected ``profiles.Profile``. It must match the checkpoint: a
  policy trained behind a slew cap learns to lean on it, asking for a target rate
  up to 17x (and on the tightest arm 50x) what the limiter publishes, so running
  such a checkpoint without the limiter is a full-speed command rather than a
  slightly faster one.
- A gripper fault now names itself and the motor it came from. ``deploy/hand.py``
  raised ``gripper fault 2 (see PROTOCOL.md Fault enum)`` and threw away the OBS
  that arrived with it, though every OBS carries per-motor ``online``, ``alert``,
  current and temperature. The fault name, that table and the fact that the
  fault LATCHES (the controller has to be power-cycled) are all in the message
  now, because ``2`` (WATCHDOG, a bus or connector problem) and ``5``
  (MOTOR_REBOOTED, a brownout) want opposite responses on the bench.
- ``deploy.home_arm`` now moves the ARM first and ramps the fingers afterwards.
  The finger ramp is open-loop, timed and uninterruptible -- no planner, no
  collision check -- so it belongs where the hand is in free space rather than
  wherever the previous run parked the arm; homing the jaw opens it to 86.9 mm
  of pad separation, sweeping each pad outward through whatever is beside it.
  The hand is left untouched when the arm fails to reach home, since that is
  exactly the case where its pose is unknown. The trade is that the arm now
  traverses with the fingers as the last run left them, so open the gripper
  first if it ended holding the cube.
- Bumped ``rsl-rl-lib`` from 5.2.0 to 5.4.0.
- Curriculum-mode terrain difficulty is now deterministic across rows
  and reaches the configured ``difficulty_range`` endpoints
  (:issue:`1027`).
- Heightfield terrains now color by absolute height with a diverging palette
  (cool below the ground plane, green at ground level, warm above) on a fixed
  scale, replacing the per-patch normalization. Color is now consistent across
  terrains, and low-amplitude terrain such as ``random_rough`` reads as gently
  tinted ground instead of high-contrast noise.
- ``BoxNestedRingsTerrainCfg`` now builds uniform-height concentric ridges
  whose separating gaps widen with difficulty, replacing the random per-ring
  heights. Rings are colored by height (like the other terrains) and the outer
  border matches the ring height.
- Terrain generation no longer prints timing information to stdout.

Fixed
^^^^^

- ``scripts/wide_scene_shot.py`` no longer asserts at env build time on a task
  with ``percept_dr``. It set the D435's ``data_types`` to ``("rgb", "depth")``,
  which drops the ``segmentation`` stream that ``camera_depth``'s surface
  blinding needs; it now ADDS ``rgb`` to whatever the task asked for, the same
  way ``render_student.py`` does.
- ``scripts/wide_spawn_gate.py`` now FAILS a spawn box containing a pose the arm
  cannot reach at all. A pose whose IK did not converge was skipped, so it never
  entered the worst-case reach number and a box could pass on the strength of the
  poses that did converge -- the reverse of what the gate is for. It also no
  longer parses ``sys.argv`` at import time, which made the first argument of any
  script importing it get read as a box half-width.
- The fine-tuned actor's ``EmpiricalNormalization`` is now frozen at the count
  it carried out of distillation. Its statistics live in buffers, so
  ``requires_grad_(False)`` does not reach them and PPO updated them from every
  rollout regardless of the critic warm-up. The student observation feeds its
  own raw action back as dims 22-32, the rate limiter leaves that action's
  magnitude without gradient, and the resulting drift inflated those eleven
  standard deviations by up to 13× while joint position and velocity did not
  move at all. Normalizing divides by that std, so the policy's action-feedback
  channel was flattened toward zero under a warm-started policy that depends on
  it. Measured: a run healthy for 60 iterations (better ``lift_hold`` than the
  distilled student it started from) lost the cube at iteration 65 and never
  recovered, finishing at 0.0% deployed success; the same configuration with the
  normalizer pinned reaches 99.2%. The distilled statistics are the ones the
  student's weights were fitted against, so they are what to keep. The critic's
  normalizer is deliberately left learning -- it starts from nothing. The
  student's ``actions`` observation is now clipped to ±5 as well, which is a
  no-op on the trained range and is required *because* of the freeze: with the
  std pinned, a runaway action would otherwise reach the network unattenuated.
- ``scripts/render_student.py`` now picks which weights to load by inspecting
  the checkpoint rather than assuming ``student_state_dict``. Asking a PPO
  checkpoint -- which is what fine-tuning a student produces -- for a key it
  does not have was a silent no-op, so the network stayed at its random
  initialization and a 99.2% policy rendered as one that never lifts anything.
  An unrecognized checkpoint now raises. The script also prints the failure env
  indices alongside the success ones, without which ``env=N`` cannot be pointed
  at a failing episode, and its docstring records that near the grasp boundary
  a re-render does not drift by the documented 25 mm but flips outright.
- ``scripts/render_student.py`` no longer fails to build the wide-claw env. It
  added the RGB video panel by overwriting the camera's ``data_types``, which
  dropped the ``segmentation`` buffer that ``camera_depth``'s surface-blinding
  randomization requires; it now appends to the tuple instead. It also gained
  ``n=`` to widen the selection pool, needed because a policy that fails 2% of
  the time yields too few failure candidates in 64 envs to survive a re-render.
- ``scripts/diag_teacher_actions.py`` now reads the actor off a PPO runner
  instead of ``runner.alg.teacher``, which only exists on the distillation
  runner and made the script raise on every teacher checkpoint.
- The wide-claw D435 no longer renders its own housing. ``enabled_geom_groups``
  dropped from the default ``(0, 1, 2)`` to ``(0, 1)``; on this model group 2
  holds the D435i's nine visual meshes and nothing else. The lens sits inside
  that housing -- ray-cast from the camera at the home pose, the mesh is
  0.21 mm away along the optical axis -- and the clearance is new: before the
  stage-B extrinsic correction moved the camera 19.9 mm, the optical axis left
  the housing without hitting anything. Backface culling hid it at the nominal
  pose, but ``dr_cam_pos`` jitters the lens by ±3 mm, fourteen times that
  clearance, and 5 of 64 environments rendered a frame that was ENTIRELY the
  inside of the housing (depth mean 0.0007–0.0086 against a healthy 0.48);
  0 of 64 with the jitter off, and 0 of 64 after the fix. Those environments
  handed the student a blank near-clip image while the teacher labelled them
  normally, so DAgger was fitting the teacher's action to no scene at all.
- ``scripts/wide_dr_check.py`` now resets the student env before stepping it.
  A freshly built env has not run its reset events or resampled its command, so
  the arm sat at ``qpos = 0`` and the cube had never been placed: every depth
  statistic in the script was measured on a perfectly reasonable-looking picture
  of the bench with neither the claw nor the cube in it. Range noise, dropout
  and the occlusion shadow all work off the table and the wall and never
  noticed. A new check counts the pixels segmentation reports for each blinding
  target, so an empty target can no longer be mistaken for a zero rate.
- Fixed the deployed finger clamp, which used the URDF's ±1.6 rad on all four
  finger joints instead of the travel the training env sets (proximal
  −0.0754…+1.60 and distal −1.60…+0.10 for the left finger, mirrored on the
  right). In sim that range is a physical stop the policy is trained inside; on
  hardware nothing stops the command, so a full-close action could be published
  well past the drive-through limit. The clamp is per-checkpoint, since round 2
  opens the proximal bound to −0.32 rad to pinch a 15 mm object.
- Fixed ``deploy/hand.py``'s gripper checkout path, which was one machine's
  absolute ``/home/yiboc/gripper/firmware/host``. It is now derived from this
  repo's own location, so any host with the two repos side by side works
  unconfigured, and ``$GRIPPER_HOST_DIR`` still overrides. The old default
  failed at ``home_arm``, i.e. after you had already walked to the robot.
- Fixed the wide-claw ``scene_cam`` extrinsic, which put the camera 19.9 mm to
  the image-left of where it actually sits, so the gripper landed in a visibly
  different place in a sim render than in the live D435 frame. The CAD chain was
  only ever corrected by a table-plane fit, and a plane cannot see rotation about
  itself or translation within itself — the two in-plane translation components
  had never been measured. They now are, from a nine-pose sweep
  (``deploy/capture_sweep.py`` + ``scripts/fit_camera_pose.py``) spanning 31.6°
  of joint1: position alone fits the set to 1.79 mm against 3.44 mm for rotation
  alone and 11.69 mm uncorrected, so it is one lateral offset. Applied as
  ``arm_cfg.CALIB_CAM_OFFSET`` and ``check_camera.CAM_POS_BASE``, which are the
  same measurement in two frames and must move together. Re-fitting the same
  sweep afterwards leaves 1.04 mm with nothing further to correct, and
  ``deploy/live_view``'s best-fit shift of real onto sim goes from du +7 px
  (19 mm sideways) to du +0 px. The offset is 6.6× ``dr_cfg._CAM_POS_JITTER``,
  so the current student checkpoint was trained against the old camera and does
  not inherit the fix; it applies from the next retrain. A joint1 zero-offset
  would look identical to the sweep — it decomposes exactly into a camera
  rotation plus a camera translation, so it is singular by construction — and
  was ruled out separately with the cube, which does not turn with joint1: at
  (0.508, 0.102) the two hypotheses predict pixels 5.1 px apart and the cube
  landed 1.79 px from the camera prediction against 4.30 px from the joint1
  one, with a left-right mirror 44.5 px away. The correction is therefore right
  for the table and the cube as well as for the arm.
- Fixed dependency resolution failing with a 404 on ``mujoco``. The pin resolved
  to a ``py.mujoco.org`` nightly, and that index garbage-collects old builds, so
  the locked ``3.8.1.dev907177387`` stopped existing (it now serves only 3.10 and
  3.11, neither of which satisfies the ``~=3.8.0`` floor). ``mujoco`` now comes
  from PyPI as stable ``3.8.1``, the release of that same dev build. This has to
  be done with ``override-dependencies``: ``mujoco-warp``'s own pyproject sources
  ``mujoco`` from the nightly index and uv honours a git dependency's sources, so
  merely dropping the entry from ``[tool.uv.sources]`` leaves it there and adding
  a competing index fails with "conflicting indexes for package ``mujoco``".
- Fixed domain randomization events that target different ``axes`` of the same
  model field (e.g. two ``dr.geom_size`` events scaling axis 0 and axis 1
  separately) silently clobbering each other. Each event now writes back only
  the axes it targeted, so per-axis events compose (:issue:`1042`).
- Regenerated the bundled MuJoCo type stubs, which had drifted from the
  installed mujoco version. CI now regenerates them and fails if they are
  stale, so they stay in sync going forward. Run ``make stubs`` to update them
  (:issue:`1048`).
- Fixed ``select_gpus`` crashing when ``CUDA_VISIBLE_DEVICES`` contains MIG UUIDs instead of numeric indices.
- Fixed pyramid-stairs terrains (``BoxPyramidStairsTerrainCfg``,
  ``BoxInvertedPyramidStairsTerrainCfg``, and ``BoxOpenStairsTerrainCfg``)
  leaving an empty, geometry-free border at difficulty 0, where the step
  height collapses to zero. The flat border frame is now always generated as
  solid geometry flush with the ground (:issue:`1033`).
- Fixed ``HfPerlinNoiseTerrainCfg`` failing to compile at difficulty 0, where
  the target height collapses to zero and MuJoCo rejects the non-positive
  heightfield size.
- Fixed ``BoxRandomGridTerrainCfg`` producing NaN colors (and failing to build)
  at difficulty 0, where the grid height is zero and the color normalization
  divided by zero.
- Fixed the center platform z-fighting with surrounding geometry in
  ``BoxRandomGridTerrainCfg`` (grid cells were left underneath the platform) and
  ``BoxRandomSpreadTerrainCfg`` (the platform duplicated the floor surface).
- Fixed ``BoxNarrowBeamsTerrainCfg`` square platform corners protruding between
  the beams at high difficulty; the platform now shrinks to stay within the
  beams' angular coverage.
- Fixed ``BoxSteppingStonesTerrainCfg`` reconfiguring abruptly at a difficulty
  threshold, where the stone grid re-tiled as its spacing crossed an integer
  boundary, and leaving an oversized gap around the center platform. The grid is
  now difficulty-independent and the platform snaps to it as a clean island.
- Fixed ``train --video``, ``play``, and ``demo`` crashing with ``OpenGL
  platform library not loaded`` on headless Linux hosts that don't pre-set
  ``MUJOCO_GL``. The default is now applied in ``mjlab/__init__.py`` (Linux
  only) so it takes effect before mujoco's GL backend selection runs.

Version 1.4.0 (May 26, 2026)
----------------------------

Added
^^^^^

- Added ``BuiltinPdActuator``, the implicit-integration version of
  ``IdealPdActuator``. Same interface (position + velocity targets,
  kp/kd gains), but expresses the PD as native MuJoCo ``<position>``
  and ``<velocity>`` elements so the ``implicit`` / ``implicitfast``
  integrators include the kp/kd derivatives in their velocity update.
  The actuator stays stable at gain/timestep combinations where
  explicit Python PD would diverge, which matters when you want to
  run a real motor's stiff on-board PD gains in sim. ``effort_limit``
  is enforced as a sum-clamp on the two PD terms via
  ``jnt_actfrcrange`` (or ``tendon_actfrcrange``). Supported by
  ``dr.pd_gains`` and ``dr.effort_limits``.
- Added ``mdp.projected_gravity_from_sensor``, an observation that derives
  projected gravity from a ``framezaxis`` up-vector sensor (negated) rather
  than from the root body orientation. Unlike ``mdp.projected_gravity``, it
  reflects the sensor's site frame, so it can observe IMU mounting domain
  randomization (e.g. via ``dr.site_quat``). Go1 and G1 ship an
  ``imu_upvector`` sensor for this.
- Added ``DebugVisualizer.add_box`` for drawing an axis-oriented box
  primitive, mirroring ``add_ellipsoid``. Supported by both the native
  and Viser viewers. ``size`` is the box half-extents (:issue:`992`).
- Added ``--log-root`` CLI option to ``train``, ``play``, and ``evaluate``
  scripts for choosing where training logs are stored. Defaults to
  ``logs/rsl_rl`` (unchanged behavior). Useful for directing outputs to a
  scratch disk or shared mount.
- ``RewardManager``, ``TerminationManager``, and ``MetricsManager`` now
  validate that every term function returns a tensor of shape
  ``(num_envs,)`` when evaluated, raising a clear ``ValueError``
  naming the offending term instead of silently broadcasting or crashing
  with an opaque error later during training.
- Added ``ContactSensor.primary_names`` property to expose the resolved
  primary names in the order they appear along the per-contact axis of the
  output tensors. This makes it possible to map a contact-data column back
  to the primary it belongs to (:issue:`914`).
- Added per-world mesh variant support via ``VariantEntityCfg``. Each
  world in a batched simulation can now use a different mesh asset for
  the same logical entity (e.g. world 0 holds a cube, world 1 a
  sphere). Variants are passed as a ``dict[str, Callable]`` of named
  spec callables; the optional ``assignment`` field controls how worlds
  map to variants and accepts ``None`` (uniform), a ``dict[str, float]``
  of per-variant weights, or a custom ``Callable[[int], Sequence[int]]``.
  Mesh-derived constants (collision bounds, body inertials, subtree
  mass, inverse weights) are compiled per-variant and stored as
  per-world arrays in the Warp model, so domain randomization, the
  native viewer, the offscreen renderer, and the Viser viewer all pick
  up the variant assignment automatically. Variants must share the
  same kinematic structure (same bodies, joints, joint types); only
  mesh geoms may differ. Assignment is fixed at simulation init. See
  :ref:`heterogeneous_worlds` for usage. With help from @XiangruiJiang.
- Per-world mesh variants now support per-variant materials and textures.
  Each variant can reference its own named material, which is automatically
  prefixed and scattered via ``geom_matid`` alongside the existing
  ``geom_dataid`` table. Variants without a material get ``matid = -1``.
  Contribution by @omarrayyann.

Changed
^^^^^^^

- ``Entity`` now raises a clear error at construction when its spec contains
  more than one freejoint. An entity models a single system rooted at one
  body, so it has at most one freejoint; a second one was previously accepted
  silently and only surfaced later as a cryptic shape mismatch when writing
  root state. Model each detached floating body as its own entry in
  ``SceneCfg.entities`` instead.
- Changed ``compute_root_relative_mpkpe`` to re-anchor the reference to the
  robot's root each step, removing yaw drift as well as translation so it
  measures intrinsic body pose error.
- Changed ``compute_joint_velocity_error`` from an L2 norm to a per-joint
  RMS, so it no longer scales with the number of joints.
- Bumped ``mujoco`` to 3.8 and ``mujoco-warp`` to 3.8.0. The ``multiccd``
  enable flag was removed in mujoco 3.8 (it became default-on), so configs
  that listed ``"multiccd"`` in ``MujocoCfg.enableflags`` need to drop it.
- Camera segmentation now matches ``mujoco_warp``'s typed segmentation
  output. ``CameraSensorData.segmentation`` stores ``(object_id,
  object_type)`` pairs in shape ``[B, H, W, 2]`` instead of the previous
  legacy geom-id-only layout. Contribution by @tkelestemur.
- Sped up ``RayCaster`` post-processing by removing boolean-mask indexing
  operations and replacing them with ``masked_fill_`` plus a clamped-distance
  formulation of ``hit_pos_w`` that places misses at the world origin. This
  removes all CUDA syncs from the ray post-process, letting the CPU thread
  proceed while GPU-based sensing runs. Contribution by @bd-pdomanico.
- Bumped ``rsl-rl-lib`` from 5.0.1 to 5.2.0. This brings ``torch.compile`` support for
  PPO and Distillation, and optional std clamping and constant std in
  ``GaussianDistribution``. No code changes required on the mjlab side.
- ``TerrainEntityCfg`` debug visualization sites (environment origins,
  terrain origins, flat patches) are now off by default. Set
  ``debug_vis=True`` to re-enable them. The sites inflated ``nsite`` and
  caused a measurable slowdown in the per-step ``site_local_to_global``
  kernel (:issue:`942`).
- Task package load failures during ``mjlab`` import now print the full
  traceback (and the entry point's module path) to ``stderr`` instead of
  just the exception message, making it easier to pinpoint the source of
  import errors when running commands like ``list-envs`` (:issue:`910`).
  Contribution by @saikishor.
- Clarified ``ContactSensor`` shape conventions: per-contact fields
  (``found``, ``force``, ``torque``, ``dist``, ``pos``, ``normal``,
  ``tangent``) have shape ``[B, P * num_slots, ...]`` while per-primary
  air-time fields (``current_air_time``, ``last_air_time``,
  ``current_contact_time``, ``last_contact_time``) have shape ``[B, P]``,
  where ``P`` is the number of resolved primaries (:issue:`914`).
- Event functions now share a single ``resolve_env_ids`` helper to expand
  ``env_ids=None`` to all environments, replacing five copies of the same
  guard. ``push_by_setting_velocity`` and ``apply_external_force_torque``
  accept ``env_ids=None`` too, so they work as global-time interval terms.
  Documented when to use ``apply_external_force_torque`` (a constant,
  self-managed wrench) versus ``apply_body_impulse`` (transient, automatic
  impulses) versus ``push_by_setting_velocity`` (an instantaneous velocity
  kick).

Fixed
^^^^^

- Removed use of deprecated ``warp-lang`` symbols (``wp.context.runtime``
  and ``wp.context.Device``) that were dropped in newer ``warp-lang``
  releases, causing ``AttributeError: module 'warp' has no attribute
  'context'`` at import/runtime. mjlab now uses
  ``wp.get_cuda_driver_version()`` and ``wp.Device`` instead
  (:issue:`967`). Contribution by @rdeits.
- Fixed the tracking ``evaluate`` script scoring each metric against the
  next motion frame; the reference is now snapshotted before each step to
  match the reward.
- Fixed the tracking end-effector metrics silently scoring zero for an
  unknown body name; they now raise ``ValueError``.
- Fixed ``compute_mpkpe`` measuring root-relative instead of global error;
  it now uses the global reference ``body_pos_w`` (:issue:`1006`).
- Fixed heavy flicker in offscreen training videos on rough-terrain tasks.
  The renderer recomputed its context "neighbor" robots every frame from
  ``env_origins``, which the terrain curriculum mutates on reset, so the
  neighbor set kept changing and robots popped in and out. The neighbor
  set is now computed once and cached (:issue:`979`).
- Fixed command delay only applying to an actuator's position target.
  ``IdealPdActuator`` and ``DcMotorActuator`` also use velocity and effort, which
  arrived undelayed and out of sync; all command targets now share one delay.
  Zero-reference setups are unaffected.
- Fixed duplicate random seeds across nodes in multi-node training. The
  per-process seed offset in ``scripts/train.py`` now uses the global
  ``RANK`` instead of ``LOCAL_RANK``. Contribution by @bd-pdomanico.
- Fixed ``apply_body_impulse`` firing an impulse on the very first step (and
  the first step after every reset) instead of starting with a cooldown as
  documented. The cooldown is now sampled lazily on the first call so impulse
  timing is decorrelated from episode resets (:issue:`973`).
- Fixed ``dr.pd_gains`` and ``dr.effort_limits`` silently no-oping when
  passed an ``Operation`` object (e.g. ``dr.scale``) instead of a string.
  Both functions now accept ``Operation | str`` like every other DR event
  and raise ``ValueError`` for unsupported operations (:issue:`971`).
- Fixed ``ContactSensor`` with ``global_frame=True`` and
  ``reduce`` ∈ {``"none"``, ``"mindist"``, ``"maxforce"``} producing forces
  rotated onto the wrong axis. The contact-frame→world rotation matrix had
  its columns ordered ``[tangent, tangent2, normal]`` instead of
  ``[normal, tangent, tangent2]``, projecting the normal-force component
  onto a tangent direction. Contribution by @bd-pdomanico.
- Fixed ``extras["log"]`` entries written by reward terms (e.g. ``Metrics/*``
  values in velocity tasks) being silently discarded on any step where at
  least one environment resets. ``_reset_idx`` was clearing the dict after
  ``reward_manager.compute()`` had already populated it. The clear now
  happens at the top of ``step()`` and ``reset()`` so that all entries
  survive (:issue:`957`).
- Fixed ``ContactSensor.compute_first_contact`` and ``compute_first_air``
  occasionally missing events when a contact began or ended right at the
  last physics substep of a control step. ``current_contact_time`` /
  ``current_air_time`` accumulate in float32 and can drift a few ULPs past
  ``dt``, but the default ``abs_tol`` of ``1e-8`` sat at the noise floor
  and rejected the comparison. Raised the default to ``1e-6``, which stays
  well below typical control ``dt`` while comfortably covering float32
  accumulation noise (:issue:`933`). Contribution by @paLeziart.
- Fixed ``out_of_terrain_bounds`` using stale terrain dimensions. It read
  ``TerrainGeneratorCfg.num_cols`` directly, which is ignored in curriculum
  mode (the generator uses ``len(sub_terrains)`` columns instead), and it
  did not account for ``border_width``. The termination now reads the
  effective grid shape from ``terrain.terrain_origins`` and includes the
  border in the footprint, so robots no longer reset while still on valid
  terrain (or fail to reset after running off it) (:issue:`923`).
- ``ObservationManager`` now skips observation groups that end up with
  zero active terms (e.g. all terms set to ``None``) with a log message,
  instead of crashing later in ``torch.stack``/``torch.cat``. This lets
  a shared runner config define groups that become empty under certain
  runtime flags (e.g. model-specific terms all disabled for one variant).
  The whole group can still be set to ``None`` to disable it explicitly.
- Fixed a runtime broadcast error in ``ContactSensor`` when combining
  ``num_slots > 1`` with ``track_air_time=True`` and more than one primary.
  Air-time tracking now reduces ``found`` across slots so that a primary is
  considered in contact when any of its slots reports a match (:issue:`914`).
- Updated the ``create_new_task.ipynb`` Colab tutorial to import
  ``XmlActuatorCfg`` instead of the removed ``XmlVelocityActuatorCfg``.
  Added a regression test (``tests/test_notebooks.py``) that parses each
  notebook cell and verifies that every ``from mjlab... import X``
  reference resolves, so future renames in the mjlab public API can't
  silently rot the tutorials (:issue:`913`).
- Fixed ``ObservationManager`` silently sharing a single ``NoiseModelCfg``
  instance across observation groups that declared terms with the same
  name. ``_group_obs_class_instances`` was keyed by term name alone, so
  the last group processed in ``_prepare_terms`` overwrote earlier
  groups' instances. Symptoms included the wrong noise config being
  applied, shared per-episode state for ``NoiseModelWithAdditiveBias``
  (e.g. bias drawn from the wrong ``bias_noise_cfg``), and missed
  ``reset()`` calls for overwritten instances. Instances are now keyed
  by ``(group_name, term_name)`` so each group owns its own noise model.
- Fixed ``CurriculumManager.get_active_iterable_terms`` raising
  ``TypeError`` when a term's state was a dict. The dict branch indexed
  the output list by term name instead of appending to the local ``data``
  list. No in-tree caller currently invokes this method, so the bug was
  latent.

Version 1.3.0 (April 14, 2026)
------------------------------

Added
^^^^^

- Added ``ManagerBasedRlEnvCfg.auto_reset`` flag. When ``True`` (default),
  ``step()`` continues to reset done environments in place and returns the
  post-reset observation. When ``False``, ``step()`` skips the reset block
  and returns the terminal observation directly; the caller must call
  ``reset(env_ids=...)`` for done environments before the next ``step()``
  or a ``RuntimeError`` is raised. Enables access to the true terminal
  state for algorithms that need it. Note that mjlab's bundled ``train.py``
  uses rsl_rl's ``OnPolicyRunner``, which does not drive manual resets, so
  ``auto_reset=False`` is intended for custom training loops (:issue:`900`).
- Added ``ActuatorCfg.viscous_damping`` for passive velocity proportional
  damping (``f = -b·v``), distinct from the PD derivative gain ``damping``
  used by position and velocity actuators. Maps to ``<joint damping>`` for
  JOINT transmission and ``<tendon damping>`` for TENDON transmission.
  Defaults to ``None`` (preserves the XML value).
- Added :class:`~mjlab.managers.RecorderManager` for logging observations,
  actions, or arbitrary environment data during rollouts. Implement a
  :class:`~mjlab.managers.RecorderTerm` subclass and register it in the
  ``recorders`` dict on ``ManagerBasedRlEnvCfg``. The manager provides
  ``record_pre_reset``, ``record_post_reset``, and ``record_post_step``
  lifecycle hooks with no opinion on how data is stored.
- Added :func:`~mjlab.envs.mdp.curriculums.termination_curriculum` for
  scheduling changes to termination term parameters during training,
  matching the existing ``reward_curriculum`` pattern. Both now share a
  single internal engine with init-time validation of stage ordering,
  field existence, and param keys.
- Added ``reduce`` field to ``MetricsTermCfg``. Setting ``reduce="last"``
  reports the value from the final step of the episode rather than the
  episode mean, which is useful for binary success metrics.
- Added :class:`~mjlab.envs.mdp.actions.RelativeJointPositionAction` for
  joint position control relative to the current configuration. The target is
  ``current_pos + action * scale``, so a zero action holds the current
  configuration rather than commanding the default pose.
- Added :func:`~mjlab.envs.mdp.dr.pair_friction` for randomizing geom-pair
  friction overrides (``pair_friction`` in ``mjModel``), with an
  ``isotropic=True`` option that mirrors the symmetric tangent and roll
  axes so single-axis randomization does not leave the paired axis stale.
- Added ``STAIRS_TERRAINS_CFG`` terrain preset for progressive stair
  curriculum training and ``@terrain_preset`` decorator for composing
  terrain configurations from reusable presets.
- Added cartpole balance and swingup tasks (``Mjlab-Cartpole-Balance`` and
  ``Mjlab-Cartpole-Swingup``) with a :ref:`tutorial <tutorial-cartpole>`
  that walks through building an environment from scratch.
- Added :ref:`motion imitation <motion-imitation>` documentation with
  preprocessing instructions. The README now links here instead of the
  BeyondMimic repository, which produced incompatible NPZ files when used
  with mjlab (:issue:`777`).
- Added ``margin``, ``gap``, and ``solmix`` fields to ``CollisionCfg``
  for per geom contact parameter configuration (:issue:`766`).
- NaN guard now captures mocap body poses (``mocap_pos``, ``mocap_quat``)
  when the model has mocap bodies, enabling full state reconstruction in
  the dump viewer for fixed-base entities.
- Implemented ``ActionTermCfg.clip`` for clamping processed actions after
  scale and offset (:issue:`771`).
- Added ``qfrc_actuator`` and ``qfrc_external`` generalized force accessors
  to ``EntityData``. ``qfrc_actuator`` gives actuator forces in joint space
  (projected through the transmission). ``qfrc_external`` recovers the
  generalized force from body external wrenches (``xfrc_applied``)
  (:issue:`776`).
- Added ``RewardBarPanel`` to the Viser viewer, showing horizontal bars for
  each reward term with a running mean over ~1 second (:issue:`800`).
- Added ``per_substep`` flag to ``MetricsTermCfg`` for evaluating metrics
  once per physics substep inside the decimation loop. The per substep
  values are averaged within each environment step, so episode averages
  remain comparable to regular per step metrics.
- Added ``project-instinct/InstinctMJ`` to the research page's list of
  projects built on mjlab.
- Added a Checkpoints tab to the Viser play viewer for hot-swapping
  checkpoints without restarting. Works with local directories and W&B
  runs (:issue:`751`). Contribution by @omarrayyann.
- Added ``"segmentation"`` camera data type for per-pixel geom ID output
  alongside RGB and depth, and a multi-cube goal-conditioned lifting task
  (``Mjlab-Multi-Cube-Seg-Yam``) that uses it (:issue:`862`).
  Contribution by @pthangeda.

Changed
^^^^^^^

- Renamed the ``list_envs`` console script to ``list-envs`` for consistency
  with the other hyphenated entry points (``viz-nan``, ``export-scene``).
  Invoke via ``uv run list-envs``.
- ``ActuatorCfg.armature`` and ``ActuatorCfg.frictionloss`` now default to
  ``None`` instead of ``0.0``. ``None`` preserves the value defined in the
  XML. Previously, builtin actuators would silently overwrite XML joint and
  tendon properties with zero when these fields were not explicitly set.
  To restore the old behavior, pass ``armature=0.0`` or ``frictionloss=0.0``
  explicitly.
- Actuator delay is now configured inline on any ``ActuatorCfg`` subclass
  (e.g. ``BuiltinPositionActuatorCfg(..., delay_min_lag=2, delay_max_lag=5)``)
  instead of wrapping with ``DelayedActuatorCfg``. ``DelayedActuator``,
  ``DelayedActuatorCfg``, and ``DelayedBuiltinActuatorGroup`` are removed.
- Removed ``delay_target`` from ``ActuatorCfg``. Delay now always applies to
  the actuator's ``command_field`` automatically. Multi-target delay
  (``delay_target=("position", "velocity")``) is no longer supported.
- ``XmlPositionActuatorCfg``, ``XmlVelocityActuatorCfg``, ``XmlMotorActuatorCfg``,
  and ``XmlMuscleActuatorCfg`` are replaced by a single ``XmlActuatorCfg`` that auto
  detects the actuator type from XML. Pass ``command_field=...`` to override detection.
- Replaced the viser viewer internals with the ``mjviser`` package. Scene
  creation, mesh conversion, and overlay rendering (contacts, forces,
  inertia, tendons, joints, frames) are now provided by mjviser. The viewer
  exposes a new Visualization tab for overlay controls and a Groups tab for
  geom/site visibility. Debug visualization and warp tensor conversion remain
  in mjlab's ``MjlabViserScene`` subclass (:issue:`839`).
- In curriculum terrain mode, each terrain type now gets exactly one column
  (``num_cols`` is set to ``len(sub_terrains)``). The ``proportion`` field
  now controls robot spawning distribution across columns rather than column
  count. Random mode is unchanged (:issue:`811`).
- ``BoxSteppingStonesTerrainCfg`` stone size now decreases with difficulty,
  interpolating from the large end of ``stone_size_range`` at difficulty 0
  to the small end at difficulty 1 (:issue:`785`).
- Removed deprecated ``TerrainImporter`` and ``TerrainImporterCfg`` aliases.
  Use ``TerrainEntity`` and ``TerrainEntityCfg`` instead (:issue:`667`).
- ``Entity.clear_state()`` is deprecated. Use ``Entity.reset()`` instead.
  ``clear_state`` only zeroed actuator targets without resetting actuator
  internal state (e.g. delay buffers), which could cause stale commands
  after teleporting the robot to a new pose.
- Removed ``EntityData.generalized_force``. The property was bugged (indexed
  free joint DOFs instead of articulated DOFs) and the name was ambiguous.
  Use ``qfrc_actuator`` or ``qfrc_external`` instead (:issue:`776`).
- ``get_wandb_checkpoint_path`` now filters checkpoints server-side via the
  ``pattern`` parameter, avoiding unnecessary pagination and tolerance to
  corrupted metadata (:issue:`898`).

Fixed
^^^^^

- ``train`` and ``play`` now print a top-level usage message when invoked
  with ``-h`` / ``--help`` and no task argument, pointing users at
  ``list-envs`` and ``<TASK> --help`` (:issue:`905`).
- Fixed ghost geom filtering in the Viser viewer. Ghost geoms were selected
  by collision flags, so collision-disabled robot geoms appeared as ghosts.
  The viewer now uses visual alpha to determine which geoms to render.
- Scene now warns when an attached entity or terrain spec has non-default
  ``<option>`` fields (e.g. ``<flag contact="disable"/>``), which are
  silently dropped by ``MjSpec.attach()``. Use ``MujocoCfg`` to set
  simulation options instead (:issue:`885`).
- Fixed ``SceneEntityCfg`` names and IDs ordering mismatch when
  ``preserve_order=False`` (:issue:`876`). Contribution by @jsw7460.
- Fixed ONNX export path resolution in the velocity, manipulation, and
  tracking runners when a parent directory name contains the word
  ``"model"`` (:issue:`867`). Contribution by @gokulp01.
- ``export-scene`` now writes only referenced assets and places them
  correctly under the output directory. Previously, asset keys containing
  path traversal could write files outside the output directory, and all
  spec assets were included regardless of whether the scene XML referenced
  them (:issue:`858`).
- ``electrical_power_cost`` now uses ``qfrc_actuator`` (joint space) instead
  of ``actuator_force`` (actuation space) for mechanical power computation.
  Previously the reward was incorrect for actuators with gear ratios other
  than 1 (:issue:`776`).
- ``create_velocity_actuator`` no longer sets ``ctrllimited=True`` with
  ``inheritrange=1.0``. This caused a ``ValueError`` for continuous joints
  (e.g. wheels) that have no position range defined (:issue:`787`).
- ``write_root_com_velocity_to_sim`` no longer fails with tensor ``env_ids``
  on floating base entities (:issue:`793`).
- Joint limits for unlimited joints are now set to [-inf, inf] instead of
  [0, 0]. Previously the zero range caused incorrect clamping for entities
  with unlimited hinge or slide joints.
- Contact force visualization now copies ``ctrl`` into the CPU ``MjData``
  before calling ``mj_forward``. Actuators that compute torques in Python
  (``DcMotorActuator``, ``IdealPdActuator``) previously showed incorrect
  contact forces because the viewer ran with ``ctrl=0``
  (:issue:`786`).
- ``BoxSteppingStonesTerrainCfg`` no longer creates a large gap around the
  platform. Stones are now only skipped when their center falls inside the
  platform; edges that extend under the platform are allowed since the
  platform covers them (:issue:`785`).
- ``dr.pseudo_inertia`` no longer loads cuSOLVER, eliminating ~4 GB of
  persistent GPU memory overhead. Cholesky and eigendecomposition are now
  computed analytically for the small matrices involved (4x4 and 3x3)
  (:issue:`753`).
- Set terrain geom mass to zero so that the static terrain body does not
  inflate ``stat.meanmass``, which made force arrow visualization invisible
  on rough terrain (:issue:`734`, :issue:`537`).
- Native viewer now syncs ``qpos0`` when domain randomized, fixing incorrect
  body positions after ``dr.joint_default_pos`` randomization
  (:issue:`760`).
- ``command_manager.compute()`` is now called during ``reset()`` so that
  derived command state (e.g. relative body positions in tracking
  environments) is populated before the first observation is returned
  (:issue:`761`).
- ``RayCastSensor`` with ``ray_alignment="yaw"`` or ``"world"`` now correctly
  aligns the frame offset when attached to a site or geom with a local offset
  from its parent body. Previously only ray directions and pattern offsets were
  aligned, causing the frame position to swing with body pitch/roll
  (:issue:`775`).

Version 1.2.0 (March 6, 2026)
-----------------------------

.. admonition:: Breaking API changes
   :class: attention

   - ``randomize_field`` no longer exists. Replace calls with typed functions
     from the new ``dr`` module (e.g. ``dr.geom_friction``, ``dr.body_mass``).
   - ``EventTermCfg`` no longer accepts ``domain_randomization``. The
     ``@requires_model_fields`` decorator on each ``dr`` function takes care
     of field expansion automatically.
   - ``Scene.to_zip()`` is deprecated. Use ``Scene.write(path, zip=True)``.
   - ``RslRlModelCfg`` no longer accepts ``stochastic``, ``init_noise_std``,
     or ``noise_std_type``. Use ``distribution_cfg`` instead
     (e.g. ``{"class_name": "GaussianDistribution", "init_std": 1.0,
     "std_type": "scalar"}``). Existing checkpoints are automatically
     migrated on load.

Added
^^^^^

- Added ``"step"`` event mode that fires every environment step.
- Added ``apply_body_impulse`` event for applying transient external wrenches
  to bodies with configurable duration and optional application point offset.
- ONNX auto-export and metadata attachment for manipulation tasks (lift cube)
  on every checkpoint save, matching the velocity and tracking task behavior.
- Multi-frame ``RayCastSensor``: pass a tuple of ``ObjRef`` to ``frame`` for
  per-site raycasting with independent body exclusion. New properties:
  ``num_frames``, ``num_rays_per_frame``. New ``RayCastData`` fields:
  ``frame_pos_w`` and ``frame_quat_w``.
- ``RingPatternCfg`` ray pattern for concentric ring sampling around each
  frame.
- ``TerrainHeightSensor``, a ``RayCastSensor`` subclass that computes
  per-frame vertical clearance above terrain (``sensor.data.heights``).
  Velocity task configs now use it for ``feet_clearance``,
  ``feet_swing_height``, and ``foot_height``, replacing the previous
  world-Z proxy that was incorrect on rough terrain.
- Cloud training support via `SkyPilot <https://skypilot.readthedocs.io/>`_
  and Lambda Cloud, with documentation covering setup, monitoring, and
  cost management.
- W&B hyperparameter sweep scripts that distribute one agent per GPU
  across a multi-GPU instance.
- Contributing guide with documentation for shared Claude Code commands
  (``/update-mjwarp``, ``/commit-push-pr``).
- Added optional ``ViewerConfig.fovy`` and apply it in native viewer camera
  setup when provided.
- Native viewer now tracks the first non-fixed body by default (matching
  the Viser viewer behavior introduced in
  ``716aaaa58ad7bfaf34d2f771549d461204d1b4ba``).
- New ``dr`` module (``mjlab.envs.mdp.dr``) replacing ``randomize_field``
  with typed per-field domain randomization functions. Each function
  automatically recomputes derived fields via ``set_const``. Highlights:

  - Camera and light randomization: ``dr.cam_fovy``, ``dr.cam_pos``,
    ``dr.cam_quat``, ``dr.cam_intrinsic``, ``dr.light_pos``,
    ``dr.light_dir``. Camera and light names are now supported in
    ``SceneEntityCfg`` (``camera_names`` / ``light_names``).
  - ``dr.pseudo_inertia`` for physics-consistent randomization of
    ``body_mass``, ``body_ipos``, ``body_inertia``, and ``body_iquat``
    via the pseudo-inertia matrix parameterization (Rucker & Wensing
    2022). Replaces the removed ``dr.body_inertia`` /
    ``dr.body_iquat``.
  - ``dr.geom_size`` with automatic recomputation of ``geom_rbound``
    and ``geom_aabb`` for broadphase consistency.
  - ``dr.tendon_armature`` and ``dr.tendon_frictionloss``.
  - ``dr.body_quat``, ``dr.geom_quat``, and ``dr.site_quat`` with RPY
    perturbation composed onto the default quaternion.
  - Extensible ``Operation`` and ``Distribution`` types. Users can define
    custom operations and distributions as class instances and pass them
    anywhere a string is accepted. Built-in instances (``dr.abs``,
    ``dr.scale``, ``dr.add``, ``dr.uniform``, ``dr.log_uniform``,
    ``dr.gaussian``) are exported from the ``dr`` module.
  - ``dr.mat_rgba`` for per-world material color randomization. Tints
    the texture color, useful for randomizing appearance of textured
    surfaces. Material names are now supported in ``SceneEntityCfg``
    (``material_names``).
  - Fixed ``dr.effort_limits`` drifting on repeated randomization.
  - Fixed ``dr.body_com_offset`` not triggering ``set_const``.

- ``export-scene`` CLI script to export any task scene or asset_zoo entity
  (``g1``, ``go1``, ``yam``) to a directory or zip archive for inspection
  and debugging.

- ``yam_lift_cube_vision_env_cfg`` now randomizes cube color (``dr.geom_rgba``)
  on every reset when ``cam_type="rgb"``.

- The native viewer now reflects per-world DR changes to visual model fields
  on each reset. Geom appearance, body and site poses, camera parameters,
  and light positions are all synced from the GPU model before rendering.
  Inertia boxes (press ``I``) and camera frustums (press ``Q``) update
  correctly when the corresponding fields are randomized. See
  :doc:`randomization` for viewer-specific caveats.

- ``MaterialCfg.geom_names_expr`` for assigning materials to geoms by
  name pattern during ``edit_spec``.

- ``TerrainEntityCfg`` now exposes ``textures``, ``materials``, and
  ``lights`` as configurable fields (previously hardcoded). Set
  ``textures=()``, ``materials=()`` to use flat ``dr.geom_rgba``
  instead of the default checker texture.

- ``DebugVisualizer`` now supports ellipsoid visualization via
  ``add_ellipsoid``.

- Interactive velocity joystick sliders in the Viser viewer. Enable the
  joystick under Commands/Twist to override velocity commands with manual
  sliders for ``lin_vel_x``, ``lin_vel_y``, and ``ang_vel_z``
  (`#666 <https://github.com/mujocolab/mjlab/issues/666>`_).
- Per-term debug visualization toggles in the Viser viewer. Individual
  command term visualizers (e.g. velocity arrows) can now be toggled
  independently under Scene/Debug Viz.
- Viewer single-step mode: press RIGHT arrow (native) or click "Step"
  (Viser) to advance exactly one physics step while paused.
- Viewer error recovery: exceptions during stepping now pause the viewer
  and log the traceback instead of crashing the process.
- Native viewer runs forward kinematics while paused, keeping
  perturbation visuals accurate.
- Viewer speed multipliers use clean power-of-2 fractions (1/32x to 1x).

- Visualizers display the realtime factor alongside FPS.

- ``joint_torques_l2`` now respects ``SceneEntityCfg.actuator_ids``,
  allowing penalization of a subset of actuators instead of all of them
  (`#703 <https://github.com/mujocolab/mjlab/pull/703>`_). Contribution by
  `@saikishor <https://github.com/saikishor>`_.

- Terrain is now a proper ``Entity`` subclass (``TerrainEntity``). This
  allows domain randomization functions to target terrain parameters
  (friction, cameras, lights) via ``SceneEntityCfg("terrain", ...)``.
  ``TerrainImporter`` / ``TerrainImporterCfg`` remain as aliases but will be
  deprecated in a future version.
- Added ``upload_model`` option to ``RslRlBaseRunnerCfg`` to control W&B model
  file uploads (``.pt`` and ``.onnx``) while keeping metric logging enabled
  (`#654 <https://github.com/mujocolab/mjlab/pull/654>`_).
- ``Scene.write(output_dir, zip=False)`` exports the scene XML and mesh
  assets to a directory (or zip archive). Replaces ``Scene.to_zip()``.
- ``Entity.write_xml()`` and ``Scene.write()`` now apply XML fixups
  (empty defaults, duplicate nested defaults) and strip buffer textures
  that ``MjSpec.to_xml()`` cannot serialize.
- ``fix_spec_xml`` and ``strip_buffer_textures`` utilities in
  ``mjlab.utils.xml``.

Changed
^^^^^^^

- Native viewer now syncs ``xfrc_applied`` to the render buffer and draws
  arrows for any nonzero applied forces. Mouse perturbation forces are
  converted to ``qfrc_applied`` (generalized joint space) so they coexist
  with programmatic forces on ``xfrc_applied`` without conflict.
- ``ViewerConfig.OriginType.WORLD`` now configures a free camera at the
  specified lookat point instead of auto tracking a body. A new ``AUTO``
  origin type (now the default) preserves the previous auto tracking
  behavior.
- Upgraded ``rsl-rl-lib`` from 4.0.1 to 5.0.1. ``RslRlModelCfg`` now
  uses ``distribution_cfg`` dict instead of ``stochastic`` /
  ``init_noise_std`` / ``noise_std_type``. Existing checkpoints are
  automatically migrated on load.
- Reorganized the Viser Controls tab into a cleaner folder hierarchy:
  Info, Simulation, Commands, Scene (with Environment, Camera, Debug Viz,
  Contacts sub-folders), and Camera Feeds. The Environment folder is
  hidden for single-env tasks and the Commands folder is hidden when no
  command terms are active.
- Viser camera tracking is now enabled by default so the agent stays in
  frame on launch.
- Self collision and illegal contact sensors now use ``history_length`` to
  catch contacts across decimation substeps. Reward and termination functions
  read ``force_history`` with a configurable ``force_threshold``.
- Replaced the single ``scale`` parameter in ``DifferentialIKActionCfg`` with
  separate ``delta_pos_scale`` and ``delta_ori_scale`` for independent scaling
  of position and orientation components.
- Improved offscreen multi environment framing by selecting neighboring
  environments around the focused env instead of first N envs.
- Tuned tracking task viewer defaults for tighter camera framing.
- Disabled shadow casting on the G1 tracking light to avoid duplicate
  stacked shadows when robots are close.

Fixed
^^^^^

- Fixed actuator target resolution for entities whose ``spec_fn`` uses
  internal ``MjSpec.attach(prefix=...)``
  (`#709 <https://github.com/mujocolab/mjlab/issues/709>`_).
- Fixed viewer physics loop starving the renderer by replacing the single
  sim-time budget with a two-clock design (tracked vs actual sim time).
  Physics now self-corrects after overshooting, keeping FPS smooth at all
  speed multipliers.
- Bundled ``ffmpeg`` for ``mediapy`` via ``imageio-ffmpeg``, removing the
  requirement for a system ``ffmpeg`` install. Thanks to
  `@rdeits-bd <https://github.com/rdeits-bd>`_ for the suggestion.
- Fixed ``height_scan`` returning ~0 for missed rays; now defaults to
  ``max_distance``. Replaced ``clip=(-1, 1)`` with ``scale`` normalization
  in the velocity task config. Thanks to `@eufrizz <https://github.com/eufrizz>`_
  for reporting and the initial fix (`#642 <https://github.com/mujocolab/mjlab/pull/642>`_).
- Fixed ghost mesh visualization for fixed-base entities by extending
  ``DebugVisualizer.add_ghost_mesh`` to optionally accept ``mocap_pos`` and
  ``mocap_quat`` (`#645 <https://github.com/mujocolab/mjlab/pull/645>`_).
- Fixed viser viewer crashing on scenes with no mocap bodies by adding
  an ``nmocap`` guard, matching the native viewer behavior.
- Fixed offscreen rendering artifacts in large vectorized scenes by applying
  a render local extent override in ``OffscreenRenderer`` and restoring the
  original extent on close.
- Fixed ``RslRlVecEnvWrapper.unwrapped`` to return the base environment,
  ensuring checkpoint state restore and logging work correctly when wrappers
  such as ``VideoRecorder`` are enabled.

Version 1.1.1 (February 14, 2026)
---------------------------------

Added
^^^^^

- Added reward term visualization to the native viewer (toggle with ``P``) (`#629 <https://github.com/mujocolab/mjlab/pull/629>`_).
- Added ``DifferentialIKAction`` for task-space control via damped
  least-squares IK. Supports weighted position/orientation tracking,
  soft joint-limit avoidance, and null-space posture regularization.
  Includes an interactive viser demo (``scripts/demos/differential_ik.py``) (`#632 <https://github.com/mujocolab/mjlab/pull/632>`_).

Fixed
^^^^^

- Fixed ``play.py`` defaulting to the base rsl-rl ``OnPolicyRunner`` instead
  of ``MjlabOnPolicyRunner``, which caused a ``TypeError`` from an unexpected
  ``cnn_cfg`` keyword argument (`#626 <https://github.com/mujocolab/mjlab/pull/626>`_). Contribution by
  `@griffinaddison <https://github.com/griffinaddison>`_.

Changed
^^^^^^^

- Removed ``body_mass``, ``body_inertia``, ``body_pos``, and ``body_quat``
  from ``FIELD_SPECS`` in domain randomization. These fields have derived
  quantities that require ``set_const`` to recompute; without that call,
  randomizing them silently breaks physics (`#631 <https://github.com/mujocolab/mjlab/pull/631>`_).
- Replaced ``moviepy`` with ``mediapy`` for video recording. ``mediapy``
  handles cloud storage paths (GCS, S3) natively (`#637 <https://github.com/mujocolab/mjlab/pull/637>`_).

.. figure:: _static/changelog/native_reward.png
   :width: 80%

Version 1.1.0 (February 12, 2026)
---------------------------------

Added
^^^^^

- Added RGB and depth camera sensors and BVH-accelerated raycasting (`#597 <https://github.com/mujocolab/mjlab/pull/597>`_).
- Added ``MetricsManager`` for logging custom metrics during training (`#596 <https://github.com/mujocolab/mjlab/pull/596>`_).
- Added terrain visualizer (`#609 <https://github.com/mujocolab/mjlab/pull/609>`_). Contribution by
  `@mktk1117 <https://github.com/mktk1117>`_.

.. figure:: _static/changelog/terrain_visualizer.jpg
   :width: 80%

- Added many new terrains including ``HfDiscreteObstaclesTerrainCfg``,
  ``HfPerlinNoiseTerrainCfg``, ``BoxSteppingStonesTerrainCfg``,
  ``BoxNarrowBeamsTerrainCfg``, ``BoxRandomStairsTerrainCfg``, and
  more. Added flat patch sampling for heightfield terrains (`#542 <https://github.com/mujocolab/mjlab/pull/542>`_, `#581 <https://github.com/mujocolab/mjlab/pull/581>`_).
- Added site group visualization to the Viser viewer (Geoms and Sites
  tabs unified into a single Groups tab) (`#551 <https://github.com/mujocolab/mjlab/pull/551>`_).
- Added ``env_ids`` parameter to ``Entity.write_ctrl_to_sim`` (`#567 <https://github.com/mujocolab/mjlab/pull/567>`_).

Changed
^^^^^^^

- Upgraded ``rsl-rl-lib`` to 4.0.0 and replaced the custom ONNX
  exporter with rsl-rl's built-in ``as_onnx()`` (`#589 <https://github.com/mujocolab/mjlab/pull/589>`_, `#595 <https://github.com/mujocolab/mjlab/pull/595>`_).
- ``sim.forward()`` is now called unconditionally after the decimation
  loop. See :ref:`faq-sim-forward` for details (`#591 <https://github.com/mujocolab/mjlab/pull/591>`_).
- Unnamed freejoints are now automatically named to prevent
  ``KeyError`` during entity init (`#545 <https://github.com/mujocolab/mjlab/pull/545>`_).

Fixed
^^^^^

- Fixed ``randomize_pd_gains`` crash with ``num_envs > 1`` (`#564 <https://github.com/mujocolab/mjlab/pull/564>`_).
- Fixed ``ctrl_ids`` index error with multiple actuated entities (`#573 <https://github.com/mujocolab/mjlab/pull/573>`_).
  Reported by `@bwrooney82 <https://github.com/bwrooney82>`_.
- Fixed Viser viewer rendering textured robots as gray (`#544 <https://github.com/mujocolab/mjlab/pull/544>`_).
- Fixed Viser plane rendering ignoring MuJoCo size parameter (`#540 <https://github.com/mujocolab/mjlab/pull/540>`_).
- Fixed ``HfDiscreteObstaclesTerrainCfg`` spawn height (`#552 <https://github.com/mujocolab/mjlab/pull/552>`_).
- Fixed ``RaycastSensor`` visualization ignoring the all-envs toggle (`#607 <https://github.com/mujocolab/mjlab/pull/607>`_).
  Contribution by `@oxkitsune <https://github.com/oxkitsune>`_.

Version 1.0.0 (January 28, 2026)
--------------------------------

Initial release of mjlab.
