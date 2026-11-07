# Instructions for coding agents

Read `AI_CONTEXT.md`, `CLAUDE.md`, and the files you will edit before acting.
This is the team's persistent DroneHacks companion project. Treat existing
mission code, notebooks and team instructions as team-owned work.

- Use ordinary ROS 2 / `px4_msgs`, camera calibration and TF interfaces.
- Keep simulator truth, private host paths, lab APIs and reset commands out of
  flight algorithms. A passing simulator test is not hardware qualification.
  Exception: custom simulated sensor **models** may use `dronehacks_sim` as
  documented in `SIM_SENSORS.md`. Keep the oracle producer separate from the
  portable navigation code and label assumed sensor errors and oracle use.
- Explain a flight plan and obtain the team's approval before arming. Inspect
  fresh telemetry; stop publishing if authority is lost or telemetry is stale.
- Preserve the sensor/estimator checks. Never invent a position fix, weaken a
  grader, or change PX4 parameters merely to make a failing mission pass.
- Run relevant project tests first; record the exact code, settings, outcome
  and evidence for moving tests. Preserve failures and explain corrections.
- Use the current editor's identity. Save and stop detached agents before
  handing editing control to another teammate. Never store credentials in Git.

The student's team owns mission approval. These participant instructions do
not override the simulator operator's separately granted SITL permissions.

After using keyboard RC, land and release RC, then use **Reset world** before
running an automated program, including a notebook or native terminal script.
The cloud follow-up run without Reset failed; the fresh-world native run passed.
The web Run button enforces this qualification boundary; terminal Python remains
ordinary ROS and does not enforce it for you. This is not a VIO fix.
