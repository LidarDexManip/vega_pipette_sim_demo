"""Humanoid lab manipulation in simulation (Vega-U + Wuji Hand: the pipette task and the tube pick-and-place task): run
a pi0.5 policy live in Isaac Sim, with a live dashboard and evaluation-style videos.

  env/          the simulated tasks (Isaac Lab; import after AppLauncher)
  sim/          one running scene: build, settle, step, read cameras; layouts; the per-step log
  control/      the policy-server client, the action-chunk scheduler, the tube task's place hint (kinematics, the
                tube-in-hand estimate)
  progress/     the milestone scorers and everything derived from a run's log
  video/        the evaluation video: frame layout, composition, high-resolution replay
  dashboard/    the live web dashboard
  live.py       the pipette runner: python -m vega_pipette.live
  pick_place.py the tube pick-and-place runner: python -m vega_pipette.pick_place
"""
