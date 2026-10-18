"""Humanoid lab manipulation in simulation (Vega-U + Wuji Hand pipette task): run a pi0.5 policy live in Isaac Sim,
with a live dashboard and evaluation-style videos.

  env/        the simulated task (Isaac Lab; import after AppLauncher)
  sim/        one running scene: build, settle, step, read cameras; layouts; the per-step log
  control/    the policy-server client and the action-chunk scheduler
  progress/   the milestone scorer and everything derived from a run's log
  video/      the evaluation video: frame layout, composition, high-resolution replay
  dashboard/  the live web dashboard
  live.py     the runner: python -m vega_pipette.live
"""
