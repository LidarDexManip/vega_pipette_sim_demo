#!/usr/bin/env bash
# run_sim_headless.sh [pipette|pick-place] [standard|hires|both] [gpu|cpu] [--runs N] [--steps N]: headless runs of a
# policy and their evaluation videos, in the background. Returns at once and prints the log to follow (tail -f).
#   pipette | pick-place  the task (default pipette)
#   standard   (default) the evaluation frame composed from the run's own cameras: 1544 x 798
#   hires      the logged motion replayed with every camera at 3x resolution, then the frame: 3088 x 1596
#   both       both videos from the same run
#   gpu | cpu  the physics device of the pipette task (default: VEGA_DEVICE in config/vega.env, else gpu); the tube
#              pick-and-place task runs with CPU physics only
#   --runs N   N runs one after the other, each on its own random layout of the props (default 1)
#   --steps N  the most control steps per run (default 1200 = 40 s for pipette, 1900 = 63 s for pick-place; a run stops
#              at its outcome)
# The simulator has been seen to hang on rare occasions. A simulator that has not started its policy run (no layout
# line in live.log) within START_S is stopped and the run started again, once; a policy run or a replay still going
# after 600 s + 1 s per step (wall time) is stopped and reported, and the next run starts.
# Output: runs/videos/<time>_<standard|hires>.mp4, named after the run dir under runs/live/. Needs ffmpeg on
# PATH. Run one script at a time: the policy server port (5555, VEGA_POLICY_PORT) and the GPU are shared.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
ARGS=("$@"); NRUNS=1; STEPS=; Q=standard; WORD=; TASK=pipette
while [ $# -gt 0 ]; do
  case "$1" in
    pipette|pick-place) TASK=$1 ;;
    standard|hires|both) Q=$1 ;;
    cpu|gpu) WORD=$1 ;;
    --runs) NRUNS=${2:?--runs needs a value}; shift ;;
    --steps) STEPS=${2:?--steps needs a value}; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
vega_task "$TASK" && vega_cpu_only "$WORD" || exit 2
if [ "${_VEGA_HEADLESS_WORKER:-}" != 1 ]; then          # detach; the work goes on in the background
  vega_check || exit 2
  command -v ffmpeg > /dev/null || { echo "error: ffmpeg is not on PATH" >&2; exit 2; }
  LOG=$VEGA_RUNS/headless_$(date +%Y%m%d-%H%M%S).log
  mkdir -p "$VEGA_RUNS"
  _VEGA_HEADLESS_WORKER=1 setsid nohup bash "$0" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
  echo "running in the background; follow it with:  tail -f $LOG"
  echo "videos go to $VEGA_RUNS/videos/"
  exit 0
fi
if [ "$TASK" = pick-place ]; then DEVICE=cpu; DEV=(); STEPS=${STEPS:-1900}
else DEVICE=$(vega_device "$WORD" cuda:0); DEV=(--device "$DEVICE"); STEPS=${STEPS:-1200}; fi
OUT=$VEGA_RUNS/videos; mkdir -p "$OUT"
LIMIT_S=$(( 600 + STEPS )); START_S=600
say() { echo "$(date '+%H:%M:%S') $*"; }

policy_run() {  # policy_run: one policy run into $R; sets RC. A simulator still starting up after START_S is stopped
                # (its log kept as live.stuck.log) and the run started once more.
  local try pid t0 stuck
  for try in 1 2; do
    vega_isaac --limit "$LIMIT_S" -m "$VEGA_RUNNER" "${REC[@]}" "${DEV[@]}" \
      --policy "zmq://127.0.0.1:$VEGA_POLICY_PORT" --max-steps "$STEPS" --out-dir "$R" > "$R/live.log" 2>&1 &
    pid=$!; t0=$(date +%s); stuck=0
    while kill -0 "$pid" 2>/dev/null; do
      if [ $(( $(date +%s) - t0 )) -ge "$START_S" ] && ! grep -q '^\[live \] random layout' "$R/live.log"; then
        pkill -TERM -f -- "--out-dir $R"; sleep 30; pkill -KILL -f -- "--out-dir $R"; stuck=1
        break
      fi
      sleep 5
    done
    wait "$pid"; RC=$?
    [ "$stuck" = 1 ] || return
    mv "$R/live.log" "$R/live.stuck.log"
    say "$L: the simulator did not start within $START_S s (log: $R/live.stuck.log)$([ $try = 1 ] && echo '; starting the run again')"
  done
}
say "$TASK: $NRUNS run(s); video $Q; physics on $DEVICE; up to $STEPS steps"
trap vega_stop_server EXIT
vega_start_server "$VEGA_POLICY_PORT" "$VEGA_RUNS/headless_server.log" || exit 1
REC=(); [ "$Q" != hires ] && REC=(--record-video)
RUNS=()
for I in $(seq "$NRUNS"); do         # 1. the policy runs (and their standard videos)
  R=$VEGA_RUNS/live/$(date +%Y%m%d-%H%M%S); L="run $I of $NRUNS"
  mkdir -p "$R"
  say "$L: policy run -> $R"
  policy_run
  grep -hE '^\[live \] (random layout|done)' "$R/live.log" | sed "s/^/$(date '+%H:%M:%S') $L: /"
  [ "$RC" = 124 ] || [ "$RC" = 137 ] && say "$L: stopped at the time limit ($LIMIT_S s)"
  [ -f "$R/objlog.npz" ] || { say "$L: the policy run failed (see $R/live.log)"; continue; }
  RUNS+=("$R")
  if [ "$Q" != hires ]; then
    vega_isaac -m vega_pipette.video.make_video --objlog "$R/objlog.npz" --cameras "$R" --scale 1 \
      --out "$OUT/$(basename "$R")_standard.mp4" 2>&1 | sed "s/^/$(date '+%H:%M:%S') $L: /"
  fi
done
vega_stop_server                      # the replays need no policy, and the GPU memory goes to them
if [ "$Q" != standard ]; then         # 2. the high-resolution replays
  for R in "${RUNS[@]}"; do
    L=$(basename "$R")
    say "$L: high-resolution replay"
    if vega_isaac --limit "$LIMIT_S" -m vega_pipette.video.render_hd --run "$R" --cam-scale 3 \
        > "$R/render_hd.log" 2>&1; then
      vega_isaac -m vega_pipette.video.make_video --objlog "$R/objlog.npz" --cameras "$R/hd" --scale 2 \
        --out "$OUT/$(basename "$R")_hires.mp4" 2>&1 | sed "s/^/$(date '+%H:%M:%S') $L: /"
    else
      say "$L: the replay failed or reached the time limit (see $R/render_hd.log)"
    fi
  done
fi
say "done"
