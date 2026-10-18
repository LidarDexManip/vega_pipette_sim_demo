#!/usr/bin/env bash
# run_sim_headless.sh [standard|hires|both] [gpu|cpu] [--runs N] [--steps N]: headless runs of the policy and their
# evaluation videos, in the background. Returns at once and prints the log to follow (tail -f).
#   standard   (default) the evaluation frame composed from the run's own cameras: 1544 x 798
#   hires      the logged motion replayed with every camera at 3x resolution, then the frame: 3088 x 1596
#   both       both videos from the same run
#   gpu | cpu  the physics device (default: VEGA_DEVICE in config/vega.env, else gpu)
#   --runs N   N runs one after the other, each on its own random layout of the props (default 1)
#   --steps N  the most control steps per run (default 1200 = 40 s; a run stops 3 s after its outcome)
# Output: runs/videos/<time>_<standard|hires>.mp4, named after the run dir under runs/live/. Needs ffmpeg on
# PATH. Run one script at a time: the policy server port (5555) and the GPU are shared.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
if [ "${_VEGA_HEADLESS_WORKER:-}" != 1 ]; then          # detach; the work goes on in the background
  vega_check || exit 2
  command -v ffmpeg > /dev/null || { echo "error: ffmpeg is not on PATH" >&2; exit 2; }
  LOG=$VEGA_RUNS/headless_$(date +%Y%m%d-%H%M%S).log
  mkdir -p "$VEGA_RUNS"
  _VEGA_HEADLESS_WORKER=1 setsid nohup bash "$0" "$@" > "$LOG" 2>&1 < /dev/null &
  echo "running in the background; follow it with:  tail -f $LOG"
  echo "videos go to $VEGA_RUNS/videos/"
  exit 0
fi
NRUNS=1; STEPS=1200; Q=standard; WORD=
while [ $# -gt 0 ]; do
  case "$1" in
    standard|hires|both) Q=$1 ;;
    cpu|gpu) WORD=$1 ;;
    --runs) NRUNS=${2:?--runs needs a value}; shift ;;
    --steps) STEPS=${2:?--steps needs a value}; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
DEVICE=$(vega_device "$WORD" cuda:0)
OUT=$VEGA_RUNS/videos; mkdir -p "$OUT"
say() { echo "$(date '+%H:%M:%S') $*"; }
say "$NRUNS run(s); video $Q; physics on $DEVICE; up to $STEPS steps"
trap vega_stop_server EXIT
vega_start_server 5555 "$VEGA_RUNS/headless_server.log" || exit 1
REC=(); [ "$Q" != hires ] && REC=(--record-video)
RUNS=()
for I in $(seq "$NRUNS"); do         # 1. the policy runs (and their standard videos)
  R=$VEGA_RUNS/live/$(date +%Y%m%d-%H%M%S); L="run $I of $NRUNS"
  mkdir -p "$R"
  say "$L: policy run -> $R"
  vega_isaac -m vega_pipette.live "${REC[@]}" --device "$DEVICE" --policy zmq://127.0.0.1:5555 \
    --max-steps "$STEPS" --out-dir "$R" > "$R/live.log" 2>&1
  grep -hE '^\[live \] (random layout|done)' "$R/live.log" | sed "s/^/$(date '+%H:%M:%S') $L: /"
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
    if vega_isaac -m vega_pipette.video.render_hd --run "$R" --cam-scale 3 > "$R/render_hd.log" 2>&1; then
      vega_isaac -m vega_pipette.video.make_video --objlog "$R/objlog.npz" --cameras "$R/hd" --scale 2 \
        --out "$OUT/$(basename "$R")_hires.mp4" 2>&1 | sed "s/^/$(date '+%H:%M:%S') $L: /"
    else
      say "$L: the replay failed (see $R/render_hd.log)"
    fi
  done
fi
say "done"
