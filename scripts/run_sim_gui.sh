#!/usr/bin/env bash
# run_sim_gui.sh [pipette|pick-place] [cpu|gpu] [--no-dashboard] [-- LIVE_OPTIONS]: run a policy live in the simulator
# window.
#   Starts the task's policy server and waits until it answers (the first start compiles the model, a few minutes), opens
#   the simulator window and the live dashboard (http://localhost:8765, opened in the browser), and runs the policy on
#   a random layout of the props, paced to the wall clock. The run stops at its outcome (pipette: 3 s after success or a
#   failure; tube pick-and-place: 1 s after the release), when the window is closed, on Ctrl+C, or at the time limit;
#   then the server stops.
#   pipette | pick-place  the task (default pipette)
#   cpu | gpu       the physics device of the pipette task (default: VEGA_DEVICE in config/vega.env, else cpu); the
#                   tube pick-and-place task runs with CPU physics only
#   --no-dashboard  no dashboard (and no side camera)
#   LIVE_OPTIONS    passed to the runner (python -m vega_pipette.live or vega_pipette.pick_place, see its --help), e.g.
#                   -- --inference-mode async
# Output: runs/live/<time>/ (run.json, objlog.npz, live.log, server.log). Run one script at a time: the policy server
# port (5555), the dashboard port (8765) and the GPU are shared (the ports: VEGA_POLICY_PORT, VEGA_DASHBOARD_PORT).
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
USAGE="usage: run_sim_gui.sh [pipette|pick-place] [cpu|gpu] [--no-dashboard] [-- LIVE_OPTIONS]"
TASK=pipette; WORD=; DASH=(--dashboard); MORE=()
while [ $# -gt 0 ]; do
  case "$1" in
    pipette|pick-place) TASK=$1 ;;
    cpu|gpu) WORD=$1 ;;
    --no-dashboard) DASH=(--no-dashboard) ;;
    --) shift; MORE=("$@"); break ;;
    -*) echo "unknown option $1 ($USAGE)" >&2; exit 2 ;;
    *) echo "unknown argument $1 ($USAGE)" >&2; exit 2 ;;
  esac
  shift
done
vega_task "$TASK" && vega_cpu_only "$WORD" && vega_check || exit 2
DEV=(); [ "$TASK" = pipette ] && DEV=(--device "$(vega_device "$WORD" cpu)")
R=$VEGA_RUNS/live/$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
echo "== starting the policy server (log: $R/server.log)"
trap vega_stop_server EXIT
trap 'vega_stop_server; exit 130' INT TERM
vega_start_server "$VEGA_POLICY_PORT" "$R/server.log" || exit 1
URL=http://localhost:$VEGA_DASHBOARD_PORT/
if [ "${DASH[0]}" = --dashboard ] && [ -n "${DISPLAY:-}" ] && command -v xdg-open > /dev/null; then
  ( for _ in $(seq 120); do
      curl -fs -o /dev/null "$URL" 2>/dev/null && { xdg-open "$URL" > /dev/null 2>&1; break; }
      sleep 0.5
    done ) &
fi
echo "== opening the simulator window; close it or press Ctrl+C to stop"
vega_isaac -m "$VEGA_RUNNER" --gui --realtime "${DASH[@]}" --dashboard-port "$VEGA_DASHBOARD_PORT" \
  "${DEV[@]}" --policy "zmq://127.0.0.1:$VEGA_POLICY_PORT" --out-dir "$R" "${MORE[@]}" 2>&1 \
  | tee "$R/live.log" | grep --line-buffered -E '^\[(live|rt)'
RC=${PIPESTATUS[0]}
echo "== run dir: $R"
exit "$RC"
