#!/usr/bin/env bash
# run_sim_gui.sh [cpu|gpu] [--no-dashboard] [-- LIVE_OPTIONS]: run the policy live in the simulator window.
#   Starts the policy server and waits until it answers (the first start compiles the model, a few minutes), opens the
#   simulator window and the live dashboard (http://localhost:8765, opened in the browser), and runs the policy on
#   a random layout of the props, paced to the wall clock. The run stops 3 s after its outcome (success or a failure), when the window is
#   closed, on Ctrl+C, or after 40 s of task time; then the server stops.
#   cpu | gpu       the physics device (default: VEGA_DEVICE in config/vega.env, else cpu)
#   --no-dashboard  no dashboard (and no side camera)
#   LIVE_OPTIONS    passed to python -m vega_pipette.live, e.g. -- --inference-mode async (see its --help)
# Output: runs/live/<time>/ (run.json, objlog.npz, live.log, server.log). Run one script at a time: the
# policy server port (5555), the dashboard port (8765) and the GPU are shared.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
WORD=; DASH=(--dashboard); MORE=()
while [ $# -gt 0 ]; do
  case "$1" in
    cpu|gpu) WORD=$1 ;;
    --no-dashboard) DASH=(--no-dashboard) ;;
    --) shift; MORE=("$@"); break ;;
    -*) echo "unknown option $1 (usage: run_sim_gui.sh [cpu|gpu] [--no-dashboard] [-- LIVE_OPTIONS])" >&2; exit 2 ;;
    *) echo "unknown argument $1 (usage: run_sim_gui.sh [cpu|gpu] [--no-dashboard] [-- LIVE_OPTIONS])" >&2; exit 2 ;;
  esac
  shift
done
vega_check || exit 2
R=$VEGA_RUNS/live/$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
echo "== starting the policy server (log: $R/server.log)"
trap vega_stop_server EXIT
trap 'vega_stop_server; exit 130' INT TERM
vega_start_server 5555 "$R/server.log" || exit 1
if [ "${DASH[0]}" = --dashboard ] && [ -n "${DISPLAY:-}" ] && command -v xdg-open > /dev/null; then
  ( for _ in $(seq 120); do
      curl -fs -o /dev/null http://localhost:8765/ 2>/dev/null && { xdg-open http://localhost:8765/ > /dev/null 2>&1; break; }
      sleep 0.5
    done ) &
fi
echo "== opening the simulator window; close it or press Ctrl+C to stop"
vega_isaac -m vega_pipette.live --gui --realtime "${DASH[@]}" --device "$(vega_device "$WORD" cpu)" \
  --policy zmq://127.0.0.1:5555 --out-dir "$R" "${MORE[@]}" 2>&1 \
  | tee "$R/live.log" | grep --line-buffered -E '^\[(live|rt)'
RC=${PIPESTATUS[0]}
echo "== run dir: $R"
exit "$RC"
