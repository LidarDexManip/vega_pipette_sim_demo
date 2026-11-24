# Source this file (bash). It sets the VEGA_* variables from config/vega.env plus defaults relative to the repository,
# and defines the helpers the run scripts share. Values already in the environment win over config/vega.env.
_vega_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VEGA_HOME="${VEGA_HOME:-$(cd "$_vega_here/.." && pwd)}"
if [ -f "$VEGA_HOME/config/vega.env" ]; then
  while IFS='=' read -r _k _v; do
    case "$_k" in VEGA_*|OMNI_KIT_ACCEPT_EULA) [ -z "${!_k:-}" ] && [ -n "$_v" ] && export "$_k=$_v" ;; esac
  done < <(grep -E '^[[:space:]]*(VEGA_[A-Z0-9_]+|OMNI_KIT_ACCEPT_EULA)=' "$VEGA_HOME/config/vega.env" | sed 's/^[[:space:]]*//')
fi
_vega_abs() { case "$1" in /*) printf '%s' "$1" ;; "~/"*) printf '%s' "$HOME/${1#"~/"}" ;; *) printf '%s' "$VEGA_HOME/$1" ;; esac; }
export VEGA_RUNS="$(_vega_abs "${VEGA_RUNS:-runs}")"
export VEGA_TMP="$(_vega_abs "${VEGA_TMP:-$VEGA_RUNS/tmp}")"
export VEGA_GPU="${VEGA_GPU:-0}"
export VEGA_POLICY_PORT="${VEGA_POLICY_PORT:-5555}"
export VEGA_DASHBOARD_PORT="${VEGA_DASHBOARD_PORT:-8765}"
export VEGA_DEVICE="${VEGA_DEVICE:-}"
export VEGA_CHECKPOINT="$(_vega_abs "${VEGA_CHECKPOINT:-checkpoints/vega-wuji-pipette-pi05}")"
export VEGA_PICK_PLACE_CHECKPOINT="$(_vega_abs "${VEGA_PICK_PLACE_CHECKPOINT:-checkpoints/vega-tube-pnp}")"
[ -n "${VEGA_ISAAC_PY:-}" ] && export VEGA_ISAAC_PY="$(_vega_abs "$VEGA_ISAAC_PY")"
[ -n "${VEGA_POLICY_PY:-}" ] && export VEGA_POLICY_PY="$(_vega_abs "$VEGA_POLICY_PY")"
unset _vega_here

vega_task() {  # vega_task pipette|pick-place: the task's runner module (VEGA_RUNNER) and checkpoint (VEGA_TASK_CHECKPOINT)
  case "$1" in
    pipette) VEGA_RUNNER=vega_pipette.live; VEGA_TASK_CHECKPOINT=$VEGA_CHECKPOINT ;;
    pick-place) VEGA_RUNNER=vega_pipette.pick_place; VEGA_TASK_CHECKPOINT=$VEGA_PICK_PLACE_CHECKPOINT ;;
    *) echo "error: unknown task $1 (pipette or pick-place)" >&2; return 1 ;;
  esac
  VEGA_TASK=$1
}

vega_check() {  # the settings every run needs: both Pythons, the accepted EULA, the task's checkpoint, the packing table
  local v
  for v in VEGA_ISAAC_PY VEGA_POLICY_PY; do
    [ -n "${!v:-}" ] || { echo "error: $v is not set (edit config/vega.env, see config/vega.env.example)" >&2; return 1; }
  done
  [ "${OMNI_KIT_ACCEPT_EULA:-}" = YES ] || { echo "error: Isaac Sim requires accepting the NVIDIA Omniverse License Agreement. Read it, then set OMNI_KIT_ACCEPT_EULA=YES in config/vega.env" >&2; return 1; }
  [ -d "$VEGA_TASK_CHECKPOINT/params" ] || { echo "error: no checkpoint at $VEGA_TASK_CHECKPOINT (see README.md: download the checkpoint)" >&2; return 1; }
  [ -f "$VEGA_HOME/assets/props/PackingTable/PackingTable.usd" ] || { echo "error: the packing table is missing (run: python3 tools/fetch_packing_table.py)" >&2; return 1; }
}

vega_device() {  # vega_device WORD DEFAULT: "cpu" or "gpu" -> the physics device; anything else -> VEGA_DEVICE, else DEFAULT
  case "$1" in cpu) echo cpu ;; gpu) echo cuda:0 ;; *) echo "${VEGA_DEVICE:-$2}" ;; esac
}

vega_cpu_only() {  # vega_cpu_only WORD: the tube task runs with CPU physics only; refuse "gpu" for it
  if [ "$VEGA_TASK" = pick-place ] && [ "$1" = gpu ]; then
    echo "error: the tube task runs with CPU physics, the physics it was trained and evaluated with" >&2
    return 1
  fi
}

vega_isaac() {  # vega_isaac [--limit SECONDS] ARGS...: the Isaac Python, run from the repository root in a clean
                # environment; with --limit it is stopped (SIGTERM, then SIGKILL 30 s later) after SECONDS
  local envv=() k pre=()
  if [ "${1:-}" = --limit ]; then pre=(timeout --kill-after=30 "$2"); shift 2; fi
  for k in HOME LANG LOGNAME SHELL USER XDG_RUNTIME_DIR DISPLAY XAUTHORITY WAYLAND_DISPLAY; do
    [ -n "${!k+x}" ] && envv+=("$k=${!k}")
  done
  for k in $(compgen -e | grep '^VEGA_'); do envv+=("$k=${!k}"); done
  envv+=("PATH=$PATH" "CUDA_VISIBLE_DEVICES=$VEGA_GPU" "TMPDIR=$VEGA_TMP" "OMNI_KIT_ACCEPT_EULA=$OMNI_KIT_ACCEPT_EULA"
         "PYTHONUNBUFFERED=1" "PYTHONNOUSERSITE=1" "PYTHONDONTWRITEBYTECODE=1")
  mkdir -p "$VEGA_TMP"
  (cd "$VEGA_HOME" && "${pre[@]}" env -i "${envv[@]}" "$VEGA_ISAAC_PY" "$@")
}

vega_start_server() {  # vega_start_server PORT LOG: start the task's policy server and wait until it answers (sets VEGA_SERVER_PID)
  local port=$1 log=$2 ready t0
  ready="$log.ready"
  rm -f "$ready"
  CUDA_VISIBLE_DEVICES=$VEGA_GPU "$VEGA_POLICY_PY" "$VEGA_HOME/policy_server/serve.py" --task "$VEGA_TASK" \
    --checkpoint "$VEGA_TASK_CHECKPOINT" --port "$port" --ready-file "$ready" > "$log" 2>&1 &
  VEGA_SERVER_PID=$!
  t0=$(date +%s)
  until [ -e "$ready" ]; do
    kill -0 "$VEGA_SERVER_PID" 2>/dev/null || { echo "error: the policy server exited; last lines of $log:" >&2; tail -20 "$log" >&2; return 1; }
    [ $(( $(date +%s) - t0 )) -lt "${SERVER_TIMEOUT_S:-900}" ] || { echo "error: the policy server was not ready after ${SERVER_TIMEOUT_S:-900} s" >&2; return 1; }
    sleep 1
  done
  echo "== policy server ready after $(( $(date +%s) - t0 )) s"
}

vega_stop_server() {
  if [ -n "${VEGA_SERVER_PID:-}" ] && kill -0 "$VEGA_SERVER_PID" 2>/dev/null; then
    kill "$VEGA_SERVER_PID" 2>/dev/null
    for _ in $(seq 20); do kill -0 "$VEGA_SERVER_PID" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$VEGA_SERVER_PID" 2>/dev/null
  fi
  VEGA_SERVER_PID=
}
