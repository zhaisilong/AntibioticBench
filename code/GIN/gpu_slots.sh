#!/usr/bin/env bash
# GPU slots for local launchers. Each slot owns one child PID and one GPU.

gpu_slots_init() {
  local gpu_list="$1" max_per_gpu="$2" run_count="$3" gpu i previous
  [[ "${max_per_gpu}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: MAX_JOBS_PER_GPU must be a positive integer" >&2; return 1; }
  [[ "${run_count}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: the run plan is empty or invalid" >&2; return 1; }
  [[ -n "${gpu_list}" && "${gpu_list}" != ,* && "${gpu_list}" != *, && "${gpu_list}" != *,,* ]] || { echo "ERROR: GPUS must contain a comma-separated list of unique GPUs" >&2; return 1; }
  IFS=',' read -r -a GPU_SLOT_DEVICES <<< "${gpu_list}"
  for ((i = 0; i < ${#GPU_SLOT_DEVICES[@]}; i++)); do
    gpu="${GPU_SLOT_DEVICES[i]}"
    gpu="${gpu#"${gpu%%[![:space:]]*}"}"
    gpu="${gpu%"${gpu##*[![:space:]]}"}"
    [[ -n "${gpu}" ]] || { echo "ERROR: GPUS contains an empty GPU identifier" >&2; return 1; }
    for previous in "${GPU_SLOT_DEVICES[@]:0:i}"; do
      [[ "${previous}" != "${gpu}" ]] || { echo "ERROR: GPUS contains a duplicate GPU identifier" >&2; return 1; }
    done
    GPU_SLOT_DEVICES[i]="${gpu}"
  done
  GPU_SLOT_MAX_PER_GPU="${max_per_gpu}"
  GPU_SLOT_COUNT=$((${#GPU_SLOT_DEVICES[@]} * max_per_gpu))
  GPU_SLOT_FAILED=0
  GPU_SLOT_PIDS=()
}

gpu_slots_acquire() {
  local slot pid
  while :; do
    # Reap all finished children before deciding whether to launch another run.
    for ((slot = 0; slot < GPU_SLOT_COUNT; slot++)); do
      pid="${GPU_SLOT_PIDS[slot]:-}"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        if ! wait "${pid}"; then GPU_SLOT_FAILED=1; fi
        unset 'GPU_SLOT_PIDS[slot]'
      fi
    done
    (( GPU_SLOT_FAILED == 0 )) || return 1
    for ((slot = 0; slot < GPU_SLOT_COUNT; slot++)); do
      if [[ -z "${GPU_SLOT_PIDS[slot]:-}" ]]; then
        GPU_SLOT_INDEX="${slot}"
        GPU_SLOT_DEVICE="${GPU_SLOT_DEVICES[slot / GPU_SLOT_MAX_PER_GPU]}"
        return 0
      fi
    done
    sleep "${GPU_SLOT_POLL_INTERVAL:-1}"
  done
}

gpu_slots_finish() {
  local slot pid
  for ((slot = 0; slot < GPU_SLOT_COUNT; slot++)); do
    pid="${GPU_SLOT_PIDS[slot]:-}"
    if [[ -n "${pid}" ]] && ! wait "${pid}"; then GPU_SLOT_FAILED=1; fi
  done
  if (( GPU_SLOT_FAILED )); then
    echo "ERROR: at least one run failed; postprocessing skipped" >&2
    return 1
  fi
}
