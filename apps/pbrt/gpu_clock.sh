# The GPU clock and the timed runs, for gpu_compare.sh and heatmap.sh to
# source. A GeForce boosts by power and temperature -- the RTX 5090 here
# runs a light scene near 3.0 GHz and a dense one lower -- so a speedup
# measured at an unlocked clock moves with the load alone (REPORT.md, "Why
# the speedup moves with the sample count"), and the sound way to time is
# at a locked clock. Locking takes root (`nvidia-smi -lgc`), which these
# scripts do not have.
#
# Whether the lock is required is BONSAI_GPU_CLOCK_REQUIRED: 1 refuses a
# run whose idle GPU does not read the lock, printing the commands; 0, the
# default since 2026-10-10 (the user's call: another project's GPU
# benchmarks share this card and must not run locked for now), only
# records what the clock reads, and every table's last line and
# heatmap.log say which it was. `--clock N` on either script asks for the
# lock at N and requires it. BONSAI_GPU_CLOCK is the lock in MHz; 2400,
# the 5090's rated boost clock, is one the card holds at full power on
# every scene (3090 is its maximum, which the power limit takes away on
# the dense scenes). The memory clock is locked at its maximum alongside.
#
# The reading is of an idle GPU: a locked clock reads the lock whatever
# runs, but an unlocked one boosts under load and passes through the
# lock's value on its way (it did, 2026-10-10 00:24, and a sweep started
# unlocked), while idle it sits at 180 MHz; so the GPU has to be free of
# compute processes and settled for two seconds at each reading, waited
# for up to a minute.
GPU_CLOCK="${BONSAI_GPU_CLOCK:-2400}"
GPU_CLOCK_REQUIRED="${BONSAI_GPU_CLOCK_REQUIRED:-0}"
GPU_CLOCK_NOTE=""

gpu_clock_lock_commands() {
  local maxmem
  maxmem=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "    sudo nvidia-smi -lgc $GPU_CLOCK,$GPU_CLOCK && sudo nvidia-smi -lmc $maxmem,$maxmem"
  echo "    (sudo nvidia-smi -rgc && sudo nvidia-smi -rmc release them; BONSAI_GPU_CLOCK=N or --clock N names another lock)"
}

# Reads the idle GPU's clocks three times and sets GPU_CLOCK_NOTE to what
# they are: locked at the lock, or unlocked. Fails only when the lock is
# required and absent, saying how to lock.
require_locked_gpu_clock() {
  local sm mem maxmem i waited locked=1
  maxmem=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1 | tr -d ' ')
  for i in 1 2 3; do
    waited=0
    while (( $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l) > 0 )); do
      if (( waited >= 60 )); then
        echo "the GPU has been busy for a minute; its clock can be read only idle" >&2
        return 1
      fi
      sleep 1
      waited=$((waited + 1))
    done
    sleep 2
    sm=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')
    mem=$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits | head -1 | tr -d ' ')
    # Within a percent: the lock snaps to the nearest clock the card
    # supports (2400 asked for may read 2407), and the memory clock's top
    # state reads a step below its listed maximum (13801 against 14001).
    if (( sm * 100 < GPU_CLOCK * 99 || sm * 100 > GPU_CLOCK * 101 || mem * 100 < maxmem * 98 )); then
      locked=0
    fi
  done
  if (( locked )); then
    GPU_CLOCK_NOTE="GPU clocks locked: $sm MHz SM, $mem MHz memory"
    return 0
  fi
  GPU_CLOCK_NOTE="GPU clock unlocked (idle reading $sm MHz SM, $mem MHz memory; boosts by load)"
  if [[ "$GPU_CLOCK_REQUIRED" == 1 ]]; then
    echo "the GPU's clocks read $sm MHz SM and $mem MHz memory idle, not the lock of $GPU_CLOCK MHz; this run requires the lock:" >&2
    gpu_clock_lock_commands >&2
    return 1
  fi
  echo "$GPU_CLOCK_NOTE" >&2
  return 0
}
