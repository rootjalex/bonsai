# The GPU clock lock that every timed run of this renderer runs under (the
# user, 2026-10-10: "all GPU benchmarks should do this from now on"), for
# gpu_compare.sh and heatmap.sh to source. A GeForce boosts by power and
# temperature -- the RTX 5090 here runs a light scene near 3.0 GHz and a
# dense one lower -- so an unlocked card's speedup over pbrt moves with the
# load alone (REPORT.md, "Why the speedup moves with the sample count").
# Locking takes root (`nvidia-smi -lgc`), which these scripts do not have,
# so require_locked_gpu_clock reads the SM clock three times, half a second
# apart, and refuses the run unless every reading is the lock: a locked
# clock does not move, an unlocked one idles at 180 MHz and boosts under
# load. BONSAI_GPU_CLOCK (or the scripts' --clock) is the lock in MHz; 2400,
# the 5090's rated boost clock, is one the card holds at full power on every
# scene (3090 is its maximum, which the power limit takes away on the dense
# scenes). The memory clock is locked at its maximum alongside -- GDDR7
# holds it under load anyway, and the lock keeps the idle state's clock out
# of a run's first launches.
GPU_CLOCK="${BONSAI_GPU_CLOCK:-2400}"
GPU_CLOCK_NOTE=""

gpu_clock_lock_commands() {
  local maxmem
  maxmem=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "    sudo nvidia-smi -lgc $GPU_CLOCK,$GPU_CLOCK && sudo nvidia-smi -lmc $maxmem,$maxmem"
  echo "    (sudo nvidia-smi -rgc && sudo nvidia-smi -rmc release them; BONSAI_GPU_CLOCK=N or --clock N names another lock)"
}

# Succeeds, and sets GPU_CLOCK_NOTE, when the GPU's clocks read the lock;
# otherwise says what they read and how to lock them, and fails. Every
# reading is of an idle GPU: a locked clock reads the lock whatever runs,
# but an unlocked one boosts under load and passes through the lock's value
# on its way (it did, 2026-10-10 00:24, and a sweep started unlocked),
# while idle it sits at 180 MHz; so the GPU has to be free of compute
# processes at each reading, waited for up to a minute.
require_locked_gpu_clock() {
  local sm mem maxmem i waited
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
    # An unlocked clock takes a second or two to fall back to idle after
    # the last process leaves; the readings are spaced for that.
    sleep 2
    sm=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')
    mem=$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits | head -1 | tr -d ' ')
    # Within a percent: the lock snaps to the nearest clock the card
    # supports (2400 asked for may read 2407), and the memory clock's top
    # state reads a step below its listed maximum (13801 against 14001).
    if (( sm * 100 < GPU_CLOCK * 99 || sm * 100 > GPU_CLOCK * 101 )); then
      echo "the GPU's SM clock reads $sm MHz, not the lock of $GPU_CLOCK MHz, and every timed run needs the clock locked:" >&2
      gpu_clock_lock_commands >&2
      return 1
    fi
    if (( mem * 100 < maxmem * 98 )); then
      echo "the GPU's memory clock reads $mem MHz, below its maximum of $maxmem MHz; lock it alongside the SM clock:" >&2
      gpu_clock_lock_commands >&2
      return 1
    fi
  done
  GPU_CLOCK_NOTE="GPU clocks locked: $sm MHz SM, $mem MHz memory"
  return 0
}
