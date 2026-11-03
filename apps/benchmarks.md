# The paper's benchmarks: what each bar is

Where each schedule of the paper's figures lives (11-01): a bar's label, then
the schedule file. pbrt's own renderers and the other references are not
schedules of ours. Each app's `plots/<figure>.txt` says what every bar
measured, and `apps/benchmark.sh` remakes every figure from the current
compiler in one go (apps/pbrt/figures.sh holds the pbrt figures' scenes,
sides, sample counts and repeats).

## pbrt GPU sweep (`apps/pbrt/plots/pbrt-gpu-sweep.pdf`; figures.sh GPU_SIDES)

| bar | schedule |
|---|---|
| wf | apps/pbrt/schedules/gpu-optix.bonsai (pbrt --gpu's wavefront, matched) |
| wf-fuse-mat | apps/pbrt/schedules/gpu-optix-fused-route.bonsai |
| wf-bounce | apps/pbrt/schedules/gpu-optix-bounce.bonsai |
| mk-flat | apps/pbrt/schedules/gpu-optix-mega-flat-samples.bonsai |
| mk-flat-ser | apps/pbrt/schedules/gpu-optix-mega-ser-flat-samples.bonsai |
| mk-flat-ser-ray | apps/pbrt/schedules/gpu-optix-mega-ser-ray-flat-samples.bonsai |
| mk-flat-ser-ray-px | apps/pbrt/schedules/gpu-optix-mega-ser-ray-flat.bonsai |

Every GPU schedule imports apps/pbrt/schedules/trees/optix.bonsai.

## pbrt CPU sweep (`apps/pbrt/plots/pbrt-cpu-sweep.pdf`; figures.sh CPU_SIDES)

| bar | schedule |
|---|---|
| wf-b-s-sp | apps/pbrt/schedules/wavefront-frame.bonsai (pbrt --wavefront, matched) |
| wf-t-v-ps | apps/pbrt/schedules/wavefront-thread-packet.bonsai |
| wf-t256-v-ps | apps/pbrt/schedules/wavefront-thread-packet-chunk256.bonsai |
| wf-t-v-ps-fuse | apps/pbrt/schedules/wavefront-thread-packet-fused.bonsai |
| wf-t-v-ps-sort | apps/pbrt/schedules/wavefront-thread-packet-sort.bonsai |
| mk-b-s-ps | apps/pbrt/schedules/scalar.bonsai |
| mk-b-s-ps-tiled | apps/pbrt/schedules/scalar-tiled.bonsai (pbrt's megakernel, matched: 32x32 tiles) |
| PBRT mk | pbrt's CPU volpath integrator (not ours) |

Every CPU schedule imports apps/pbrt/schedules/trees/bvh.bonsai.

## pbrt GPU heatmap (`apps/pbrt/plots/pbrt-gpu-heatmap.pdf`)

gpu-optix (the match of pbrt --gpu's wavefront) against pbrt --gpu on every
scene the converter takes, at 16, 64 and 128 spp.

## rtq (`apps/rtq/plots/`; apps/rtq/plot.py, run by apps/rtq/benchmark.sh)

rtq-firsthit, rtq-anyhit, rtq-closestpoint, rtq-queries-cpu:

| bar | schedule |
|---|---|
| bonsai (Embree) | apps/rtq/schedules/embree.bonsai (Embree 4's schedule and layout, matched) |
| bonsai (FCPW) | apps/rtq/schedules/fcpw4w16.bonsai (FCPW's 4-wide, 16-leaf tree, matched) |
| tuned (--tuned) | apps/rtq/schedules/tuned.bonsai, apps/rtq/schedules/fcpw4w16t.bonsai |

rtq-fcpwb-cpu: fcpwb = apps/rtq/schedules/fcpwb.bonsai (FCPW's binary BVH,
its scalar build, matched); fcpws = apps/rtq/schedules/fcpws.bonsai (FCPW's
cone hierarchy, silhouettes).

rtq-fcpwb-gpu: fcpwb-gpu = apps/rtq/schedules/fcpwb-gpu.bonsai (against
FCPW's GPUScene); fcpws-gpu = apps/rtq/schedules/fcpws-gpu.bonsai.

Layouts under apps/rtq/schedules/layouts/, trees under
apps/rtq/schedules/trees/.

## wosx GPU, walk on spheres (`apps/wosx/plots/wosx-gpu-speedup.pdf`; plot.py GPU_DEFAULT_SCHEDULES)

| bar | schedule |
|---|---|
| persistent | apps/wosx/schedules/fcpwb-gpu.bonsai (WoSX's GPU solver, matched) |
| parallel | apps/wosx/schedules/fcpwb-gpu-parallel.bonsai |
| persistent-reorder | apps/wosx/schedules/fcpwb-gpu-reorder.bonsai |
| persistent-256k | apps/wosx/schedules/fcpwb-gpu-256k.bonsai |
| persistent-reorder-256k | apps/wosx/schedules/fcpwb-gpu-reorder-256k.bonsai |
| parallel-reorder-blk64 | apps/wosx/schedules/fcpwb-gpu-parallel-reorder-blk64.bonsai |
| optix-ser | apps/wosx/schedules/fcpwb-gpu-optix-ser.bonsai |
| defer-block | apps/wosx/schedules/fcpwb-gpu-defer-block.bonsai |

## wosx GPU, walk on stars (`apps/wosx/plots/wosx-stars-gpu-speedup.pdf`; plot.py STARS_GPU_DEFAULT_SCHEDULES)

The same eight labels, each file prefixed `wost-`:
apps/wosx/schedules/wost-fcpwb-gpu.bonsai, wost-fcpwb-gpu-parallel.bonsai,
wost-fcpwb-gpu-reorder.bonsai, wost-fcpwb-gpu-256k.bonsai,
wost-fcpwb-gpu-reorder-256k.bonsai, wost-fcpwb-gpu-parallel-reorder-blk64.bonsai,
wost-fcpwb-gpu-optix-ser.bonsai, wost-fcpwb-gpu-defer-block.bonsai.
Layouts under apps/wosx/schedules/layouts/.
